"""
Phase 3: Knowledge distillation training.

Stage A: MSE loss — train student to reproduce teacher embeddings.
         A Dense(384→768) projection layer is added during training to match
         the teacher's dimension, then removed after training so the student
         ships at native 384d.
Stage B: Contrastive loss — fine-tune on CodeSearchNet (query, code) pairs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    models,
)

from ..config import Config

logger = logging.getLogger(__name__)


def _load_corpus_dataset(corpus_path: Path, embeddings_path: Path, dimensions: int) -> Dataset:
    """Load corpus texts + teacher embeddings as a HuggingFace Dataset."""
    texts: list[str] = []
    with open(corpus_path / "corpus.jsonl") as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    n = len(texts)
    embeddings = np.memmap(
        embeddings_path / "teacher_embeddings.npy",
        dtype="float32",
        mode="r",
        shape=(n, dimensions),
    )

    # Copy memmap into a regular numpy array (2.7GB for 880K×768),
    # then let HF Dataset wrap it with Arrow zero-copy instead of
    # converting to Python lists (which would use ~30GB of RAM).
    labels = np.array(embeddings)

    return Dataset.from_dict({
        "sentence": texts,
        "label": labels,
    })


def _build_student_for_distillation(
    base_model_id: str,
    student_dim: int,
    teacher_dim: int,
    max_seq_length: int,
) -> SentenceTransformer:
    """Build student model with a Dense projection layer for MSE distillation.

    The projection maps student_dim → teacher_dim during training.
    It gets removed after training so the model outputs native student_dim.
    """
    transformer = models.Transformer(base_model_id, max_seq_length=max_seq_length)
    pooling = models.Pooling(transformer.get_word_embedding_dimension())
    projection = models.Dense(
        in_features=student_dim,
        out_features=teacher_dim,
        activation_function=torch.nn.Identity(),
    )
    return SentenceTransformer(modules=[transformer, pooling, projection])


def _save_student_without_projection(student: SentenceTransformer, save_path: str) -> None:
    """Save the student model without the Dense projection layer.

    The saved model outputs native 384d embeddings.
    """
    # Remove the last module (Dense projection)
    transformer = student[0]
    pooling = student[1]
    clean_student = SentenceTransformer(modules=[transformer, pooling])
    clean_student.save(save_path)


def _get_training_args(config: Config, stage_config) -> SentenceTransformerTrainingArguments:
    """Build training arguments, auto-disabling fp16 on MPS."""
    use_fp16 = stage_config.fp16
    if use_fp16 and not torch.cuda.is_available():
        logger.info("CUDA not available, disabling fp16")
        use_fp16 = False

    return SentenceTransformerTrainingArguments(
        output_dir=stage_config.output_dir,
        num_train_epochs=stage_config.num_epochs,
        per_device_train_batch_size=stage_config.batch_size,
        learning_rate=stage_config.learning_rate,
        warmup_ratio=stage_config.warmup_ratio,
        weight_decay=stage_config.weight_decay,
        fp16=use_fp16,
        logging_steps=stage_config.logging_steps,
        save_steps=stage_config.save_steps,
        save_total_limit=3,
    )


def train_stage_a(config: Config) -> str:
    """Stage A: MSE distillation from teacher embeddings.

    Returns path to the best model checkpoint (384d, no projection).
    """
    logger.info("=== Stage A: MSE Distillation ===")

    student = _build_student_for_distillation(
        base_model_id=config.student.base_model_id,
        student_dim=config.student.dimensions,
        teacher_dim=config.teacher.dimensions,
        max_seq_length=config.data.max_seq_length,
    )

    logger.info(f"Student: {config.student.base_model_id} ({config.student.dimensions}d)")
    logger.info(f"Projection: {config.student.dimensions}→{config.teacher.dimensions} (training only)")

    train_ds = _load_corpus_dataset(
        corpus_path=Path(config.data.corpus_path),
        embeddings_path=Path(config.data.teacher_embeddings_path),
        dimensions=config.teacher.dimensions,
    )
    logger.info(f"Training samples: {len(train_ds)}")

    train_loss = losses.MSELoss(model=student)
    args = _get_training_args(config, config.stage_a)

    trainer = SentenceTransformerTrainer(
        model=student,
        args=args,
        train_dataset=train_ds,
        loss=train_loss,
    )

    trainer.train()

    best_path = str(Path(config.stage_a.output_dir) / "final")
    _save_student_without_projection(student, best_path)
    logger.info(f"Stage A complete. Model saved to {best_path} ({config.student.dimensions}d)")
    return best_path


def train_stage_b(config: Config, stage_a_model_path: str) -> str:
    """Stage B: Contrastive fine-tuning on CodeSearchNet pairs.

    Returns path to the final model.
    """
    logger.info("=== Stage B: Contrastive Fine-tuning ===")

    student = SentenceTransformer(stage_a_model_path)

    # Load CodeSearchNet pairs
    eval_path = Path(config.data.eval_path)
    pairs_file = eval_path / "codesearchnet_train.jsonl"

    queries = []
    codes = []
    with open(pairs_file) as f:
        for line in f:
            record = json.loads(line)
            queries.append(record["query"])
            codes.append(record["code"])

    train_ds = Dataset.from_dict({
        "anchor": queries,
        "positive": codes,
    })
    logger.info(f"Training pairs: {len(train_ds)}")

    train_loss = losses.MultipleNegativesRankingLoss(model=student)
    args = _get_training_args(config, config.stage_b)

    trainer = SentenceTransformerTrainer(
        model=student,
        args=args,
        train_dataset=train_ds,
        loss=train_loss,
    )

    trainer.train()

    best_path = str(Path(config.stage_b.output_dir) / "final")
    student.save(best_path)
    logger.info(f"Stage B complete. Model saved to {best_path}")
    return best_path
