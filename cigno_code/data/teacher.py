"""
Phase 2: Generate teacher embeddings for the training corpus.

Loads the teacher model (jina-embeddings-v2-base-code), encodes all corpus
snippets, and saves the resulting embeddings to disk as a numpy memmap file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..config import Config

logger = logging.getLogger(__name__)


def load_corpus_texts(corpus_path: Path) -> list[str]:
    """Load all text entries from the corpus JSONL."""
    texts = []
    with open(corpus_path / "corpus.jsonl") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
    return texts


def generate_teacher_embeddings(config: Config) -> None:
    """Encode the entire corpus with the teacher model and save to disk."""
    corpus_path = Path(config.data.corpus_path)
    output_path = Path(config.data.teacher_embeddings_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading corpus...")
    texts = load_corpus_texts(corpus_path)
    n = len(texts)
    logger.info(f"Corpus size: {n} snippets")

    logger.info(f"Loading teacher model: {config.teacher.model_id}")
    teacher = SentenceTransformer(config.teacher.model_id, trust_remote_code=True)
    # Cap sequence length to avoid OOM on long code snippets.
    # The student uses 256 tokens, so there's no benefit to encoding longer
    # sequences in the teacher — the extra tokens won't be seen by the student.
    teacher.max_seq_length = config.data.max_seq_length

    # Use memmap so we don't need 15GB of RAM
    emb_file = output_path / "teacher_embeddings.npy"
    embeddings = np.memmap(
        emb_file,
        dtype="float32",
        mode="w+",
        shape=(n, config.teacher.dimensions),
    )

    logger.info(f"Encoding {n} snippets with batch_size={config.teacher.batch_size}...")
    batch_size = config.teacher.batch_size
    for start in tqdm(range(0, n, batch_size), desc="Teacher encoding"):
        end = min(start + batch_size, n)
        batch = texts[start:end]
        batch_embeddings = teacher.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings[start:end] = batch_embeddings

    # Flush to disk
    embeddings.flush()

    # Save metadata
    meta = {"n": n, "dimensions": config.teacher.dimensions, "model_id": config.teacher.model_id}
    with open(output_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Teacher embeddings saved to {emb_file} ({emb_file.stat().st_size / 1e9:.2f} GB)")
