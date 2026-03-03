"""Load training configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    languages: list[str] = field(
        default_factory=lambda: [
            "python", "typescript", "javascript", "go", "rust", "java", "c", "c++"
        ]
    )
    target_snippets: int = 1_000_000
    max_body_chars: int = 512
    max_seq_length: int = 256
    corpus_path: str = "./data/corpus"
    teacher_embeddings_path: str = "./data/teacher_embeddings"
    eval_path: str = "./data/eval"


@dataclass
class TeacherConfig:
    model_id: str = "jinaai/jina-embeddings-v2-base-code"
    dimensions: int = 768
    batch_size: int = 256


@dataclass
class StudentConfig:
    base_model_id: str = "nreimers/MiniLM-L6-H384-uncased"
    dimensions: int = 384


@dataclass
class TrainingStageConfig:
    num_epochs: int = 3
    batch_size: int = 256
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    eval_steps: int = 1000
    save_steps: int = 5000
    logging_steps: int = 100
    output_dir: str = "./output/stage_a"


@dataclass
class ExportConfig:
    onnx_output_dir: str = "./output/onnx"
    quantized_output_dir: str = "./output/onnx-q8"


@dataclass
class EvaluationConfig:
    codesearchnet_test_size: int = 10_000
    codesearchnet_dev_size: int = 10_000
    mrr_threshold: float = 0.85


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    stage_a: TrainingStageConfig = field(default_factory=TrainingStageConfig)
    stage_b: TrainingStageConfig = field(
        default_factory=lambda: TrainingStageConfig(
            num_epochs=1,
            batch_size=128,
            learning_rate=1e-5,
            eval_steps=500,
            save_steps=2000,
            logging_steps=50,
            output_dir="./output/stage_b",
        )
    )
    export: ExportConfig = field(default_factory=ExportConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _merge(target: Any, source: dict) -> None:
    """Recursively merge a dict into a dataclass."""
    for key, value in source.items():
        if hasattr(target, key) and isinstance(value, dict):
            _merge(getattr(target, key), value)
        elif hasattr(target, key):
            setattr(target, key, value)


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load config from YAML, falling back to defaults for missing keys."""
    config = Config()
    path = Path(path)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        _merge(config, raw)
    return config
