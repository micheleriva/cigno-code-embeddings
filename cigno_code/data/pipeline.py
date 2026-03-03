"""
Data pipeline: extract symbols from StarCoderData and format for training.

This is the Phase 1 entry point. It:
  1. Streams code files from bigcode/starcoderdata (via HuggingFace datasets)
  2. Extracts symbols using tree-sitter
  3. Formats them using the same text format as Cigno's embedder
  4. Saves the corpus to disk as JSONL

Dataset: bigcode/starcoderdata — pure parquet, 86 languages, permissive license.
Requires accepting BigCode terms at https://huggingface.co/datasets/bigcode/starcoderdata
and running `huggingface-cli login`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from .extractor import extract_symbols
from .text_formatter import format_embedding_text
from ..config import Config

logger = logging.getLogger(__name__)

# Our language keys → starcoderdata data_dir names
LANGUAGE_TO_DATA_DIR = {
    "python": "python",
    "typescript": "typescript",
    "javascript": "javascript",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "c": "c",
    "c++": "cpp",
    "cpp": "cpp",
    "ruby": "ruby",
    "php": "php",
    "csharp": "c-sharp",
    "kotlin": "kotlin",
    "swift": "swift",
    "scala": "scala",
    "shell": "shell",
    "lua": "lua",
    "haskell": "haskell",
    "julia": "julia",
    "elixir": "elixir",
    "ocaml": "ocaml",
    "zig": "zig",
    "fortran": "fortran",
    "sql": "sql",
    "powershell": "powershell",
    "commonlisp": "common-lisp",
}


def _load_language_stream(language: str):
    """Load a streaming dataset for a single language from starcoderdata."""
    data_dir = LANGUAGE_TO_DATA_DIR[language]
    return load_dataset(
        "bigcode/starcoderdata",
        data_dir=data_dir,
        split="train",
        streaming=True,
    )


def extract_corpus(config: Config, **kwargs) -> None:
    """Stream starcoderdata, extract symbols, save as JSONL.

    Output format (one JSON object per line):
        {"text": "<formatted embedding text>", "language": "python", "symbol_type": "function"}
    """
    output_dir = Path(config.data.corpus_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "corpus.jsonl"

    target = config.data.target_snippets
    per_language = target // len(config.data.languages)

    counts: dict[str, int] = {lang: 0 for lang in config.data.languages}
    total = 0

    logger.info(f"Dataset: bigcode/starcoderdata")
    logger.info(f"Extracting {target} snippets ({per_language} per language)")
    logger.info(f"Output: {output_file}")

    with open(output_file, "w") as f:
        for language in config.data.languages:
            if language not in LANGUAGE_TO_DATA_DIR:
                logger.warning(f"Skipping unsupported language: {language}")
                continue
            if counts.get(language, 0) >= per_language:
                continue

            logger.info(f"Streaming {language}...")

            try:
                ds = _load_language_stream(language)
            except Exception as e:
                logger.error(f"Failed to load {language}: {e}")
                continue

            pbar = tqdm(
                desc=language,
                total=per_language,
                initial=counts.get(language, 0),
            )

            for example in ds:
                if counts[language] >= per_language:
                    break

                content = example.get("content", "")
                if not content or len(content) < 50:
                    continue

                symbols = extract_symbols(
                    source_code=content,
                    language=language,
                    file_path=example.get("max_stars_repo_path"),
                )

                for symbol in symbols:
                    if counts[language] >= per_language:
                        break

                    text = format_embedding_text(symbol)
                    if len(text) < 30:
                        continue

                    record = {
                        "text": text,
                        "language": language,
                        "symbol_type": symbol.symbol_type,
                    }
                    f.write(json.dumps(record) + "\n")
                    counts[language] += 1
                    total += 1
                    pbar.update(1)

            pbar.close()
            logger.info(f"{language}: {counts[language]} snippets extracted")

    logger.info(f"Total: {total} snippets saved to {output_file}")
    logger.info(f"Per-language breakdown: {counts}")
