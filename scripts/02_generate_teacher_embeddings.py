#!/usr/bin/env python3
"""Phase 2: Generate teacher embeddings for the training corpus."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cigno_code.config import load_config
from cigno_code.data.teacher import generate_teacher_embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate teacher embeddings")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)
    generate_teacher_embeddings(config)


if __name__ == "__main__":
    main()
