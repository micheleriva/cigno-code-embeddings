#!/usr/bin/env python3
"""Phase 1: Extract code symbols from bigcode/starcoderdata."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.pipeline import extract_corpus


def main():
    parser = argparse.ArgumentParser(description="Extract code symbols from starcoderdata")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--target", type=int, help="Override target snippet count")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)
    if args.target:
        config.data.target_snippets = args.target

    extract_corpus(config)


if __name__ == "__main__":
    main()
