#!/usr/bin/env python3
"""Download and prepare CodeSearchNet data for evaluation and Stage B training.

Creates:
  - data/eval/codesearchnet_train.jsonl  (for Stage B contrastive fine-tuning)
  - data/eval/codesearchnet_dev.jsonl    (for hyperparameter tuning)
  - data/eval/codesearchnet_test.jsonl   (for final evaluation)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Prepare CodeSearchNet evaluation data")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    output_dir = Path(config.data.eval_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_size = config.evaluation.codesearchnet_dev_size
    test_size = config.evaluation.codesearchnet_test_size

    # CodeSearchNet has splits for: python, javascript, go, java, ruby, php
    languages = ["python", "javascript", "go", "java"]

    all_pairs = []
    for lang in languages:
        logger.info(f"Loading CodeSearchNet {lang}...")
        ds = load_dataset("code_search_net", lang, split="test", trust_remote_code=True)
        for example in ds:
            query = example.get("func_documentation_string", "").strip()
            code = example.get("func_code_string", "").strip()
            if query and code and len(query) > 10 and len(code) > 20:
                all_pairs.append({"query": query, "code": code, "language": lang})

    logger.info(f"Total pairs: {len(all_pairs)}")

    # Shuffle deterministically
    import random
    random.seed(42)
    random.shuffle(all_pairs)

    # Split
    test_pairs = all_pairs[:test_size]
    dev_pairs = all_pairs[test_size : test_size + dev_size]
    train_pairs = all_pairs[test_size + dev_size :]

    for name, pairs in [
        ("codesearchnet_test.jsonl", test_pairs),
        ("codesearchnet_dev.jsonl", dev_pairs),
        ("codesearchnet_train.jsonl", train_pairs),
    ]:
        path = output_dir / name
        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        logger.info(f"Wrote {len(pairs)} pairs to {path}")


if __name__ == "__main__":
    main()
