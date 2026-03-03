#!/usr/bin/env python3
"""Phase 4: Run evaluation benchmarks."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.evaluation.benchmark import run_evaluation, compare_with_teacher


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model")
    parser.add_argument("model_path", help="Path to the model to evaluate")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--compare-teacher", action="store_true", help="Also evaluate teacher")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)
    results = run_evaluation(config, args.model_path)

    if args.compare_teacher:
        comparison = compare_with_teacher(config, args.model_path)
        results["comparison"] = comparison

    if args.output:
        # Serialize dataclasses to dicts
        serializable = {}
        for key, val in results.items():
            if hasattr(val, "__dataclass_fields__"):
                serializable[key] = {k: getattr(val, k) for k in val.__dataclass_fields__}
            else:
                serializable[key] = val

        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logging.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
