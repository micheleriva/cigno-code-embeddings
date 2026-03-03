#!/usr/bin/env python3
"""Phase 3: Run distillation training (Stage A + Stage B)."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cigno_code.config import load_config
from cigno_code.training.distillation import train_stage_a, train_stage_b


def main():
    parser = argparse.ArgumentParser(description="Run distillation training")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument(
        "--stage", choices=["a", "b", "both"], default="both", help="Which stage to run"
    )
    parser.add_argument(
        "--stage-a-model",
        help="Path to Stage A model (required if --stage=b)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)

    if args.stage in ("a", "both"):
        stage_a_path = train_stage_a(config)
    else:
        stage_a_path = args.stage_a_model
        if not stage_a_path:
            parser.error("--stage-a-model is required when --stage=b")

    if args.stage in ("b", "both"):
        train_stage_b(config, stage_a_path)


if __name__ == "__main__":
    main()
