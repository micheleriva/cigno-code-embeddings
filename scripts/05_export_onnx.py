#!/usr/bin/env python3
"""Phase 5: Export to ONNX and quantize."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.export.onnx import export_to_onnx, quantize_onnx, verify_onnx


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX and quantize")
    parser.add_argument("model_path", help="Path to the trained model")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--skip-quantize", action="store_true", help="Skip INT8 quantization")
    parser.add_argument("--verify", action="store_true", help="Run verification benchmarks")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)

    export_to_onnx(config, args.model_path)

    if not args.skip_quantize:
        quantize_onnx(config)

    if args.verify:
        verify_onnx(config)


if __name__ == "__main__":
    main()
