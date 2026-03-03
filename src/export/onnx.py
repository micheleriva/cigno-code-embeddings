"""
Phase 5: Export trained model to ONNX and quantize to INT8.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ..config import Config

logger = logging.getLogger(__name__)


def export_to_onnx(config: Config, model_path: str) -> str:
    """Export a sentence-transformers model to ONNX format.

    Returns the path to the ONNX model directory.
    """
    from optimum.exporters.onnx import main_export

    output_dir = config.export.onnx_output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting {model_path} to ONNX...")
    main_export(
        model_name_or_path=model_path,
        output=output_dir,
        task="feature-extraction",
    )

    logger.info(f"ONNX model saved to {output_dir}")
    return output_dir


def quantize_onnx(config: Config) -> str:
    """Quantize ONNX model to INT8 (dynamic quantization).

    Returns the path to the quantized model directory.
    """
    from optimum.onnxruntime import ORTQuantizer, AutoQuantizationConfig

    onnx_dir = config.export.onnx_output_dir
    output_dir = config.export.quantized_output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Quantizing {onnx_dir} to INT8...")
    quantizer = ORTQuantizer.from_pretrained(onnx_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
    quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)

    # Report file size
    model_file = Path(output_dir) / "model_quantized.onnx"
    if model_file.exists():
        size_mb = model_file.stat().st_size / 1e6
        logger.info(f"Quantized model size: {size_mb:.1f} MB")
    else:
        # Check for other onnx files
        for f in Path(output_dir).glob("*.onnx"):
            size_mb = f.stat().st_size / 1e6
            logger.info(f"Quantized model: {f.name} ({size_mb:.1f} MB)")

    logger.info(f"Quantized model saved to {output_dir}")
    return output_dir


def verify_onnx(config: Config) -> None:
    """Verify the quantized ONNX model: load time, inference speed, file size."""
    import onnxruntime as ort
    import numpy as np

    model_dir = Path(config.export.quantized_output_dir)
    model_files = list(model_dir.glob("*.onnx"))
    if not model_files:
        logger.error(f"No ONNX files found in {model_dir}")
        return

    model_file = model_files[0]

    # File size
    size_mb = model_file.stat().st_size / 1e6
    logger.info(f"Model file: {model_file.name}")
    logger.info(f"File size: {size_mb:.1f} MB (target: ≤25 MB)")

    # Load time
    t0 = time.perf_counter()
    session = ort.InferenceSession(str(model_file))
    load_time = time.perf_counter() - t0
    logger.info(f"Load time: {load_time:.2f}s (target: ≤1s)")

    # Inference speed — run 100 dummy inputs
    dummy_ids = np.ones((1, 64), dtype=np.int64)
    dummy_mask = np.ones((1, 64), dtype=np.int64)
    dummy_type = np.zeros((1, 64), dtype=np.int64)

    input_names = [inp.name for inp in session.get_inputs()]
    feed = {}
    for name in input_names:
        if "input_ids" in name:
            feed[name] = dummy_ids
        elif "attention_mask" in name:
            feed[name] = dummy_mask
        elif "token_type" in name:
            feed[name] = dummy_type

    # Warmup
    for _ in range(10):
        session.run(None, feed)

    # Benchmark
    n_runs = 100
    t0 = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, feed)
    avg_ms = (time.perf_counter() - t0) / n_runs * 1000

    logger.info(f"Inference: {avg_ms:.1f}ms per symbol (target: ≤5ms)")
