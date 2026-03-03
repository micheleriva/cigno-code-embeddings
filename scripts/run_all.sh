#!/bin/bash
# Full training pipeline for cigno-code-small-v1
# Run this on RunPod with: nohup bash scripts/run_all.sh > training.log 2>&1 &
set -e

echo "=== cigno-code-small-v1 training pipeline ==="
echo "Started at $(date)"

# Phase 1: Extract corpus
echo ""
echo "=== Phase 1: Extracting corpus ==="
uv run python scripts/01_extract_corpus.py --target 1000000

# Prep CodeSearchNet
echo ""
echo "=== Preparing CodeSearchNet data ==="
uv run python scripts/prep_codesearchnet.py

# Phase 2: Teacher embeddings
echo ""
echo "=== Phase 2: Generating teacher embeddings ==="
uv run python scripts/02_generate_teacher_embeddings.py

# Phase 3: Training
echo ""
echo "=== Phase 3A: MSE Distillation ==="
uv run python scripts/03_train.py --stage a

echo ""
echo "=== Phase 3B: Contrastive Fine-tuning ==="
uv run python scripts/03_train.py --stage b --stage-a-model output/stage_a/final

# Phase 4: Evaluation
echo ""
echo "=== Phase 4: Evaluation ==="
uv run python scripts/04_evaluate.py output/stage_b/final --compare-teacher --output output/eval_results.json

echo ""
echo "=== Done ==="
echo "Finished at $(date)"
echo "Model saved to: output/stage_b/final"
echo "Eval results: output/eval_results.json"
