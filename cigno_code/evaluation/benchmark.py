"""
Phase 4: Run evaluation benchmarks.

Evaluates on:
  1. CodeSearchNet test set
  2. Cigno evaluation set (hand-curated)
  3. Teacher comparison (student vs teacher on same queries)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

from ..config import Config
from .metrics import compute_retrieval_metrics, RetrievalMetrics

logger = logging.getLogger(__name__)


def load_codesearchnet_eval(eval_path: Path) -> tuple[list[str], list[str], list[list[int]]]:
    """Load CodeSearchNet test pairs.

    Returns (queries, corpus, relevant_ids) where each query has
    exactly one relevant corpus document (its index).
    """
    queries = []
    corpus = []
    relevant_ids = []

    with open(eval_path / "codesearchnet_test.jsonl") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            queries.append(record["query"])
            corpus.append(record["code"])
            relevant_ids.append([i])  # 1:1 mapping

    return queries, corpus, relevant_ids


def load_cigno_eval(eval_path: Path) -> tuple[list[str], list[str], list[list[int]]]:
    """Load the hand-curated Cigno evaluation set.

    Expected format: {"query": "...", "relevant_codes": ["...", "..."]}
    All relevant codes are appended to corpus; each query points to its set.
    """
    queries = []
    corpus = []
    relevant_ids = []

    with open(eval_path / "cigno_eval.jsonl") as f:
        for line in f:
            record = json.loads(line)
            queries.append(record["query"])
            ids = []
            for code in record["relevant_codes"]:
                ids.append(len(corpus))
                corpus.append(code)
            relevant_ids.append(ids)

    return queries, corpus, relevant_ids


def run_evaluation(config: Config, model_path: str) -> dict[str, RetrievalMetrics]:
    """Run all evaluation benchmarks and return results."""
    eval_path = Path(config.data.eval_path)
    model = SentenceTransformer(model_path)

    results: dict[str, RetrievalMetrics] = {}

    # 1. CodeSearchNet
    csn_file = eval_path / "codesearchnet_test.jsonl"
    if csn_file.exists():
        logger.info("Evaluating on CodeSearchNet test set...")
        queries, corpus, rel_ids = load_codesearchnet_eval(eval_path)
        results["codesearchnet"] = compute_retrieval_metrics(
            model, queries, corpus, rel_ids, k=10
        )
        logger.info(f"  MRR@10: {results['codesearchnet'].mrr_at_k:.4f}")
        logger.info(f"  Recall@10: {results['codesearchnet'].recall_at_k:.4f}")
        logger.info(f"  NDCG@10: {results['codesearchnet'].ndcg_at_k:.4f}")
    else:
        logger.warning(f"CodeSearchNet test file not found: {csn_file}")

    # 2. Cigno evaluation set
    cigno_file = eval_path / "cigno_eval.jsonl"
    if cigno_file.exists():
        logger.info("Evaluating on Cigno evaluation set...")
        queries, corpus, rel_ids = load_cigno_eval(eval_path)
        results["cigno"] = compute_retrieval_metrics(
            model, queries, corpus, rel_ids, k=10
        )
        logger.info(f"  MRR@10: {results['cigno'].mrr_at_k:.4f}")
        logger.info(f"  Recall@10: {results['cigno'].recall_at_k:.4f}")
        logger.info(f"  NDCG@10: {results['cigno'].ndcg_at_k:.4f}")
    else:
        logger.warning(f"Cigno eval file not found: {cigno_file}")

    return results


def compare_with_teacher(
    config: Config, student_path: str
) -> dict[str, dict[str, RetrievalMetrics]]:
    """Run both teacher and student on the same eval sets and compare."""
    eval_path = Path(config.data.eval_path)

    logger.info("Loading student model...")
    student = SentenceTransformer(student_path)

    logger.info(f"Loading teacher model: {config.teacher.model_id}...")
    teacher = SentenceTransformer(config.teacher.model_id)

    comparison: dict[str, dict[str, RetrievalMetrics]] = {}

    csn_file = eval_path / "codesearchnet_test.jsonl"
    if csn_file.exists():
        queries, corpus, rel_ids = load_codesearchnet_eval(eval_path)

        logger.info("Evaluating student on CodeSearchNet...")
        student_metrics = compute_retrieval_metrics(student, queries, corpus, rel_ids, k=10)

        logger.info("Evaluating teacher on CodeSearchNet...")
        teacher_metrics = compute_retrieval_metrics(teacher, queries, corpus, rel_ids, k=10)

        comparison["codesearchnet"] = {
            "student": student_metrics,
            "teacher": teacher_metrics,
        }

        ratio = student_metrics.mrr_at_k / max(teacher_metrics.mrr_at_k, 1e-10)
        logger.info(f"  Student MRR@10: {student_metrics.mrr_at_k:.4f}")
        logger.info(f"  Teacher MRR@10: {teacher_metrics.mrr_at_k:.4f}")
        logger.info(f"  Ratio: {ratio:.2%}")
        if ratio >= config.evaluation.mrr_threshold:
            logger.info(f"  PASS (>= {config.evaluation.mrr_threshold:.0%} of teacher)")
        else:
            logger.warning(f"  FAIL (< {config.evaluation.mrr_threshold:.0%} of teacher)")

    return comparison
