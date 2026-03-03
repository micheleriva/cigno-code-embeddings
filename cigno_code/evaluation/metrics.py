"""
Evaluation metrics for code retrieval.

Computes MRR@K, Recall@K, and NDCG@K on (query, code) pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    mrr_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    k: int
    n_queries: int


def compute_retrieval_metrics(
    model: SentenceTransformer,
    queries: list[str],
    corpus: list[str],
    relevant_ids: list[list[int]],
    k: int = 10,
    batch_size: int = 128,
) -> RetrievalMetrics:
    """Compute MRR@K, Recall@K, NDCG@K for a retrieval task.

    Args:
        model: The embedding model to evaluate.
        queries: List of query strings.
        corpus: List of corpus documents (code snippets).
        relevant_ids: For each query, the list of relevant corpus indices.
        k: Cutoff for metrics.
        batch_size: Encoding batch size.
    """
    logger.info(f"Encoding {len(queries)} queries...")
    query_embeddings = model.encode(
        queries, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )

    logger.info(f"Encoding {len(corpus)} corpus documents...")
    corpus_embeddings = model.encode(
        corpus, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )

    # Compute similarity matrix: (n_queries, n_corpus)
    similarities = np.dot(query_embeddings, corpus_embeddings.T)

    # Top-K indices per query
    top_k_indices = np.argsort(-similarities, axis=1)[:, :k]

    mrr_sum = 0.0
    recall_sum = 0.0
    ndcg_sum = 0.0

    for i, relevant in enumerate(relevant_ids):
        relevant_set = set(relevant)
        retrieved = top_k_indices[i]

        # MRR: reciprocal rank of first relevant result
        for rank, idx in enumerate(retrieved, 1):
            if idx in relevant_set:
                mrr_sum += 1.0 / rank
                break

        # Recall@K: fraction of relevant docs in top-K
        hits = sum(1 for idx in retrieved if idx in relevant_set)
        recall_sum += hits / max(len(relevant_set), 1)

        # NDCG@K
        dcg = 0.0
        for rank, idx in enumerate(retrieved, 1):
            if idx in relevant_set:
                dcg += 1.0 / np.log2(rank + 1)
        # Ideal DCG
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant_set), k) + 1))
        ndcg_sum += dcg / max(idcg, 1e-10)

    n = len(queries)
    return RetrievalMetrics(
        mrr_at_k=mrr_sum / n,
        recall_at_k=recall_sum / n,
        ndcg_at_k=ndcg_sum / n,
        k=k,
        n_queries=n,
    )
