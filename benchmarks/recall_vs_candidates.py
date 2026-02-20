#!/usr/bin/env python3
"""
Recall vs Candidates Benchmark
===============================

Shows how Recall@100 scales with the number of Hamming candidates
retrieved at a fixed bit width (256). This directly addresses the
precision/recall tradeoff: more candidates = higher recall but
slower reranking.

No extra dependencies beyond numpy.

Usage:
    python benchmarks/recall_vs_candidates.py
"""

import os
import time

import numpy as np

from fourierlsh import FourierLSH


GLOVE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "glove.6B.300d.txt",
)


# ---------------------------------------------------------------------------
# Helpers (self-contained — no cross-import needed)
# ---------------------------------------------------------------------------

def load_glove(path: str, n_vectors: int) -> np.ndarray | None:
    if not os.path.isfile(path):
        return None
    vectors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(vectors) >= n_vectors:
                break
            parts = line.rstrip().split(" ")
            vectors.append([float(x) for x in parts[1:]])
    data = np.array(vectors, dtype=np.float64)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    data /= norms
    return data


def brute_force_knn(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    n_queries = queries.shape[0]
    indices = np.empty((n_queries, k), dtype=np.int32)
    for i, q in enumerate(queries):
        dists = np.linalg.norm(data - q, axis=1)
        indices[i] = np.argpartition(dists, k)[:k]
        indices[i] = indices[i][np.argsort(dists[indices[i]])]
    return indices


def recall_at_k(ground_truth: np.ndarray, predicted: np.ndarray, k: int) -> float:
    n_queries = ground_truth.shape[0]
    recalls = []
    for i in range(n_queries):
        gt_set = set(ground_truth[i, :k])
        pred_set = set(predicted[i, :k])
        recalls.append(len(gt_set & pred_set) / k)
    return float(np.mean(recalls))


def run_recall_at_candidates(db_hashes, lsh, data, queries, ground_truth,
                              n_candidates: int, k: int = 100):
    """Run ANN search and return recall for a given candidate count."""
    n_queries = queries.shape[0]
    predicted = np.empty((n_queries, k), dtype=np.int32)

    for i, q in enumerate(queries):
        q_hash = lsh.encode(q)
        hamming = lsh.hamming_batch(q_hash, db_hashes)
        cand_count = min(n_candidates, len(data))
        cand_idx = np.argpartition(hamming, cand_count)[:cand_count]
        cand_dists = np.linalg.norm(data[cand_idx] - q, axis=1)
        actual_k = min(k, len(cand_dists))
        if actual_k >= len(cand_dists):
            top_k_local = np.argsort(cand_dists)[:actual_k]
        else:
            top_k_local = np.argpartition(cand_dists, actual_k)[:actual_k]
            top_k_local = top_k_local[np.argsort(cand_dists[top_k_local])]
        result = cand_idx[top_k_local]
        if len(result) < k:
            result = np.pad(result, (0, k - len(result)), constant_values=-1)
        predicted[i] = result

    return recall_at_k(ground_truth, predicted, k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark():
    bits = 256
    k = 100
    candidate_counts = [100, 200, 500, 1000, 2000]

    # Pick dataset
    glove_total = 10_000 + 1_000
    glove_vectors = load_glove(GLOVE_PATH, glove_total)

    if glove_vectors is not None:
        dataset_name = "GloVe 6B 300D (10K db, 1K queries)"
        data = glove_vectors[:10_000]
        queries = glove_vectors[10_000:glove_total]
    else:
        dataset_name = "Synthetic (random Gaussian, 300D, 5K db)"
        dim = 300
        rng = np.random.RandomState(42)
        data = rng.randn(5000, dim).astype(np.float64)
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        queries = rng.randn(100, dim).astype(np.float64)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    print("=" * 52)
    print("Recall@100 vs Number of Candidates")
    print("=" * 52)
    print(f"  Dataset:    {dataset_name}")
    print(f"  Hash bits:  {bits}")
    print(f"  k:          {k}")
    print()

    print("Computing ground truth (brute force)...", flush=True)
    gt = brute_force_knn(data, queries, k)
    print("Done.\n")

    lsh = FourierLSH(bits=bits)
    db_hashes = lsh.encode_batch(data)

    print(f"{'Candidates':>12}  {'Recall@100':>10}  {'Rerank ratio':>12}")
    print("-" * 40)

    for nc in candidate_counts:
        rec = run_recall_at_candidates(db_hashes, lsh, data, queries, gt,
                                       n_candidates=nc, k=k)
        ratio = nc / len(data)
        print(f"{nc:>12}  {rec:>10.4f}  {ratio:>11.1%}")

    print("-" * 40)
    print()
    print("More candidates = higher recall but more reranking work.")
    print("FourierLSH narrows the search space; exact reranking finishes the job.")


if __name__ == "__main__":
    run_benchmark()
