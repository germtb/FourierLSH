#!/usr/bin/env python3
"""
Test: does recall keep improving as we add more bits beyond dim?

With multi-round encoding, FourierLSH automatically runs multiple FFT
passes with independent sign flips when bits > dim. From the caller's
perspective it's just more bits — same as standard LSH.
"""

import time
import numpy as np
from fourierlsh import FourierLSH


def brute_force_knn(data, queries, k):
    n_queries = queries.shape[0]
    indices = np.empty((n_queries, k), dtype=np.int32)
    for i, q in enumerate(queries):
        dists = np.linalg.norm(data - q, axis=1)
        indices[i] = np.argpartition(dists, k)[:k]
        indices[i] = indices[i][np.argsort(dists[indices[i]])]
    return indices


def recall_at_k(gt, predicted, k):
    recalls = []
    for i in range(gt.shape[0]):
        gt_set = set(gt[i, :k])
        pred_set = set(predicted[i, :k])
        recalls.append(len(gt_set & pred_set) / k)
    return float(np.mean(recalls))


def run_test(data, queries, gt, bits, k=100, n_candidates=500):
    lsh = FourierLSH(bits=bits, seed=0)

    t0 = time.perf_counter()
    db_hashes = lsh.encode_batch(data)
    encode_time = time.perf_counter() - t0

    n_queries = queries.shape[0]
    predicted = np.empty((n_queries, k), dtype=np.int32)
    t0 = time.perf_counter()
    for i, q in enumerate(queries):
        q_hash = lsh.encode(q)
        dists = lsh.hamming_batch(q_hash, db_hashes)
        cand_count = min(n_candidates, len(data))
        cand_idx = np.argpartition(dists, cand_count)[:cand_count]
        cand_dists = np.linalg.norm(data[cand_idx] - q, axis=1)
        actual_k = min(k, len(cand_dists))
        if actual_k >= len(cand_dists):
            top_k = np.argsort(cand_dists)[:actual_k]
        else:
            top_k = np.argpartition(cand_dists, actual_k)[:actual_k]
            top_k = top_k[np.argsort(cand_dists[top_k])]
        result = cand_idx[top_k]
        if len(result) < k:
            result = np.pad(result, (0, k - len(result)), constant_values=-1)
        predicted[i] = result
    query_time = time.perf_counter() - t0

    rec = recall_at_k(gt, predicted, k)
    mem_per_vec = lsh.code_size
    return rec, encode_time, query_time, mem_per_vec


# --- Setup ---
dim = 300
n_points = 10_000
n_queries = 200
k = 100
n_candidates = 500

rng = np.random.RandomState(42)
data = rng.randn(n_points, dim).astype(np.float64)
data /= np.linalg.norm(data, axis=1, keepdims=True)
queries = rng.randn(n_queries, dim).astype(np.float64)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

print(f"dim={dim}, n={n_points}, queries={n_queries}, k={k}, candidates={n_candidates}")
print()
print("Computing ground truth...", flush=True)
gt = brute_force_knn(data, queries, k)
print("Done.\n")

print(f"{'Bits':>8}  {'Bytes/vec':>10}  {'Recall@100':>12}  "
      f"{'Encode(ms)':>12}  {'Query(ms)':>12}")
print("-" * 62)

for bits in [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]:
    rec, enc, qt, mem = run_test(data, queries, gt, bits,
                                  k=k, n_candidates=n_candidates)
    print(f"{bits:>8}  {mem:>10}  {rec:>12.4f}  "
          f"{enc*1000:>12.1f}  {qt*1000:>12.1f}")
