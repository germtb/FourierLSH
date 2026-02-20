#!/usr/bin/env python3
"""
Isolated Encoding Benchmark
============================

Fair comparison of FourierLSH encoding vs FAISS LSH encoding vs WHT (Hadamard).
- Encoding, Hamming search, and reranking timed separately.
- Warmup iterations before timing.
- Three encoders: FourierLSH (Rust FFT + rayon), WHT (Rust Hadamard + rayon), FAISS (BLAS matmul).
- Hamming search: Rust NEON+rayon vs FAISS IndexBinaryFlat.
- End-to-end: all use FAISS IndexBinaryFlat for search (isolates encoding).

Usage:
    python benchmarks/encoding_isolated.py
"""

import time
import numpy as np

from fourierlsh import FourierLSH
from fourierlsh._native import hadamard_encode as _hadamard_encode_native

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("ERROR: faiss-cpu required. Install with: uv sync --group bench")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_fn(fn, warmup=2, repeats=5):
    """Time a function with warmup. Returns median time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times)


# ---------------------------------------------------------------------------
# Encoding benchmarks
# ---------------------------------------------------------------------------

def bench_fourier_encode(data, bits):
    """FourierLSH encoding via Rust rustfft + rayon."""
    lsh = FourierLSH(bits=bits, seed=0)
    def encode():
        return lsh.encode_batch(data)
    t = time_fn(encode)
    codes = lsh.encode_batch(data)
    return codes, t


def bench_hadamard_encode(data, bits):
    """WHT encoding via Rust (rayon-parallel). Same structure as FALCONN."""
    vecs = np.ascontiguousarray(data, dtype=np.float32)
    def encode():
        return np.asarray(_hadamard_encode_native(vecs, bits, 0))
    t = time_fn(encode)
    codes = np.asarray(_hadamard_encode_native(vecs, bits, 0))
    return codes, t


def bench_faiss_encode(data, bits):
    """FAISS LSH encoding (BLAS matmul)."""
    dim = data.shape[1]
    vecs = np.ascontiguousarray(data, dtype=np.float32)
    def encode():
        idx = faiss.IndexLSH(dim, bits)
        idx.add(vecs)
        n = idx.ntotal
        code_size = idx.code_size
        codes = faiss.rev_swig_ptr(idx.codes.data(), n * code_size)
        return np.array(codes, dtype=np.uint8).reshape(n, code_size)
    t = time_fn(encode)
    codes = encode()
    return codes, t


# ---------------------------------------------------------------------------
# Hamming search benchmarks
# ---------------------------------------------------------------------------

def bench_hamming_search_faiss(q_codes, db_codes, bits, n_candidates):
    binary_index = faiss.IndexBinaryFlat(bits)
    binary_index.add(np.ascontiguousarray(db_codes))
    q = np.ascontiguousarray(q_codes)
    def search():
        return binary_index.search(q, n_candidates)
    t = time_fn(search)
    _, cand_indices = binary_index.search(q, n_candidates)
    return cand_indices, t


def bench_hamming_search_rust(lsh, q_codes, db_codes, n_candidates):
    q = np.ascontiguousarray(q_codes)
    db = np.ascontiguousarray(db_codes)
    def search():
        return lsh.hamming_top_k(q, db, n_candidates)
    t = time_fn(search)
    cand_indices = lsh.hamming_top_k(q, db, n_candidates)
    return cand_indices, t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rerank(data, queries, cand_indices, k):
    n_queries = queries.shape[0]
    predicted = np.empty((n_queries, k), dtype=np.int32)
    for i in range(n_queries):
        cand_idx = cand_indices[i]
        cand_idx = cand_idx[cand_idx >= 0]
        cand_dists = np.linalg.norm(data[cand_idx] - queries[i], axis=1)
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
    return predicted


def recall_at_k(ground_truth, predicted, k):
    n_queries = ground_truth.shape[0]
    recalls = []
    for i in range(n_queries):
        gt_set = set(ground_truth[i, :k])
        pred_set = set(predicted[i, :k])
        recalls.append(len(gt_set & pred_set) / k)
    return float(np.mean(recalls))


def brute_force_knn(data, queries, k):
    n_queries = queries.shape[0]
    indices = np.empty((n_queries, k), dtype=np.int32)
    for i, q in enumerate(queries):
        dists = np.linalg.norm(data - q, axis=1)
        indices[i] = np.argpartition(dists, k)[:k]
        indices[i] = indices[i][np.argsort(dists[indices[i]])]
    return indices


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def bench_hnsw(data, queries, k, ef_search_values):
    """Build HNSW index and search at various efSearch. Returns list of (label, build_ms, query_ms, recall)."""
    dim = data.shape[1]
    vecs = np.ascontiguousarray(data, dtype=np.float32)
    qvecs = np.ascontiguousarray(queries, dtype=np.float32)

    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200

    t0 = time.perf_counter()
    index.add(vecs)
    build_time = time.perf_counter() - t0

    results = []
    for ef in ef_search_values:
        index.hnsw.efSearch = ef
        def search(ef=ef):
            index.hnsw.efSearch = ef
            return index.search(qvecs, k)
        qt = time_fn(search)
        _, indices = index.search(qvecs, k)
        results.append((f"HNSW ef={ef}", build_time, qt, indices))
    return results


def run_encoding_benchmark(name, data, queries, bit_widths, n_candidates=500, k=100):
    dim = data.shape[1]
    n_db = data.shape[0]
    n_queries = queries.shape[0]

    print("=" * 90)
    print(f"Dataset: {name}")
    print(f"  {n_db} vectors, {n_queries} queries, dim={dim}")
    print("=" * 90)

    # Ground truth
    print("Computing ground truth...", flush=True)
    gt = brute_force_knn(data, queries, k)
    print("Done.\n")

    # --- Part 1: Encoding time comparison ---
    print("PART 1: ENCODING TIME (FFT vs WHT vs dense projection)")
    print("-" * 90)
    print(f"{'Bits':>6}  {'FFT(Rust)':>12}  {'WHT(Rust)':>12}  {'FAISS BLAS':>12}  "
          f"{'FFT/FAISS':>10}  {'WHT/FAISS':>10}  {'FFT/WHT':>8}")
    print("-" * 90)

    for bits in bit_widths:
        _, fft_time = bench_fourier_encode(data, bits)
        _, wht_time = bench_hadamard_encode(data, bits)
        _, fx_time = bench_faiss_encode(data, bits)
        fft_ratio = fft_time / fx_time
        wht_ratio = wht_time / fx_time
        fft_wht = fft_time / wht_time
        print(f"{bits:>6}  {fft_time*1000:>10.1f}ms  {wht_time*1000:>10.1f}ms  "
              f"{fx_time*1000:>10.1f}ms  {fft_ratio:>9.2f}x  {wht_ratio:>9.2f}x  {fft_wht:>7.2f}x")

    print()

    # --- Part 2: Hamming search comparison ---
    print("PART 2: HAMMING SEARCH (Rust NEON+rayon vs FAISS SIMD)")
    print("-" * 90)
    bits_for_search = [b for b in [256, 1024, 4096] if b in bit_widths]
    print(f"{'Bits':>6}  {'Rust NEON':>12}  {'FAISS BinFlat':>14}  {'Ratio':>8}")
    print("-" * 90)

    for bits in bits_for_search:
        lsh = FourierLSH(bits=bits, seed=0)
        db_codes = lsh.encode_batch(data)
        q_codes = lsh.encode_batch(queries)

        _, rust_time = bench_hamming_search_rust(lsh, q_codes, db_codes, n_candidates)
        _, faiss_time = bench_hamming_search_faiss(q_codes, db_codes, bits, n_candidates)
        ratio = rust_time / faiss_time
        print(f"{bits:>6}  {rust_time*1000:>10.1f}ms  {faiss_time*1000:>12.1f}ms  {ratio:>7.2f}x")

    print()

    # --- Part 3: End-to-end with SAME search backend (FAISS BinaryFlat) ---
    print("PART 3: END-TO-END RECALL (all use FAISS search — isolates encoding quality)")
    print("-" * 90)
    print(f"{'Bits':>6}  {'Method':<16}  {'Encode':>10}  {'Search':>10}  "
          f"{'Total':>10}  {'Recall@100':>10}")
    print("-" * 90)

    for bits in bit_widths:
        results = []

        for label, encode_fn in [
            ("FFT (Rust)",  lambda: bench_fourier_encode(data, bits)),
            ("WHT (Rust)",  lambda: bench_hadamard_encode(data, bits)),
            ("FAISS LSH",   lambda: bench_faiss_encode(data, bits)),
        ]:
            codes, enc_time = encode_fn()

            # Encode queries with matching encoder
            if "FFT" in label:
                q_codes = FourierLSH(bits=bits, seed=0).encode_batch(queries)
            elif "WHT" in label:
                q_codes = np.asarray(_hadamard_encode_native(
                    np.ascontiguousarray(queries, dtype=np.float32), bits, 0))
            else:
                # FAISS query encoding
                faiss_lsh = faiss.IndexLSH(dim, bits)
                faiss_lsh.add(np.ascontiguousarray(queries, dtype=np.float32))
                q_codes = np.array(
                    faiss.rev_swig_ptr(faiss_lsh.codes.data(), queries.shape[0] * faiss_lsh.code_size),
                    dtype=np.uint8
                ).reshape(queries.shape[0], faiss_lsh.code_size)
                faiss_lsh.reset()

            cands, search_time = bench_hamming_search_faiss(q_codes, codes, bits, n_candidates)
            pred = rerank(data, queries, cands, k)
            rec = recall_at_k(gt, pred, k)

            print(f"{bits:>6}  {label:<16}  {enc_time*1000:>8.1f}ms  {search_time*1000:>8.1f}ms  "
                  f"{(enc_time+search_time)*1000:>8.1f}ms  {rec:>10.4f}")

        print()

    # --- Part 4: vs HNSW (the real competitor) ---
    print("PART 4: FourierLSH vs FAISS HNSW — full pipeline")
    print("-" * 90)
    print(f"{'Method':<24}  {'Build':>10}  {'Query':>10}  {'Total':>10}  {'Recall@100':>10}")
    print("-" * 90)

    # FourierLSH at select bit widths, using Rust encode + Rust Hamming search
    for bits in [256, 512, 1024, 2048, 4096]:
        if bits not in bit_widths:
            continue
        lsh = FourierLSH(bits=bits, seed=0)

        # Build: encode database
        def do_encode(lsh=lsh):
            return lsh.encode_batch(data)
        build_time = time_fn(do_encode)
        db_codes = lsh.encode_batch(data)
        q_codes = lsh.encode_batch(queries)

        # Query: Rust Hamming top-k + rerank
        def do_query(lsh=lsh, q_codes=q_codes, db_codes=db_codes):
            return lsh.hamming_top_k(q_codes, db_codes, n_candidates)
        query_time = time_fn(do_query)
        cand_indices = lsh.hamming_top_k(q_codes, db_codes, n_candidates)

        t0 = time.perf_counter()
        pred = rerank(data, queries, cand_indices, k)
        rerank_time = time.perf_counter() - t0
        query_total = query_time + rerank_time

        rec = recall_at_k(gt, pred, k)
        label = f"FourierLSH {bits}b"
        print(f"{label:<24}  {build_time*1000:>8.1f}ms  {query_total*1000:>8.1f}ms  "
              f"{(build_time+query_total)*1000:>8.1f}ms  {rec:>10.4f}")

    # HNSW at various efSearch
    hnsw_results = bench_hnsw(data, queries, k, [32, 64, 128, 256, 512])
    for label, build_t, query_t, indices in hnsw_results:
        rec = recall_at_k(gt, indices, k)
        print(f"{label:<24}  {build_t*1000:>8.1f}ms  {query_t*1000:>8.1f}ms  "
              f"{(build_t+query_t)*1000:>8.1f}ms  {rec:>10.4f}")

    print()
    print("=" * 90)


def main():
    k = 100
    n_candidates = 500

    for dim, n_points, n_queries in [(128, 10_000, 200), (300, 10_000, 200), (1536, 10_000, 200)]:
        rng = np.random.RandomState(42)
        data = rng.randn(n_points, dim).astype(np.float64)
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        queries = rng.randn(n_queries, dim).astype(np.float64)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        bit_widths = [64, 128, 256, 512, 1024, 2048, 4096]
        run_encoding_benchmark(
            f"Synthetic Gaussian {dim}D ({n_points} db, {n_queries} queries)",
            data, queries, bit_widths, n_candidates=n_candidates, k=k,
        )
        print("\n\n")


if __name__ == "__main__":
    main()
