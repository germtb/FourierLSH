#!/usr/bin/env python3
"""
Recall & Performance Benchmark
================================

Compares FourierLSH (FFT + random sign flips) against FAISS LSH and FAISS HNSW
at various bit-widths / efSearch values. Measures both recall and query throughput.

Datasets:
    - Synthetic: 5000 random Gaussian vectors in 300D (always runs)
    - GloVe:     First 10K vectors from glove.6B.300d.txt (if available)
    - OpenAI:    10K text-embedding-3-small (1536D) from HuggingFace (if datasets installed)

Usage:
    python benchmarks/recall_vs_metadata.py
"""

import os
import time

import numpy as np

from fourierlsh import FourierLSH

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


GLOVE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "glove.6B.300d.txt",
)


# ---------------------------------------------------------------------------
# FAISS LSH baseline (requires faiss-cpu)
# ---------------------------------------------------------------------------

class FaissLSH:
    """FAISS LSH with IndexBinaryFlat for Hamming search.

    Uses faiss.IndexLSH for encoding, then faiss.IndexBinaryFlat for
    packed SIMD Hamming search. This gives FAISS its best possible
    search path — the same packed + SIMD approach as FourierLSH.
    """

    def __init__(self, dim: int, bits: int = 256):
        self.bits = bits
        self.dim = dim
        self._lsh = faiss.IndexLSH(dim, bits)

    def _extract_packed_codes(self) -> np.ndarray:
        """Extract packed uint8 codes from the internal LSH index."""
        n = self._lsh.ntotal
        code_size = self._lsh.code_size
        codes = faiss.rev_swig_ptr(self._lsh.codes.data(), n * code_size)
        return np.array(codes, dtype=np.uint8).reshape(n, code_size)

    def build_and_search(self, data, queries, n_candidates, k):
        """Encode data, build binary index, search, rerank. Returns (recall_inputs, times).

        This runs the full FAISS-native pipeline:
        1. IndexLSH encodes all vectors to packed binary codes
        2. IndexBinaryFlat does SIMD Hamming search for candidates
        3. Exact distance reranking on candidates
        """
        vecs = np.ascontiguousarray(data, dtype=np.float32)
        qvecs = np.ascontiguousarray(queries, dtype=np.float32)

        # Encode database
        t0 = time.perf_counter()
        self._lsh.reset()
        self._lsh.add(vecs)
        db_codes = self._extract_packed_codes()
        # Build binary index
        binary_index = faiss.IndexBinaryFlat(self.bits)
        binary_index.add(db_codes)
        encode_time = time.perf_counter() - t0

        # Encode queries
        self._lsh.reset()
        self._lsh.add(qvecs)
        q_codes = self._extract_packed_codes()
        self._lsh.reset()

        # Search: FAISS SIMD Hamming + rerank
        n_queries = queries.shape[0]
        predicted = np.empty((n_queries, k), dtype=np.int32)
        t0 = time.perf_counter()
        _, cand_indices = binary_index.search(q_codes, n_candidates)
        for i in range(n_queries):
            cand_idx = cand_indices[i]
            cand_idx = cand_idx[cand_idx >= 0]  # filter -1 padding
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
        query_time = time.perf_counter() - t0

        return predicted, encode_time, query_time


# ---------------------------------------------------------------------------
# FAISS HNSW baseline (requires faiss-cpu)
# ---------------------------------------------------------------------------

class FaissHNSW:
    """FAISS IndexHNSWFlat — the gold-standard graph-based ANN baseline.

    Uses M=32, efConstruction=200. efSearch is varied at query time.
    """

    def __init__(self, dim: int, M: int = 32, ef_construction: int = 200):
        self.dim = dim
        self._index = faiss.IndexHNSWFlat(dim, M)
        self._index.hnsw.efConstruction = ef_construction

    def build(self, data: np.ndarray) -> float:
        """Add vectors to the HNSW index. Returns build time in seconds."""
        vecs = np.ascontiguousarray(data, dtype=np.float32)
        t0 = time.perf_counter()
        self._index.add(vecs)
        return time.perf_counter() - t0

    def search(self, queries: np.ndarray, k: int, ef_search: int) -> tuple[np.ndarray, float]:
        """Search the index. Returns (indices, query_time_seconds)."""
        self._index.hnsw.efSearch = ef_search
        q = np.ascontiguousarray(queries, dtype=np.float32)
        t0 = time.perf_counter()
        _, indices = self._index.search(q, k)
        query_time = time.perf_counter() - t0
        return indices, query_time


# ---------------------------------------------------------------------------
# GloVe loader
# ---------------------------------------------------------------------------

def load_glove(path: str, n_vectors: int) -> np.ndarray | None:
    """Load vectors from a GloVe text file.

    Each line is: word f1 f2 ... f300
    Returns unit-normalized float64 array of shape (n_vectors, 300),
    or None if the file is not found.
    """
    if not os.path.isfile(path):
        return None

    vectors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(vectors) >= n_vectors:
                break
            parts = line.rstrip().split(" ")
            # skip word token, parse floats
            vectors.append([float(x) for x in parts[1:]])

    data = np.array(vectors, dtype=np.float64)
    # Normalize to unit vectors
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero
    data /= norms
    return data


# ---------------------------------------------------------------------------
# OpenAI embeddings loader (HuggingFace / Qdrant)
# ---------------------------------------------------------------------------

def load_openai_hf(n_vectors: int) -> np.ndarray | None:
    """Load OpenAI text-embedding-3-small vectors from HuggingFace.

    Streams the Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K
    dataset. Returns unit-normalized float64 array of shape (n_vectors, 1536),
    or None if the `datasets` library is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    ds = load_dataset(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K",
        split="train", streaming=True,
    )
    vectors = []
    for item in ds:
        if len(vectors) >= n_vectors:
            break
        vectors.append(item["text-embedding-3-small-1536-embedding"])
    data = np.array(vectors, dtype=np.float64)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    data /= norms
    return data


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def brute_force_knn(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Ground truth: exact k-NN by exhaustive search.

    Returns array of shape (n_queries, k) with indices into data.
    """
    n_queries = queries.shape[0]
    indices = np.empty((n_queries, k), dtype=np.int32)
    for i, q in enumerate(queries):
        dists = np.linalg.norm(data - q, axis=1)
        indices[i] = np.argpartition(dists, k)[:k]
        indices[i] = indices[i][np.argsort(dists[indices[i]])]
    return indices


def recall_at_k(ground_truth: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """Compute mean Recall@k across queries."""
    n_queries = ground_truth.shape[0]
    recalls = []
    for i in range(n_queries):
        gt_set = set(ground_truth[i, :k])
        pred_set = set(predicted[i, :k])
        recalls.append(len(gt_set & pred_set) / k)
    return float(np.mean(recalls))


def run_ann_search(hasher, data, queries, ground_truth,
                   n_candidates: int = 200, k: int = 100):
    """Run ANN search with FourierLSH and compute recall.

    Uses batch Rust SIMD Hamming top-k (one call for all queries).
    """
    # Encode
    t0 = time.perf_counter()
    db_hashes = hasher.encode_batch(data)
    q_hashes = hasher.encode_batch(queries)
    encode_time = time.perf_counter() - t0

    n_queries = queries.shape[0]
    predicted = np.empty((n_queries, k), dtype=np.int32)

    # Batch Hamming top-k: one Rust call for all queries
    t0 = time.perf_counter()
    cand_indices = hasher.hamming_top_k(q_hashes, db_hashes, n_candidates)

    # Rerank candidates by exact distance
    for i in range(n_queries):
        cand_idx = cand_indices[i]
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
    query_time = time.perf_counter() - t0

    rec = recall_at_k(ground_truth, predicted, k)
    return rec, encode_time, query_time


# ---------------------------------------------------------------------------
# Dataset benchmark (shared logic for any dataset)
# ---------------------------------------------------------------------------

def run_dataset_benchmark(name: str, data: np.ndarray, queries: np.ndarray,
                          k: int = 100, n_candidates: int = 500,
                          bit_widths: list[int] | None = None,
                          hnsw_ef_search: list[int] | None = None):
    """Run the full benchmark on a single dataset."""
    if bit_widths is None:
        bit_widths = [64, 128, 256, 512, 1024, 2048, 4096]
    if hnsw_ef_search is None:
        hnsw_ef_search = [32, 64, 128, 256]

    dim = data.shape[1]
    n_points = data.shape[0]
    n_queries = queries.shape[0]

    print("=" * 72)
    print(f"Dataset: {name}")
    print("=" * 72)
    print(f"  Vectors:    {n_points} points in {dim}D")
    print(f"  Queries:    {n_queries}")
    print(f"  Metric:     Recall@{k} (candidates={n_candidates})")
    print()

    # Ground truth
    print("Computing ground truth (brute force)...", flush=True)
    gt = brute_force_knn(data, queries, k)
    print("Done.\n")

    # Header
    print(f"{'Bits':>6}  {'Method':<20}  {'Recall@100':>10}  "
          f"{'Build(ms)':>10}  {'Query(ms)':>10}")
    print("-" * 72)

    for bits in bit_widths:
        # FourierLSH
        lsh = FourierLSH(bits=bits)
        lsh_recall, lsh_enc, lsh_q = run_ann_search(
            lsh, data, queries, gt, n_candidates=n_candidates, k=k)

        # FAISS LSH + IndexBinaryFlat (if available)
        if HAS_FAISS:
            fl = FaissLSH(dim, bits=bits)
            fl_predicted, fl_enc, fl_q = fl.build_and_search(
                data, queries, n_candidates=n_candidates, k=k)
            fl_recall = recall_at_k(gt, fl_predicted, k)

        print(f"{bits:>6}  {'FourierLSH':<20}  {lsh_recall:>10.4f}  "
              f"{lsh_enc*1000:>10.1f}  {lsh_q*1000:>10.1f}")
        if HAS_FAISS:
            print(f"{bits:>6}  {'FAISS LSH':<20}  {fl_recall:>10.4f}  "
                  f"{fl_enc*1000:>10.1f}  {fl_q*1000:>10.1f}")
        print()

    # FAISS HNSW (if available)
    if HAS_FAISS:
        print("-" * 72)
        hnsw = FaissHNSW(dim)
        build_time = hnsw.build(data)
        print(f"  HNSW index built in {build_time*1000:.1f} ms "
              f"(M=32, efConstruction=200)")
        print()
        for ef in hnsw_ef_search:
            indices, query_time = hnsw.search(queries, k, ef_search=ef)
            rec = recall_at_k(gt, indices, k)
            label = f"HNSW ef={ef}"
            print(f"{'--':>6}  {label:<20}  {rec:>10.4f}  "
                  f"{build_time*1000:>10.1f}  {query_time*1000:>10.1f}")
        print()

    print("=" * 72)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    k = 100
    n_candidates = 500
    bit_widths = [64, 128, 256, 512, 1024, 2048, 4096]

    # --- Synthetic dataset (always runs) ---
    dim = 300
    n_points = 5000
    n_queries = 100

    rng = np.random.RandomState(42)
    data = rng.randn(n_points, dim).astype(np.float64)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    queries = rng.randn(n_queries, dim).astype(np.float64)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    run_dataset_benchmark(
        "Synthetic (random Gaussian, 300D)",
        data, queries, k=k, n_candidates=n_candidates, bit_widths=bit_widths,
    )

    # --- GloVe dataset (if available) ---
    print("\n")
    glove_n_db = 10_000
    glove_n_queries = 1_000
    glove_total = glove_n_db + glove_n_queries

    glove_vectors = load_glove(GLOVE_PATH, glove_total)

    if glove_vectors is None:
        print("=" * 72)
        print("GloVe dataset not found — skipping.")
        print()
        print("To enable the GloVe benchmark, download the embeddings:")
        print()
        print("    mkdir -p data && cd data")
        print("    wget https://nlp.stanford.edu/data/glove.6B.zip")
        print("    unzip glove.6B.zip")
        print()
        print("Only glove.6B.300d.txt is needed (~1 GB).")
        print("=" * 72)
    else:
        glove_data = glove_vectors[:glove_n_db]
        glove_queries = glove_vectors[glove_n_db:glove_total]

        run_dataset_benchmark(
            f"GloVe 6B 300D (first {glove_n_db} as DB, next {glove_n_queries} as queries)",
            glove_data, glove_queries,
            k=k, n_candidates=n_candidates, bit_widths=bit_widths,
        )

    # --- OpenAI text-embedding-3-small dataset (if datasets installed) ---
    print("\n")
    openai_n_db = 50_000
    openai_n_queries = 1_000
    openai_total = openai_n_db + openai_n_queries

    print("Loading OpenAI text-embedding-3-small from HuggingFace...", flush=True)
    openai_vectors = load_openai_hf(openai_total)

    if openai_vectors is None:
        print("=" * 72)
        print("OpenAI embeddings dataset not available — skipping.")
        print()
        print("To enable, install the datasets library:")
        print()
        print("    uv sync --group bench")
        print("=" * 72)
    else:
        openai_data = openai_vectors[:openai_n_db]
        openai_queries = openai_vectors[openai_n_db:openai_total]

        run_dataset_benchmark(
            f"OpenAI text-embedding-3-small 1536D (first {openai_n_db} as DB, next {openai_n_queries} as queries)",
            openai_data, openai_queries,
            k=k, n_candidates=n_candidates, bit_widths=bit_widths,
        )


if __name__ == "__main__":
    run_benchmark()
