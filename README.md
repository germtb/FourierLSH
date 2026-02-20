# FourierLSH

**LSH via random sign flips + FFT.**

Hash high-dimensional vectors into compact binary codes using O(L log d) operations and a single integer seed. No projection matrix needed.

```python
from fourierlsh import FourierLSH
import numpy as np

lsh = FourierLSH(bits=256, seed=0)

# Encode a single vector (packed bytes)
h = lsh.encode(np.random.randn(300))    # shape (32,), dtype uint8

# Encode a batch
hashes = lsh.encode_batch(vectors)      # shape (n, 32)

# Find nearest neighbors via Hamming distance (Rust SIMD popcount)
distances = lsh.hamming_batch(query_hash, db_hashes)

```

## What is this?

FourierLSH replaces the dense random projection matrix in standard LSH with a random sign flip + FFT. The theoretical encoding complexity is O(L log d) instead of O(dL). In practice, BLAS-optimized matmul (used by FAISS) is faster than NumPy's FFT, so FourierLSH is **not faster** than FAISS LSH at encoding or search.

The actual advantage is simpler: **no projection matrix**. FourierLSH needs only a single integer seed to produce identical hashes on any machine. Standard LSH requires storing and distributing an L x d float matrix.

| | Random Projection | FourierLSH |
|-|-|-|
| **Hash cost** | O(d L) matmul | O(L log d) FFT |
| **Projection storage** | L x d float matrix | single integer seed |
| **Encoding speed** | **faster** (BLAS) | slower (FFT) |

## Benchmarks

All methods use packed binary codes + SIMD Hamming search (FAISS `IndexBinaryFlat` for FAISS LSH, Rust popcount for FourierLSH). This is an apples-to-apples comparison.

### GloVe 300D (10K vectors, 1K queries, Recall@100, candidates=500)

| Bits | Method | Recall@100 | Build (ms) | Query (ms) |
| ---: | :----- | ---------: | ---------: | ---------: |
|  256 | FourierLSH |  88.7% |        58 |       291 |
|  256 | FAISS LSH  |  88.9% |     **4** |   **261** |
|  512 | FourierLSH |  96.9% |        25 |       295 |
|  512 | FAISS LSH  |  97.4% |     **7** |   **264** |
| 1024 | FourierLSH |  99.6% |        39 |       351 |
| 1024 | FAISS LSH  |  99.6% |    **18** |   **257** |
| 4096 | FourierLSH | 100.0% |       131 |       668 |
| 4096 | FAISS LSH  | 100.0% |    **80** |   **292** |
|   -- | HNSW ef=64 |  96.0% |       344 |    **17** |
|   -- | HNSW ef=256 | 99.9% |       344 |    **48** |

### OpenAI text-embedding-3-small 1536D (50K vectors, 1K queries)

| Bits | Method | Recall@100 | Build (ms) | Query (ms) |
| ---: | :----- | ---------: | ---------: | ---------: |
|  256 | FourierLSH |  58.4% |     1,452 |     1,503 |
|  256 | FAISS LSH  |  58.4% |    **68** | **1,063** |
|  512 | FourierLSH |  80.7% |       776 |     1,980 |
|  512 | FAISS LSH  |  81.5% |    **97** | **1,245** |
| 1024 | FourierLSH |  95.2% |     1,148 |     1,996 |
| 1024 | FAISS LSH  |  95.3% |   **199** | **1,342** |
| 2048 | FourierLSH |  99.3% |     3,293 |     9,738 |
| 2048 | FAISS LSH  |  99.6% | **1,120** | **1,326** |
|   -- | HNSW ef=64 |  93.9% |    37,005 |   **336** |
|   -- | HNSW ef=256 | 99.6% |    37,005 | **1,018** |

### What the numbers say

**FourierLSH vs FAISS LSH**: Same recall at every bit width. FAISS is faster at both encoding and query — BLAS matmul is heavily optimized and beats NumPy's FFT in practice, even though FourierLSH has better asymptotic complexity. The gap is smaller at query time (both use SIMD Hamming) than at encoding time.

**FourierLSH vs HNSW**: HNSW is much faster and achieves higher recall. But HNSW requires ~6.4 KB per vector (full float32 + graph) vs 128 bytes at 1024 bits — a 50x memory difference. HNSW also requires 37s to build the index.

**Multi-round encoding**: When `bits > dim`, FourierLSH automatically runs multiple FFT rounds with independent sign flips from the same seed. Recall scales to 100% — no ceiling.

## How it works

1. **Sign flip** — multiply each element by a random ±1 drawn from a seeded RNG. Breaks any alignment between the input and the DFT basis.
2. **FFT** — `np.fft.rfft(scrambled, n=d)` to extract ~d bits per round.
3. **Sign bits** — for each complex coefficient, take `sign(real)` then `sign(imag)`.
4. **Multi-round** — when `bits > dim`, run additional rounds with fresh sign flips from the same RNG stream. Each round produces genuinely independent bits.

One seed produces unlimited bits — round 0 uses the first d random signs, round 1 the next d, and so on. No projection matrix is ever materialized or stored.

## FAISS integration

Hashes are already packed bytes, so they plug directly into FAISS binary indices:

```python
import faiss
import numpy as np
from fourierlsh import FourierLSH

lsh = FourierLSH(bits=256, seed=0)
db_hashes = lsh.encode_batch(db_vectors)    # (n, 32) uint8, already packed

index = faiss.IndexBinaryFlat(256)
index.add(db_hashes)

q_hash = lsh.encode(query).reshape(1, -1)
D, I = index.search(q_hash, k=100)
```

## Installation

```bash
uv sync
```

## API

### `FourierLSH(bits=256, seed=0)`

Create a hasher. `bits` controls hash length, `seed` controls the random sign flips.

### `.encode(vector) -> ndarray`

Hash a single vector. Returns packed `uint8` bytes (shape `(code_size,)`).

### `.encode_batch(vectors) -> ndarray`

Hash a batch of vectors. Returns `(n, code_size)` packed bytes.

### `.hamming_batch(query, database) -> ndarray`

Hamming distances from one packed hash to many. Uses Rust SIMD popcount when available, numpy LUT fallback otherwise.

### `.hamming_top_k(queries, database, k) -> ndarray`

Batch top-k search: for each query, find the k nearest database rows by Hamming distance. One Rust call for all queries. Returns `(n_queries, k)` indices.

### `.unpack(packed) -> ndarray`

Unpack to individual bits (0/1 uint8) if you need them.

## Prefix safety

The first k bits of an n-bit hash equal the standalone k-bit hash (same seed), for any k ≤ n — even when bits > dim:

```python
lsh64   = FourierLSH(bits=64)
lsh4096 = FourierLSH(bits=4096)

v = np.random.randn(300)  # dim=300, 4096 >> 300 — still works
assert np.array_equal(lsh64.unpack(lsh64.encode(v)),
                      lsh4096.unpack(lsh4096.encode(v))[:64])
```

This works because multi-round encoding draws sign flips sequentially from the same RNG stream. Round boundaries don't affect the bit sequence.

## License

MIT.
