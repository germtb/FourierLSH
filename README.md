# FourierLSH

**LSH via random sign flips + FFT.**

Hash high-dimensional vectors into compact binary codes using O(L log d) operations and a single integer seed. No projection matrix needed.

```python
from fourierlsh import FourierLSH
import numpy as np

lsh = FourierLSH(bits=256, seed=0)

# Encode a single vector (packed bytes)
h = lsh.encode(np.random.randn(300))    # shape (32,), dtype uint8

# Encode a batch (Rust FFT + rayon parallel)
hashes = lsh.encode_batch(vectors) # shape (n, 32)

# Find nearest neighbors via Hamming distance (NEON SIMD + rayon)
indices = lsh.hamming_top_k(query_hashes, db_hashes, k=100)
```

## What is this?

FourierLSH replaces the dense random projection matrix in standard LSH with a random sign flip + FFT. The encoding complexity is O(L log d) instead of O(dL), and with a Rust implementation (rustfft + rayon + NEON SIMD), this translates to large wall-clock speedups.

| | Random Projection (FAISS) | FourierLSH (Rust) |
|-|-|-|
| **Hash cost** | O(d L) matmul | O(L log d) FFT |
| **Projection storage** | L x d float matrix | single integer seed |
| **Encoding speed (d=1536, L=4096)** | 677 ms | **22 ms (31x faster)** |
| **Hamming search** | FAISS IndexBinaryFlat | **Rust NEON+rayon (2.4-4.7x faster)** |

## Benchmarks

All benchmarks: Apple Silicon (8 cores), 10K vectors, 200 queries, Recall@100 with 500 candidates, median of 5 runs with warmup.

### Encoding time (the O(L log d) vs O(dL) question)

Encoding phase only — isolated from search and reranking.

| Bits | FourierLSH (Rust) | FAISS LSH (BLAS) | Speedup | | FourierLSH (Rust) | FAISS LSH (BLAS) | Speedup |
| ---: | ----------------: | ---------------: | ------: |-| ----------------: | ---------------: | ------: |
| | **d=300** | | | | **d=1536** | | |
|   64 |   2.2 ms |   0.8 ms |  0.4x | |   4.5 ms |   2.5 ms |   0.6x |
|  256 |   2.6 ms |   2.7 ms | **1.0x** | |   4.6 ms |   8.2 ms | **1.8x** |
|  512 |   4.3 ms |   7.4 ms | **1.7x** | |   5.4 ms |  17.6 ms | **3.3x** |
| 1024 |   7.7 ms |  27.0 ms | **3.5x** | |   6.8 ms |  37.2 ms | **5.5x** |
| 2048 |  11.6 ms | 110.1 ms | **9.5x** | |  11.1 ms | 133.1 ms | **12x** |
| 4096 |  24.1 ms | 632.0 ms | **26x** | |  22.1 ms | 676.7 ms | **31x** |

Crossover at ~256 bits. The advantage grows with L because FAISS scales as O(dL) while FourierLSH scales as O(L log d).

### Hamming search (Rust NEON+rayon vs FAISS IndexBinaryFlat)

Same packed binary codes, different search implementations. d=1536, 200 queries vs 10K database.

| Bits | Rust NEON | FAISS BinaryFlat | Speedup |
| ---: | --------: | ---------------: | ------: |
|  256 |   0.6 ms  |   2.8 ms | **4.7x** |
| 1024 |   0.9 ms  |   3.4 ms | **3.8x** |
| 4096 |   2.0 ms  |   4.7 ms | **2.4x** |

### FourierLSH vs HNSW

Build includes index construction; Query includes search + reranking. Memory/vec for HNSW = float32 vectors + graph edges.

**d=300, 10K vectors:**

| Method | Build | Query | Mem/vec | Recall@100 |
| :----- | ----: | ----: | ------: | ---------: |
| FourierLSH 1024b |   7 ms |  39 ms | 128 B | 82.2% |
| FourierLSH 2048b |  12 ms |  40 ms | 256 B | 94.4% |
| FourierLSH 4096b |  24 ms |  43 ms | 512 B | 99.1% |
| HNSW ef=128 | 789 ms |   4 ms | 1,712 B | 87.0% |
| HNSW ef=256 | 789 ms |   7 ms | 1,712 B | 97.7% |
| HNSW ef=512 | 789 ms |  11 ms | 1,712 B | 99.9% |

**d=1536, 10K vectors:**

| Method | Build | Query | Mem/vec | Recall@100 |
| :----- | ----: | ----: | ------: | ---------: |
| FourierLSH 2048b |   11 ms | 143 ms | 256 B | 57.7% |
| FourierLSH 4096b |   18 ms | 151 ms | 512 B | 76.6% |
| HNSW ef=128 | 7,032 ms |  31 ms | 6,656 B | 72.1% |
| HNSW ef=256 | 7,032 ms |  46 ms | 6,656 B | 91.0% |
| HNSW ef=512 | 7,032 ms |  57 ms | 6,656 B | 99.1% |

**At matched recall, the comparison separates cleanly:**

| Axis | Winner | Margin |
|------|--------|--------|
| **Build time** | FourierLSH | 33-384x faster |
| **Query time** | HNSW | 4-10x faster |
| **Memory** | FourierLSH | 3-13x smaller |

FourierLSH does **not** beat HNSW at query time — HNSW's O(d log n) search is fundamentally faster than brute-force Hamming scan O(nL). For static indexes with high query volume (the typical AI deployment), HNSW queries faster. FourierLSH's advantage is memory: at 100M vectors (d=1536), FourierLSH needs 51 GB vs HNSW's 666 GB.

### When to use what

- **FourierLSH**: memory-constrained deployments, streaming/changing data, distributed (seed-only hashing), prefiltering stage in multi-stage retrieval
- **HNSW**: static databases, high query volume, highest recall needed, query latency is the priority

## How it works

1. **Sign flip** — XOR the IEEE 754 sign bit of each float32 with a random ±1 drawn from a seeded RNG. Breaks alignment between the input and the DFT basis.
2. **FFT** — compute rFFT of the scrambled vector, yielding ~d complex coefficients.
3. **Sign bits** — for each coefficient, take `sign(real)` then `sign(imag)`.
4. **Multi-round** — when `bits > dim`, run additional rounds with fresh sign flips from the same RNG stream. Each round produces genuinely independent bits.
5. **Pack** — concatenate bits and pack into uint8 bytes.

The sign flips must go on the **input** (before FFT), not on the output bits. Flipping output bits is just XOR — an isometry in Hamming space that adds no information. Input sign flips change the effective projection vectors, creating genuinely different hash functions per round.

## FAISS integration

Hashes are already packed bytes, so they plug directly into FAISS binary indices:

```python
import faiss
import numpy as np
from fourierlsh import FourierLSH

lsh = FourierLSH(bits=256, seed=0)
db_hashes = lsh.encode_batch(db_vectors)  # (n, 32) uint8

index = faiss.IndexBinaryFlat(256)
index.add(db_hashes)

q_hash = lsh.encode_batch(query.reshape(1, -1))
D, I = index.search(q_hash, k=100)
```

## Installation

```bash
uv sync
```

Build the Rust extension (required for Rust FFT encoder and NEON Hamming search):

```bash
uv run --with maturin maturin develop --release
```

## API

### `FourierLSH(bits=256, seed=0)`

Create a hasher. `bits` controls hash length, `seed` controls the random sign flips.

### `.encode(vector) -> ndarray`

Hash a single vector. Returns packed `uint8` bytes (shape `(code_size,)`).

### `.encode_batch(vectors) -> ndarray`

Hash a batch using Rust (rustfft + rayon). Parallel across vectors, ~3-31x faster than FAISS at high bit counts. Returns `(n, code_size)` packed bytes.

### `.hamming_batch(query, database) -> ndarray`

Hamming distances from one packed hash to many. Uses Rust NEON SIMD + rayon.

### `.hamming_top_k(queries, database, k) -> ndarray`

Batch top-k search: for each query, find the k nearest database rows by Hamming distance. NEON SIMD + rayon parallel. Returns `(n_queries, k)` indices.

### `.unpack(packed) -> ndarray`

Unpack to individual bits (0/1 uint8) if you need them.

## Prefix safety

The first k bits of an n-bit hash equal the standalone k-bit hash (same seed), for any k <= n — even when bits > dim:

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
