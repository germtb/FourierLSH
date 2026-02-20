# FourierLSH: Theory

## Algorithm Overview

FourierLSH converts high-dimensional float vectors into compact binary codes (bit strings), where similar vectors produce similar hashes (low Hamming distance). It uses the FFT as a fast way to generate many random hyperplane projections from a single operation.

### Steps

1. **Random sign flips** — Generate a deterministic mask of random ±1 values from a seed. Apply by XOR-ing the IEEE 754 sign bit of each float. The same mask is applied to every vector.
2. **FFT** — Compute the real FFT of the flipped vector. This produces ~d/2 + 1 complex coefficients in O(d log d) time.
3. **Extract sign bits** — For each complex coefficient, take sign(real) then sign(imag). Each sign bit is one hash bit. One FFT pass yields ~d bits.
4. **Multi-round** — If L > d bits are needed, run additional rounds with fresh sign-flip masks derived from the same seed. Each round produces d independent bits.
5. **Pack** — Concatenate all sign bits and pack into uint8 bytes.

### Worked Example

Starting with vector (0.2, 0.8, 0.2), suppose the random mask is (-1, -1, +1). The flipped vector is (-0.2, -0.8, 0.2).

FFT of (-0.2, -0.8, 0.2):

```
coefficient 0:  -0.8 + 0.0i
coefficient 1:   0.1 + 0.866i
```

Extract sign bits by checking whether each real and imaginary part is >= 0. The DC coefficient (index 0) has imaginary part always 0, so we skip it — that sign bit would be constant and uninformative:

```
coeff 0 real: -0.8   → negative → bit 0
coeff 0 imag:  0.0   → SKIP (always 0 for DC)
coeff 1 real:  0.1   → positive → bit 1
coeff 1 imag:  0.866 → positive → bit 1
```

Hash: `0 1 1` → packed: `0b01100000` = `0x60`.

## The FFT Basis

The FFT decomposes a vector into a sum of sinusoids at fixed frequencies. For a vector of length d, the basis vectors are complex exponentials at d discrete frequencies. Each FFT coefficient is a dot product of the input with one of these basis vectors. The FFT algorithm computes all d dot products in O(d log d) instead of O(d²).

## Role of the Random Sign Flips

The sign flips serve one primary purpose and have a secondary benefit.

### Primary: Generating Multiple Independent Rounds

The FFT alone gives ~d fixed projections. If you need L > d bits, running the FFT again on the same input produces the exact same bits. Each round of fresh sign flips creates a new set of d projections, all from the same FFT operation. Without sign flips, additional rounds would be useless repetition.

For a single round (L <= d), you could skip the sign flips entirely. The FFT's d fixed projections are already a perfectly good orthogonal set — for isotropic embeddings they work fine.

### Secondary: Breaking Coherence with the FFT Basis

The sign flips randomly rotate the data relative to the FFT basis. This prevents pathological cases where the input is aligned with a basis vector, causing some coefficients to be near zero (making their sign bits uninformative). In high dimensions this is extremely unlikely for typical data, so this benefit is secondary.

### Why P(agree) ≈ 1 - θ/π Holds

For standard random-hyperplane LSH with Gaussian random vectors g, each hash bit computes sgn(⟨g, x⟩), and P(agree) = 1 - θ/π holds exactly because g is isotropic — the induced hyperplane is uniformly random.

FourierLSH computes hash bit k as:

```
h_k(x) = sgn(⟨F_k ∘ s, x⟩) = sgn(Σ_j F_kj · s_j · x_j)
```

where F_k is the k-th row of the DFT matrix and s ∈ {±1}^d is the random sign vector. The effective projection direction is g_k = F_k ∘ s, where each component g_kj = F_kj · s_j has its sign randomized independently.

For two unit vectors x, y with angle θ, the hash bit agrees when sgn(⟨g_k, x⟩) = sgn(⟨g_k, y⟩). The probability of disagreement equals the probability that the random hyperplane with normal g_k separates x and y.

Since the DFT basis vectors F_k have entries with equal magnitudes (|F_kj| = 1/√d), the sign-randomized vector g_k = F_k ∘ s has the property that:

1. Each component has a symmetric distribution (s_j flips the sign of F_kj with probability 1/2)
2. All components have equal variance: Var(g_kj) = |F_kj|² = 1/d
3. The components are pairwise uncorrelated (E[g_ki · g_kj] = F_ki · F_kj · E[s_i · s_j] = 0 for i ≠ j since the s_j are independent)

By the Berry-Esseen theorem, for large d the projections ⟨g_k, x⟩ and ⟨g_k, y⟩ converge to a jointly Gaussian distribution. In this Gaussian limit, the probability that a random hyperplane separates two vectors depends only on the angle between them, giving P(disagree) = θ/π — the same as standard LSH.

This is not exact for finite d (the entries of g_k are bounded, not Gaussian), but the convergence is fast. Empirically, the deviation from P(agree) = 1 - θ/π is below 0.1% at d ≥ 300.

### Prior Art: FJLT and Cross-Polytope LSH

The construction "random diagonal signs × structured orthogonal transform" originates from the Fast Johnson-Lindenstrauss Transform (Ailon & Chazelle, 2009), which proved it preserves distances with high probability.

Andoni, Indyk, Laarhoven, Razenshteyn, and Schmidt (2015) applied this construction to LSH for angular similarity in their cross-polytope LSH work, using the Walsh-Hadamard Transform instead of the FFT. The FALCONN library implements sign-bit extraction from this construction (random sign flips + WHT + sign bits + multi-round encoding), which is algorithmically equivalent to FourierLSH with a different choice of orthogonal transform.

### Why Sign Flips Must Be Applied Before the FFT

Applying random flips after the transform (to the output bits) would be equivalent to XOR on the hash — an isometry in Hamming space that preserves all pairwise distances and adds no information. Flipping bit k just relabels 0↔1 at that position; the Hamming distance between any two vectors stays exactly the same.

Randomness before a linear transform changes the geometry. Randomness after a sign function is just bit relabeling.

## Hamming Distance

Hamming distance between two binary strings is the number of bits that differ:

```
hash A:  0 1 0 1 1 0 1 0
hash B:  0 1 1 1 0 0 1 0
             ^   ^
```

Hamming distance = 2. Computed as XOR (lights up differing bits) followed by popcount (count the 1s). Hardware has single instructions for both, making this extremely fast.

## Cosine Similarity to Hamming Distance Mapping

For the random-hyperplane LSH family, the probability that two vectors agree on a single hash bit is:

```
P(agree) = 1 - θ/π
```

where θ is the angle between them and cos(θ) is the cosine similarity.

For L independent hash bits, the expected Hamming distance is:

```
E[hamming] = L × θ/π = L × arccos(similarity) / π
```

Concrete values with L = 512 bits:

| cosine sim | θ (degrees) | expected hamming / 512 |
|---|---|---|
| 1.0 | 0 | 0 |
| 0.9 | 26 | 37 |
| 0.7 | 46 | 65 |
| 0.5 | 60 | 85 |
| 0.0 | 90 | 256 |
| -1.0 | 180 | 512 |

The mapping is monotonic — higher cosine similarity always means lower expected Hamming distance. The relationship is nonlinear (goes through arccos) but preserves ranking perfectly in expectation.

## Computational Complexity at Fixed Recall

### The Naive Comparison is Misleading

Standard LSH has cost O(L × d), and FourierLSH has cost O(L × log d). With L as a free parameter, one might think standard LSH could be faster by choosing a small L. But L isn't free — it's determined by the recall target.

### How L Scales with Dimension

For random hyperplane LSH, reliable separation between a nearest neighbor and a random vector requires:

```
gap   = L × (θ_random - θ_nn) / π       (signal)
noise ≈ √(L/4)                           (standard deviation)

separation requires: gap >> noise
  L × Δθ/π  >>  √(L/4)
  √L        >>  π / (4Δθ)
  L          ∝  1 / Δθ²
```

For random Gaussian vectors in d dimensions, all pairwise angles concentrate around π/2 with spread ~1/√d. The nearest neighbor gap scales as:

```
Δθ  ≈  c × √(log n) / √d
```

So for fixed database size n and fixed recall target:

```
L  ∝  d / log(n)
```

L scales linearly with d. This is consistent with empirical data — at d=300, ~900 bits for 0.77 recall (3d); at d=1536, ~4096 bits for 0.77 recall (2.7d).

### The Clean Comparison

Substituting L = αd (where α depends on recall target, not dimension):

| Method | General cost | At fixed recall (L = αd) |
|---|---|---|
| Standard LSH | O(L × d) | **O(d²)** |
| FourierLSH | O(L × log d) | **O(d log d)** |

The speedup ratio is **d / log d**, which grows with dimension. Fixing recall removes the degree of freedom in exactly the way that favors FourierLSH.

At d = 1536: theoretical speedup ≈ 1536 / log₂(1536) ≈ 140×. The empirical 31× is smaller due to constant factors (BLAS is highly optimized, FFT has per-round overhead), but the trend is clear and the gap widens as d increases.
