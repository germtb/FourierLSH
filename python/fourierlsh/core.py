"""
FourierLSH: LSH via random sign flips + FFT.

How It Works:
    1. Apply seeded random ±1 sign flips to break coherence with the DFT basis.
    2. Compute the FFT to extract up to ~d bits per round.
    3. For each complex coefficient, emit ``sign(real)`` then ``sign(imag)``,
       skipping the always-zero imaginary parts (DC and Nyquist).
    4. When more bits are requested than one FFT provides, run additional
       rounds with fresh sign flips derived from the same seed.
    5. Concatenate and pack into uint8 bytes.

All sign flips are generated from a single seeded RNG: round 0 consumes the
first d random signs, round 1 the next d, and so on. This means one seed
produces an unlimited number of independent hash bits.

Encoding uses a Rust implementation (rustfft + rayon) for performance.
Hamming batch/top-k uses Rust with NEON SIMD on aarch64.
"""

import numpy as np

from fourierlsh._native import hamming_distances as _hamming_native
from fourierlsh._native import hamming_top_k as _hamming_top_k_native
from fourierlsh._native import fourier_encode as _fourier_encode_native

# Byte popcount lookup table for numpy hamming fallback
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


class FourierLSH:
    """FFT-based locality-sensitive hasher with seeded random sign flips.

    Hashes are returned as packed uint8 arrays (ceil(bits/8) bytes per hash).

    Parameters
    ----------
    bits : int
        Number of hash bits to produce (default 256). Must be >= 1.
    seed : int
        Random seed for sign flip generation (default 0).

    Examples
    --------
    >>> lsh = FourierLSH(bits=128, seed=0)
    >>> h = lsh.encode(np.random.randn(300))
    >>> h.shape
    (16,)
    >>> h.dtype
    dtype('uint8')
    """

    __slots__ = ("bits", "seed", "code_size")

    def __init__(self, bits: int = 256, seed: int = 0):
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.bits = bits
        self.seed = seed
        self.code_size = (bits + 7) // 8

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, vector: np.ndarray) -> np.ndarray:
        """Encode a single vector to a packed binary hash.

        Returns
        -------
        ndarray, shape (code_size,), dtype uint8
        """
        v = np.ascontiguousarray(
            np.asarray(vector, dtype=np.float32)[np.newaxis, :]
        )
        return np.asarray(_fourier_encode_native(v, self.bits, self.seed))[0]

    def encode_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Encode a batch of vectors to packed binary hashes.

        Uses the Rust FFT encoder with rayon parallelism.

        Returns
        -------
        ndarray, shape (n, code_size), dtype uint8
        """
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        return np.asarray(_fourier_encode_native(vecs, self.bits, self.seed))

    # ------------------------------------------------------------------
    # Hamming distance
    # ------------------------------------------------------------------

    def hamming(self, a: np.ndarray, b: np.ndarray) -> int:
        """Hamming distance between two packed binary hashes."""
        return int(_POPCOUNT_LUT[a ^ b].sum())

    def hamming_batch(
        self, query: np.ndarray, database: np.ndarray
    ) -> np.ndarray:
        """Hamming distances from one packed hash to many.

        Parameters
        ----------
        query : ndarray, shape (code_size,), dtype uint8
        database : ndarray, shape (n, code_size), dtype uint8

        Returns
        -------
        ndarray, shape (n,), dtype int32
        """
        return _hamming_native(
            np.ascontiguousarray(query),
            np.ascontiguousarray(database),
        )

    def hamming_top_k(
        self, queries: np.ndarray, database: np.ndarray, k: int
    ) -> np.ndarray:
        """Batch top-k Hamming search: all queries against database in one call.

        Parameters
        ----------
        queries : ndarray, shape (n_queries, code_size), dtype uint8
        database : ndarray, shape (n, code_size), dtype uint8
        k : int

        Returns
        -------
        ndarray, shape (n_queries, k), dtype int32 — indices into database
        """
        return _hamming_top_k_native(
            np.ascontiguousarray(queries),
            np.ascontiguousarray(database),
            k,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def unpack(self, packed: np.ndarray) -> np.ndarray:
        """Unpack a packed hash to individual bits (0/1 uint8).

        Parameters
        ----------
        packed : ndarray, shape (..., code_size)

        Returns
        -------
        ndarray, shape (..., bits), dtype uint8
        """
        flat = packed.reshape(-1, self.code_size)
        unpacked = np.unpackbits(flat, axis=-1)[:, : self.bits]
        return unpacked.reshape(packed.shape[:-1] + (self.bits,))

    def __repr__(self) -> str:
        return f"FourierLSH(bits={self.bits}, seed={self.seed})"
