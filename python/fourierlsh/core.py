"""
FourierLSH: LSH via random sign flips + FFT.

How It Works:
    1. Apply seeded random ±1 sign flips to break coherence with the DFT basis.
    2. Compute ``np.fft.rfft(scrambled, n=d)`` to extract up to ~d bits per round.
    3. For each complex coefficient, emit ``sign(real)`` then ``sign(imag)``.
    4. When more bits are requested than one FFT provides, run additional
       rounds with fresh sign flips derived from the same seed.
    5. Concatenate and pack into uint8 bytes.

All sign flips are generated from a single seeded RNG: round 0 consumes the
first d random signs, round 1 the next d, and so on. This means one seed
produces an unlimited number of independent hash bits.
"""

import numpy as np

try:
    from fourierlsh._native import hamming_distances as _hamming_native
    from fourierlsh._native import hamming_top_k as _hamming_top_k_native
except ImportError:
    _hamming_native = None
    _hamming_top_k_native = None

# Byte popcount lookup table for numpy fallback
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

    __slots__ = ("bits", "seed", "code_size", "_sign_masks", "_sign_masks_dim")

    def __init__(self, bits: int = 256, seed: int = 0):
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.bits = bits
        self.seed = seed
        self.code_size = (bits + 7) // 8
        self._sign_masks: list[np.ndarray] | None = None
        self._sign_masks_dim: int | None = None

    # ------------------------------------------------------------------
    # Sign flip cache
    # ------------------------------------------------------------------

    def _get_sign_masks(self, dim: int) -> list[np.ndarray]:
        """Return cached uint32 XOR masks for each round of sign flips.

        A single RNG seeded once produces d signs per round sequentially,
        so round 0 uses signs [0, d), round 1 uses [d, 2d), etc.
        """
        if dim != self._sign_masks_dim:
            bits_per_round = dim  # one FFT of length d gives ~d useful bits
            n_rounds = max(1, (self.bits + bits_per_round - 1) // bits_per_round)
            rng = np.random.RandomState(self.seed)
            masks = []
            for _ in range(n_rounds):
                flips = rng.randint(0, 2, size=dim, dtype=np.uint32)
                masks.append(flips << np.uint32(31))
            self._sign_masks = masks
            self._sign_masks_dim = dim
        return self._sign_masks

    # ------------------------------------------------------------------
    # Core: sign-flip → FFT → packed sign bits (multi-round)
    # ------------------------------------------------------------------

    def _encode_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Multi-round encode: sign-flip, FFT, extract bits, concatenate, pack.

        When bits <= dim, this is a single round (identical to the old path).
        When bits > dim, multiple rounds with independent sign flips produce
        genuinely independent hash bits instead of redundant zero-padded ones.
        """
        d = vectors.shape[-1]
        n = vectors.shape[0]
        masks = self._get_sign_masks(d)
        bits_per_round = d
        all_bits = []
        bits_remaining = self.bits

        for mask in masks:
            take = min(bits_per_round, bits_remaining)
            scrambled = (vectors.view(np.uint32) ^ mask).view(np.float32)
            coeffs = np.fft.rfft(scrambled, n=d, axis=-1)
            n_coeffs = coeffs.shape[-1]
            ri = np.empty((n, n_coeffs, 2), dtype=np.float32)
            ri[:, :, 0] = coeffs.real
            ri[:, :, 1] = coeffs.imag
            round_bits = ri.reshape(n, 2 * n_coeffs)[:, :take] >= 0
            all_bits.append(round_bits)
            bits_remaining -= take
            if bits_remaining <= 0:
                break

        return np.packbits(np.concatenate(all_bits, axis=1), axis=-1)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, vector: np.ndarray) -> np.ndarray:
        """Encode a single vector to a packed binary hash.

        Returns
        -------
        ndarray, shape (code_size,), dtype uint8
        """
        v = np.asarray(vector, dtype=np.float32)[np.newaxis, :]
        return self._encode_vectors(v)[0]

    def encode_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Encode a batch of vectors to packed binary hashes.

        Returns
        -------
        ndarray, shape (n, code_size), dtype uint8
        """
        return self._encode_vectors(np.asarray(vectors, dtype=np.float32))

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

        Uses native SIMD extension if available, otherwise numpy fallback.

        Parameters
        ----------
        query : ndarray, shape (code_size,), dtype uint8
        database : ndarray, shape (n, code_size), dtype uint8

        Returns
        -------
        ndarray, shape (n,), dtype int32
        """
        if _hamming_native is not None:
            return _hamming_native(
                np.ascontiguousarray(query),
                np.ascontiguousarray(database),
            )
        return _POPCOUNT_LUT[database ^ query].sum(axis=-1)

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
        if _hamming_top_k_native is not None:
            return _hamming_top_k_native(
                np.ascontiguousarray(queries),
                np.ascontiguousarray(database),
                k,
            )
        # Numpy fallback: per-query loop
        n_queries = queries.shape[0]
        n_db = database.shape[0]
        actual_k = min(k, n_db)
        results = np.empty((n_queries, actual_k), dtype=np.int32)
        for i in range(n_queries):
            dists = _POPCOUNT_LUT[database ^ queries[i]].sum(axis=-1)
            idx = np.argpartition(dists, actual_k)[:actual_k]
            idx = idx[np.argsort(dists[idx])]
            results[i] = idx
        return results

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
