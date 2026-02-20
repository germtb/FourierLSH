import numpy as np
import pytest

from fourierlsh import FourierLSH


class TestEncode:
    def test_shape_and_dtype(self):
        lsh = FourierLSH(bits=128)
        h = lsh.encode(np.random.randn(300))
        assert h.shape == (16,)  # 128 / 8
        assert h.dtype == np.uint8

    def test_deterministic(self):
        lsh = FourierLSH(bits=256)
        v = np.random.RandomState(0).randn(300)
        assert np.array_equal(lsh.encode(v), lsh.encode(v))

    def test_different_instances_agree(self):
        v = np.random.RandomState(1).randn(300)
        h1 = FourierLSH(bits=256).encode(v)
        h2 = FourierLSH(bits=256).encode(v)
        assert np.array_equal(h1, h2)

    def test_1_bit(self):
        lsh = FourierLSH(bits=1)
        h = lsh.encode(np.random.randn(50))
        assert h.shape == (1,)  # ceil(1/8) = 1

    def test_zero_vector(self):
        lsh = FourierLSH(bits=64)
        h = lsh.encode(np.zeros(100))
        assert h.shape == (8,)
        assert h.dtype == np.uint8


class TestEncodeBatch:
    def test_shape(self):
        lsh = FourierLSH(bits=128)
        hashes = lsh.encode_batch(np.random.randn(50, 300))
        assert hashes.shape == (50, 16)

    def test_matches_single_encode(self):
        lsh = FourierLSH(bits=256)
        rng = np.random.RandomState(42)
        vectors = rng.randn(10, 200)
        batch = lsh.encode_batch(vectors)
        for i in range(len(vectors)):
            assert np.array_equal(batch[i], lsh.encode(vectors[i]))


class TestUnpack:
    def test_roundtrip(self):
        lsh = FourierLSH(bits=256)
        v = np.random.RandomState(0).randn(300)
        packed = lsh.encode(v)
        unpacked = lsh.unpack(packed)
        assert unpacked.shape == (256,)
        assert set(np.unique(unpacked)).issubset({0, 1})
        # Re-pack and verify
        assert np.array_equal(np.packbits(unpacked), packed)

    def test_batch_roundtrip(self):
        lsh = FourierLSH(bits=128)
        rng = np.random.RandomState(1)
        packed = lsh.encode_batch(rng.randn(10, 200))
        unpacked = lsh.unpack(packed)
        assert unpacked.shape == (10, 128)
        assert np.array_equal(np.packbits(unpacked, axis=-1), packed)


class TestPrefixSafety:
    def test_prefix_is_exact_shorter_hash(self):
        v = np.random.RandomState(7).randn(300)
        lsh64 = FourierLSH(bits=64)
        lsh256 = FourierLSH(bits=256)
        bits64 = lsh64.unpack(lsh64.encode(v))
        bits256 = lsh256.unpack(lsh256.encode(v))
        assert np.array_equal(bits64, bits256[:64])

    def test_prefix_multiple_lengths(self):
        """Prefix safety holds when all bit widths <= dim (same FFT length)."""
        v = np.random.RandomState(8).randn(300)
        bits256 = FourierLSH(bits=256).unpack(FourierLSH(bits=256).encode(v))
        for n_bits in [1, 32, 64, 128]:
            short = FourierLSH(bits=n_bits).unpack(FourierLSH(bits=n_bits).encode(v))
            assert np.array_equal(short, bits256[:n_bits])


class TestHighBitsShape:
    def test_more_bits_than_dim(self):
        lsh = FourierLSH(bits=512)
        h = lsh.encode(np.random.randn(100))
        assert h.shape == (64,)  # 512 / 8
        assert h.dtype == np.uint8

    def test_batch_more_bits_than_dim(self):
        lsh = FourierLSH(bits=512)
        hashes = lsh.encode_batch(np.random.randn(5, 100))
        assert hashes.shape == (5, 64)


class TestPrefixAcrossPadding:
    def test_prefix_safety_holds_across_rounds(self):
        """Prefix safety holds even when bits > dim (multi-round)."""
        v = np.random.RandomState(9).randn(100)
        bits64 = FourierLSH(bits=64).unpack(FourierLSH(bits=64).encode(v))
        bits512 = FourierLSH(bits=512).unpack(FourierLSH(bits=512).encode(v))
        assert np.array_equal(bits64, bits512[:64])


class TestHamming:
    def test_identical_hashes(self):
        lsh = FourierLSH(bits=128)
        h = lsh.encode(np.random.randn(100))
        assert lsh.hamming(h, h) == 0

    def test_opposite_hashes(self):
        lsh = FourierLSH(bits=64)
        a = np.zeros(8, dtype=np.uint8)
        b = np.full(8, 0xFF, dtype=np.uint8)
        assert lsh.hamming(a, b) == 64

    def test_batch(self):
        lsh = FourierLSH(bits=128)
        rng = np.random.RandomState(3)
        vectors = rng.randn(20, 200)
        hashes = lsh.encode_batch(vectors)
        q_hash = hashes[0]
        dists = lsh.hamming_batch(q_hash, hashes)
        assert dists.shape == (20,)
        assert dists[0] == 0


class TestLocalitySensitivity:
    def test_similar_vectors_closer_than_random(self):
        lsh = FourierLSH(bits=256)
        rng = np.random.RandomState(99)
        base = rng.randn(300)
        base /= np.linalg.norm(base)
        noise = rng.randn(300) * 0.05
        similar = base + noise
        similar /= np.linalg.norm(similar)
        distant = rng.randn(300)
        distant /= np.linalg.norm(distant)

        h_base = lsh.encode(base)
        h_similar = lsh.encode(similar)
        h_distant = lsh.encode(distant)

        d_similar = lsh.hamming(h_base, h_similar)
        d_distant = lsh.hamming(h_base, h_distant)
        assert d_similar < d_distant


class TestSeeds:
    def test_different_seeds_differ(self):
        v = np.random.RandomState(0).randn(300)
        h0 = FourierLSH(bits=256, seed=0).encode(v)
        h1 = FourierLSH(bits=256, seed=1).encode(v)
        assert not np.array_equal(h0, h1)


class TestValidation:
    def test_bits_must_be_positive(self):
        with pytest.raises(ValueError):
            FourierLSH(bits=0)
        with pytest.raises(ValueError):
            FourierLSH(bits=-1)

    def test_repr(self):
        assert repr(FourierLSH(bits=128)) == "FourierLSH(bits=128, seed=0)"
