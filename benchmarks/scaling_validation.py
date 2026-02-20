#!/usr/bin/env python3
"""
Scaling Validation
==================

Does the measured encoding time actually follow O(L log d) for FourierLSH
and O(dL) for FAISS LSH? We fit the empirical data to the theoretical models
and check residuals.

If FourierLSH is truly O(L log d), then:
  T_fourier(L, d) = c1 * L * log2(d) + c0

If FAISS LSH is truly O(dL), then:
  T_faiss(L, d) = c1 * d * L + c0

We sweep L and d independently and check that the predicted model fits.
"""

import time
import numpy as np
from fourierlsh import FourierLSH

try:
    import faiss
except ImportError:
    print("ERROR: faiss-cpu required")
    raise SystemExit(1)


def time_fn(fn, warmup=3, repeats=7):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times)


def bench_fourier(data, bits):
    lsh = FourierLSH(bits=bits, seed=0)
    return time_fn(lambda: lsh.encode_batch(data))


def bench_faiss(data, bits):
    dim = data.shape[1]
    vecs = np.ascontiguousarray(data, dtype=np.float32)
    def encode():
        idx = faiss.IndexLSH(dim, bits)
        idx.add(vecs)
    return time_fn(encode)


# -----------------------------------------------------------------------
# Experiment 1: Fix d, sweep L — check scaling in L
# -----------------------------------------------------------------------

def sweep_L(dim, n_vecs=5000):
    print(f"\n{'='*70}")
    print(f"SWEEP L (fix d={dim}, n={n_vecs})")
    print(f"{'='*70}")

    rng = np.random.RandomState(42)
    data = rng.randn(n_vecs, dim).astype(np.float64)
    data /= np.linalg.norm(data, axis=1, keepdims=True)

    bit_widths = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    f_times = []
    x_times = []

    print(f"\n{'Bits':>6}  {'Fourier':>12}  {'FAISS':>12}  {'F ratio':>10}  {'X ratio':>10}")
    print("-" * 60)

    for bits in bit_widths:
        ft = bench_fourier(data, bits)
        xt = bench_faiss(data, bits)
        f_times.append(ft)
        x_times.append(xt)

        # Ratio vs previous
        f_ratio = f_times[-1] / f_times[-2] if len(f_times) > 1 else 0
        x_ratio = x_times[-1] / x_times[-2] if len(x_times) > 1 else 0
        print(f"{bits:>6}  {ft*1000:>10.2f}ms  {xt*1000:>10.2f}ms  "
              f"{f_ratio:>9.2f}x  {x_ratio:>9.2f}x")

    print()
    print("Theory check (when L doubles):")
    print(f"  O(L log d): time should ~2x  (ratio ≈ 2.0)")
    print(f"  O(dL):      time should ~2x  (ratio ≈ 2.0)")
    print(f"  Both linear in L, so ratio should be ~2.0 for both.")
    print(f"  The DIFFERENCE is in scaling with d (see sweep_d).")

    return bit_widths, f_times, x_times


# -----------------------------------------------------------------------
# Experiment 2: Fix L, sweep d — this is the key test
# -----------------------------------------------------------------------

def sweep_d(bits, n_vecs=5000):
    print(f"\n{'='*70}")
    print(f"SWEEP d (fix L={bits}, n={n_vecs})")
    print(f"{'='*70}")

    dims = [64, 128, 256, 512, 1024, 2048, 4096]

    f_times = []
    x_times = []

    print(f"\n{'dim':>6}  {'Fourier':>12}  {'FAISS':>12}  {'F ratio':>10}  {'X ratio':>10}")
    print("-" * 60)

    for dim in dims:
        rng = np.random.RandomState(42)
        data = rng.randn(n_vecs, dim).astype(np.float64)
        data /= np.linalg.norm(data, axis=1, keepdims=True)

        ft = bench_fourier(data, bits)
        xt = bench_faiss(data, bits)
        f_times.append(ft)
        x_times.append(xt)

        f_ratio = f_times[-1] / f_times[-2] if len(f_times) > 1 else 0
        x_ratio = x_times[-1] / x_times[-2] if len(x_times) > 1 else 0
        print(f"{dim:>6}  {ft*1000:>10.2f}ms  {xt*1000:>10.2f}ms  "
              f"{f_ratio:>9.2f}x  {x_ratio:>9.2f}x")

    print()
    print("Theory check (when d doubles):")
    print(f"  O(L log d): time should increase by log(2d)/log(d) ≈ 1 + 1/log2(d)")
    print(f"              e.g., d=512→1024: ratio ≈ 1 + 1/9 ≈ 1.11")
    print(f"              e.g., d=1024→2048: ratio ≈ 1 + 1/10 ≈ 1.10")
    print(f"  O(dL):      time should ~2x   (ratio ≈ 2.0)")
    print()

    # Fit models
    dims_arr = np.array(dims, dtype=np.float64)
    f_arr = np.array(f_times)
    x_arr = np.array(x_times)

    # For Fourier: T = c * L * log2(d) => T / L ∝ log2(d)
    # When L >= d, rounds = ceil(L/d), so T = c * ceil(L/d) * d * log2(d)
    # This is more complex. Let's fit: T = a * n_rounds * d * log2(d) + b
    n_rounds = np.ceil(bits / dims_arr).astype(int)
    fourier_model_x = n_rounds * dims_arr * np.log2(dims_arr)

    # For FAISS: T = c * d * L + b
    faiss_model_x = dims_arr * bits

    # Least squares fit
    A_f = np.column_stack([fourier_model_x, np.ones(len(dims))])
    c_f, _, _, _ = np.linalg.lstsq(A_f, f_arr, rcond=None)

    A_x = np.column_stack([faiss_model_x, np.ones(len(dims))])
    c_x, _, _, _ = np.linalg.lstsq(A_x, x_arr, rcond=None)

    f_pred = A_f @ c_f
    x_pred = A_x @ c_x

    f_r2 = 1 - np.sum((f_arr - f_pred)**2) / np.sum((f_arr - f_arr.mean())**2)
    x_r2 = 1 - np.sum((x_arr - x_pred)**2) / np.sum((x_arr - x_arr.mean())**2)

    print(f"Model fit (R² — closer to 1.0 = better fit):")
    print(f"  FourierLSH ~ c * ceil(L/d) * d * log2(d):  R² = {f_r2:.4f}")
    print(f"  FAISS LSH  ~ c * d * L:                     R² = {x_r2:.4f}")

    print(f"\nPredicted vs actual:")
    for i, dim in enumerate(dims):
        print(f"  d={dim:>5}: Fourier {f_arr[i]*1000:>8.2f}ms (pred {f_pred[i]*1000:>8.2f}ms)  "
              f"FAISS {x_arr[i]*1000:>8.2f}ms (pred {x_pred[i]*1000:>8.2f}ms)")

    return dims, f_times, x_times


# -----------------------------------------------------------------------
# Experiment 3: Crossover analysis
# -----------------------------------------------------------------------

def crossover_analysis():
    print(f"\n{'='*70}")
    print(f"CROSSOVER: at what L does FourierLSH become faster?")
    print(f"{'='*70}")

    n_vecs = 5000

    for dim in [128, 300, 512, 1024, 1536, 2048, 4096]:
        rng = np.random.RandomState(42)
        data = rng.randn(n_vecs, dim).astype(np.float64)
        data /= np.linalg.norm(data, axis=1, keepdims=True)

        crossover = None
        for bits in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
            ft = bench_fourier(data, bits)
            xt = bench_faiss(data, bits)
            if ft < xt and crossover is None:
                crossover = bits
                break

        if crossover:
            print(f"  d={dim:>5}: crossover at L={crossover} bits "
                  f"(theory: d/log2(d) = {dim/np.log2(dim):.0f})")
        else:
            print(f"  d={dim:>5}: no crossover found up to 8192 bits")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    # Sweep L at fixed d
    sweep_L(dim=300)
    sweep_L(dim=1536)

    # Sweep d at fixed L — the key test
    sweep_d(bits=1024)
    sweep_d(bits=4096)

    # Crossover points
    crossover_analysis()
