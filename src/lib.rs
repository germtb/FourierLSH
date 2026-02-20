use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use rustfft::{num_complex::Complex32, FftPlanner};

// ---------------------------------------------------------------------------
// Hamming distance — NEON SIMD on aarch64, u64 popcount fallback elsewhere
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn hamming_one(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::aarch64::*;
    let len = a.len();
    let n_16 = len / 16;
    let mut total: u32 = 0;

    unsafe {
        let mut acc = vdupq_n_u8(0);
        for chunk in 0..n_16 {
            let va = vld1q_u8(a.as_ptr().add(chunk * 16));
            let vb = vld1q_u8(b.as_ptr().add(chunk * 16));
            let xor = veorq_u8(va, vb);
            let bits = vcntq_u8(xor);
            acc = vaddq_u8(acc, bits);

            // Flush every 255 chunks to prevent u8 saturation
            if (chunk & 0xFF) == 0xFF {
                total += vaddlvq_u8(acc) as u32;
                acc = vdupq_n_u8(0);
            }
        }
        total += vaddlvq_u8(acc) as u32;
    }

    let tail_off = n_16 * 16;
    for j in tail_off..len {
        total += (a[j] ^ b[j]).count_ones();
    }
    total
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn hamming_one(a: &[u8], b: &[u8]) -> u32 {
    let code_size = a.len();
    let n_u64 = code_size / 8;
    let tail = code_size % 8;
    let mut dist: u32 = 0;

    for j in 0..n_u64 {
        let off = j * 8;
        let va = u64::from_ne_bytes(a[off..off + 8].try_into().unwrap());
        let vb = u64::from_ne_bytes(b[off..off + 8].try_into().unwrap());
        dist += (va ^ vb).count_ones();
    }

    let tail_off = n_u64 * 8;
    for j in 0..tail {
        dist += (a[tail_off + j] ^ b[tail_off + j]).count_ones();
    }
    dist
}

// ---------------------------------------------------------------------------
// Helper: get contiguous data pointer from a 2D array
// ---------------------------------------------------------------------------

/// Get a flat contiguous slice for a C-contiguous 2D array, plus the row stride.
/// Returns (flat_slice, code_size) so row i = &flat[i*code_size .. (i+1)*code_size].
fn get_contiguous_rows<'a>(arr: &'a numpy::ndarray::ArrayView2<'a, u8>) -> (&'a [u8], usize) {
    let n = arr.shape()[0];
    let code_size = arr.shape()[1];
    // Try to get the full contiguous slice
    let flat = arr
        .as_slice()
        .expect("array must be C-contiguous");
    assert_eq!(flat.len(), n * code_size);
    (flat, code_size)
}

// ---------------------------------------------------------------------------
// Hamming distances: single query vs database
// ---------------------------------------------------------------------------

#[pyfunction]
fn hamming_distances<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'py, u8>,
    database: PyReadonlyArray2<'py, u8>,
) -> Bound<'py, PyArray1<i32>> {
    let query = query.as_slice().expect("query must be contiguous");
    let db = database.as_array();
    let n = db.shape()[0];
    let (flat, code_size) = get_contiguous_rows(&db);

    let distances: Vec<i32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &flat[i * code_size..(i + 1) * code_size];
            hamming_one(query, row) as i32
        })
        .collect();

    PyArray1::from_vec(py, distances)
}

// ---------------------------------------------------------------------------
// Batch Hamming top-k with rayon parallelism
// ---------------------------------------------------------------------------

#[pyfunction]
fn hamming_top_k<'py>(
    py: Python<'py>,
    queries: PyReadonlyArray2<'py, u8>,
    database: PyReadonlyArray2<'py, u8>,
    k: usize,
) -> Bound<'py, PyArray2<i32>> {
    let qs = queries.as_array();
    let db = database.as_array();
    let n_queries = qs.shape()[0];
    let n_db = db.shape()[0];
    let actual_k = k.min(n_db);

    let (db_flat, db_cs) = get_contiguous_rows(&db);
    let (q_flat, q_cs) = get_contiguous_rows(&qs);

    // Parallel over queries
    let results: Vec<Vec<i32>> = (0..n_queries)
        .into_par_iter()
        .map(|qi| {
            let q = &q_flat[qi * q_cs..(qi + 1) * q_cs];

            let mut dists: Vec<(u32, i32)> = (0..n_db)
                .map(|i| {
                    let row = &db_flat[i * db_cs..(i + 1) * db_cs];
                    (hamming_one(q, row), i as i32)
                })
                .collect();

            dists.select_nth_unstable_by_key(actual_k - 1, |&(d, _)| d);
            dists[..actual_k].sort_unstable_by_key(|&(d, _)| d);

            dists[..actual_k].iter().map(|&(_, idx)| idx).collect()
        })
        .collect();

    PyArray2::from_vec2(py, &results).expect("failed to create result array")
}

// ---------------------------------------------------------------------------
// Rust-native FourierLSH encoder
// ---------------------------------------------------------------------------

/// Generate sign-flip masks for each round using xorshift64.
/// Returns n_rounds masks, each of length dim. Each value is either 0 or (1<<31).
fn generate_sign_masks(seed: u64, dim: usize, n_rounds: usize) -> Vec<Vec<u32>> {
    let mut state = seed.wrapping_add(1);
    let mut masks = Vec::with_capacity(n_rounds);
    for _ in 0..n_rounds {
        let mut mask = Vec::with_capacity(dim);
        for _ in 0..dim {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            mask.push(((state >> 63) as u32) << 31);
        }
        masks.push(mask);
    }
    masks
}

/// Encode a batch of f32 vectors to packed binary hashes using FFT + sign flips.
#[pyfunction]
fn fourier_encode<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'py, f32>,
    bits: usize,
    seed: u64,
) -> Bound<'py, PyArray2<u8>> {
    let vecs = vectors.as_array();
    let n = vecs.shape()[0];
    let dim = vecs.shape()[1];
    let code_size = (bits + 7) / 8;

    // Informative bits per FFT round: for real input of length d, the rFFT
    // produces d/2+1 complex coefficients, but the DC (index 0) and Nyquist
    // (index d/2, even d only) coefficients are purely real — their imaginary
    // parts are always 0, so those sign bits are uninformative. We skip them.
    let n_coeffs = dim / 2 + 1;
    let uninformative = if dim % 2 == 0 { 2 } else { 1 }; // DC imag + Nyquist imag
    let bits_per_round = 2 * n_coeffs - uninformative;
    let n_rounds = (bits + bits_per_round - 1) / bits_per_round;
    let masks = generate_sign_masks(seed, dim, n_rounds);

    // Pre-plan FFT (thread-safe, shared across all workers)
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(dim);
    let scratch_len = fft.get_inplace_scratch_len();

    // Get flat contiguous data
    let flat = vecs.as_slice().expect("vectors must be C-contiguous");

    // Parallel encode
    let packed: Vec<Vec<u8>> = (0..n)
        .into_par_iter()
        .map(|vi| {
            let row = &flat[vi * dim..(vi + 1) * dim];

            let mut all_bits: Vec<bool> = Vec::with_capacity(bits);
            let mut bits_remaining = bits;
            let n_coeffs = dim / 2 + 1;
            let mut scratch = vec![Complex32::default(); scratch_len];

            for mask in &masks {
                let take = bits_per_round.min(bits_remaining);

                // Apply sign flip and prepare complex input
                let mut buffer: Vec<Complex32> = row
                    .iter()
                    .zip(mask.iter())
                    .map(|(&v, &m)| {
                        let flipped = f32::from_bits(v.to_bits() ^ m);
                        Complex32::new(flipped, 0.0)
                    })
                    .collect();

                // In-place FFT
                fft.process_with_scratch(&mut buffer, &mut scratch);

                // Extract sign bits: for each coefficient, real then imag.
                // Skip imaginary parts that are always 0 (DC and Nyquist).
                let nyquist = if dim % 2 == 0 { dim / 2 } else { usize::MAX };
                let mut count = 0;
                for c in 0..n_coeffs {
                    if count >= take {
                        break;
                    }
                    all_bits.push(buffer[c].re >= 0.0);
                    count += 1;
                    if count >= take {
                        break;
                    }
                    // Skip DC imag (c==0) and Nyquist imag (c==dim/2 for even dim)
                    if c == 0 || c == nyquist {
                        continue;
                    }
                    all_bits.push(buffer[c].im >= 0.0);
                    count += 1;
                }

                bits_remaining -= take;
                if bits_remaining == 0 {
                    break;
                }
            }

            // Pack bits into bytes (MSB first, matching numpy packbits)
            let mut packed = vec![0u8; code_size];
            for (i, &b) in all_bits.iter().enumerate().take(bits) {
                if b {
                    packed[i / 8] |= 1 << (7 - (i % 8));
                }
            }
            packed
        })
        .collect();

    PyArray2::from_vec2(py, &packed).expect("failed to create result array")
}

// ---------------------------------------------------------------------------
// Walsh-Hadamard Transform encoder (for benchmarking vs FFT)
// ---------------------------------------------------------------------------

/// In-place Fast Walsh-Hadamard Transform on a f32 slice.
/// Length must be a power of 2.
#[inline]
fn fwht_inplace(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

/// Next power of 2 >= n.
#[inline]
fn next_pow2(n: usize) -> usize {
    if n.is_power_of_two() {
        n
    } else {
        1 << (usize::BITS - (n - 1).leading_zeros())
    }
}

/// Generate sign-flip masks for WHT rounds. Same RNG as FFT encoder
/// but generates `padded_dim` signs per round (to cover padding).
fn generate_sign_masks_wht(seed: u64, padded_dim: usize, n_rounds: usize) -> Vec<Vec<u32>> {
    let mut state = seed.wrapping_add(1);
    let mut masks = Vec::with_capacity(n_rounds);
    for _ in 0..n_rounds {
        let mut mask = Vec::with_capacity(padded_dim);
        for _ in 0..padded_dim {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            mask.push(((state >> 63) as u32) << 31);
        }
        masks.push(mask);
    }
    masks
}

/// Encode a batch of f32 vectors using WHT + sign flips (for benchmarking).
#[pyfunction]
fn hadamard_encode<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'py, f32>,
    bits: usize,
    seed: u64,
) -> Bound<'py, PyArray2<u8>> {
    let vecs = vectors.as_array();
    let n = vecs.shape()[0];
    let dim = vecs.shape()[1];
    let code_size = (bits + 7) / 8;

    let padded_dim = next_pow2(dim);
    let bits_per_round = padded_dim; // WHT gives padded_dim real outputs = padded_dim sign bits
    let n_rounds = (bits + bits_per_round - 1) / bits_per_round;
    let masks = generate_sign_masks_wht(seed, padded_dim, n_rounds);

    let flat = vecs.as_slice().expect("vectors must be C-contiguous");

    let packed: Vec<Vec<u8>> = (0..n)
        .into_par_iter()
        .map(|vi| {
            let row = &flat[vi * dim..(vi + 1) * dim];

            let mut all_bits: Vec<bool> = Vec::with_capacity(bits);
            let mut bits_remaining = bits;

            for mask in &masks {
                let take = bits_per_round.min(bits_remaining);

                // Zero-pad to power of 2 and apply sign flips
                let mut buffer = vec![0.0f32; padded_dim];
                for j in 0..dim {
                    buffer[j] = f32::from_bits(row[j].to_bits() ^ mask[j]);
                }
                // Sign flips on padding positions (mask[dim..padded_dim]) flip zeros,
                // which stay zero — but we include them for consistency with FALCONN.
                for j in dim..padded_dim {
                    // padding is 0.0, sign flip of 0.0 is -0.0, both map to sign bit 1
                    // so we just leave buffer[j] = 0.0
                    let _ = mask[j]; // consume the mask entry
                }

                // In-place WHT
                fwht_inplace(&mut buffer);

                // Extract sign bits from all padded_dim real outputs
                for j in 0..take {
                    all_bits.push(buffer[j] >= 0.0);
                }

                bits_remaining -= take;
                if bits_remaining == 0 {
                    break;
                }
            }

            let mut packed = vec![0u8; code_size];
            for (i, &b) in all_bits.iter().enumerate().take(bits) {
                if b {
                    packed[i / 8] |= 1 << (7 - (i % 8));
                }
            }
            packed
        })
        .collect();

    PyArray2::from_vec2(py, &packed).expect("failed to create result array")
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hamming_distances, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(fourier_encode, m)?)?;
    m.add_function(wrap_pyfunction!(hadamard_encode, m)?)?;
    Ok(())
}
