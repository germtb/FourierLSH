use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Hamming distance between one packed row and another.
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

/// Hamming distance between a single packed query and a database of packed hashes.
///
/// Processes 8 bytes at a time as u64 for SIMD-friendly popcount.
#[pyfunction]
fn hamming_distances<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'py, u8>,
    database: PyReadonlyArray2<'py, u8>,
) -> Bound<'py, PyArray1<i32>> {
    let query = query.as_slice().expect("query must be contiguous");
    let db = database.as_array();
    let n = db.shape()[0];

    let distances: Vec<i32> = (0..n)
        .map(|i| {
            let row = db.row(i);
            let row = row.as_slice().expect("database rows must be contiguous");
            hamming_one(query, row) as i32
        })
        .collect();

    PyArray1::from_vec(py, distances)
}

/// Batch Hamming top-k search: for each query, find the k nearest database rows.
///
/// Returns a (n_queries, k) array of database indices, sorted by Hamming distance.
/// This avoids Python-level per-query loops entirely.
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

    let mut results = vec![0i32; n_queries * actual_k];

    for qi in 0..n_queries {
        let q_row = qs.row(qi);
        let q = q_row.as_slice().expect("query rows must be contiguous");

        // Compute all distances
        let mut dists: Vec<(u32, i32)> = (0..n_db)
            .map(|i| {
                let row = db.row(i);
                let row = row.as_slice().expect("database rows must be contiguous");
                (hamming_one(q, row), i as i32)
            })
            .collect();

        // Partial sort to find top-k
        dists.select_nth_unstable_by_key(actual_k - 1, |&(d, _)| d);
        dists[..actual_k].sort_unstable_by_key(|&(d, _)| d);

        let offset = qi * actual_k;
        for (j, &(_, idx)) in dists[..actual_k].iter().enumerate() {
            results[offset + j] = idx;
        }
    }

    PyArray2::from_vec2(py, &results.chunks(actual_k).map(|c| c.to_vec()).collect::<Vec<_>>())
        .expect("failed to create result array")
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hamming_distances, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_top_k, m)?)?;
    Ok(())
}
