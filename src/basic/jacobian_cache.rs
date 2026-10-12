//! V4 symbolic Jacobian cache for bus order `[PQ | PV | slack]`.
//!
//! Quadrant starts are derived from CSC column pointers and segment lengths.
//! Unlike the V3 baseline's `JacobianPattern2`, no quadrant-start tables are stored.

#[derive(Clone)]
pub struct JacobianCache {
    pub j_col_ptrs: Vec<usize>,
    pub j_row_indices: Vec<usize>,
    pub pq_ends: Vec<usize>,
    pub active_ends: Vec<usize>,
    pub diag_ptrs: Vec<usize>,
}

impl JacobianCache {
    pub fn build_from_permuted(
        y_col_ptrs: &[usize],
        y_row_indices: &[usize],
        npv: usize,
        npq: usize,
    ) -> Self {
        let n_active = npq + npv;
        let mut cache = Self {
            j_col_ptrs: Vec::with_capacity(n_active + npq + 1),
            j_row_indices: Vec::new(),
            pq_ends: Vec::with_capacity(n_active),
            active_ends: Vec::with_capacity(n_active),
            diag_ptrs: Vec::with_capacity(n_active),
        };

        for k in 0..n_active {
            let start = y_col_ptrs[k];
            let rows = &y_row_indices[start..y_col_ptrs[k + 1]];
            let (pq_end, active_end, diagonal) =
                super::new_dsdvbus2::column_boundaries::<true>(rows, k, npq, n_active);
            cache.pq_ends.push(pq_end);
            cache.active_ends.push(active_end);
            cache
                .diag_ptrs
                .push(start + diagonal.expect("Ybus is missing a structural diagonal entry"));
        }

        let nnz_theta: usize = cache
            .active_ends
            .iter()
            .zip(&cache.pq_ends)
            .map(|(active, pq)| active + pq)
            .sum();
        let nnz_magnitude: usize = cache.active_ends[..npq]
            .iter()
            .zip(&cache.pq_ends[..npq])
            .map(|(active, pq)| active + pq)
            .sum();
        cache.j_row_indices.reserve_exact(nnz_theta + nnz_magnitude);

        // Angle columns for active buses, followed by magnitude columns for PQ buses.
        // Both use the same ordered row segments: [active | shifted PQ].
        for k in (0..n_active).chain(0..npq) {
            let rows = &y_row_indices[y_col_ptrs[k]..y_col_ptrs[k + 1]];
            cache.j_col_ptrs.push(cache.j_row_indices.len());
            cache
                .j_row_indices
                .extend_from_slice(&rows[..cache.active_ends[k]]);
            cache
                .j_row_indices
                .extend(rows[..cache.pq_ends[k]].iter().map(|&i| n_active + i));
        }
        cache.j_col_ptrs.push(cache.j_row_indices.len());
        cache
    }
}

#[cfg(test)]
pub(super) fn mirror_positions(
    ybus: &nalgebra_sparse::CscMatrix<num_complex::Complex64>,
) -> Vec<usize> {
    let cp = ybus.col_offsets();
    let ri = ybus.row_indices();
    let mut mirror = vec![0; ybus.nnz()];
    for k in 0..ybus.ncols() {
        for p in cp[k]..cp[k + 1] {
            let i = ri[p];
            mirror[p] = cp[i]
                + ri[cp[i]..cp[i + 1]]
                    .binary_search(&k)
                    .expect("row traversal requires a symmetric Ybus pattern");
        }
    }
    mirror
}
