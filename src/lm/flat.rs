//! Phase 2 — the flat view (doc §3.3) and the `fill_kkt` dispatchers (§3.4).
//!
//! [`FlatLayout`] materializes **only** the CSC triple a direct solver needs
//! (`col_offsets`, `row_indices`, and the caller-owned `values`): the global
//! `[μI+H  Jᵀ; J  −I]` matrix as one CSC with
//!
//! ```text
//! δ-column c     (0..n)  : [H col c  | J col c ]    rows [graph rows | graph rows + n]
//! s-column n + c (0..n)  : [Jᵀ col c | −I diag ]    rows [graph rows | n + c]
//! ```
//!
//! Column pointers are affine in the shared graph pattern
//! (`gp[c] = 2·cs[c]`, `gp[n+c] = 2·nnz + cs[c] + c`), so nothing beyond the
//! CSC triple is stored: every fill position is re-derived from the column's
//! own base and the Ybus structure inside the kernels (`FLAT = true`).
//!
//! The `−I` block is constant: stamped once via [`FlatLayout::stamp_neg_i`]
//! after allocation, never re-written by any fill (the write-once coverage
//! test asserts exactly that).

use num_complex::Complex64;
use nalgebra_sparse::CscMatrix;

use super::kernels::{fill_h, fill_jt};
use super::pattern::KktPattern;
use crate::basic::new_dsdvbus4::fill_jacobian_v4;

/// The global CSC of the LM augmented system, symbolic part.
pub struct FlatLayout {
    /// `n_state = n_active + n_pq`; the global matrix is `2·n_state` square.
    pub n_state: usize,
    /// `3·nnz + n_state`.
    pub nnz_flat: usize,
    /// Length `2·n_state + 1`.
    pub col_offsets: Vec<usize>,
    /// Length `nnz_flat`, sorted within every column.
    pub row_indices: Vec<usize>,
}

impl FlatLayout {
    pub fn build(pat: &KktPattern) -> Self {
        let cache = &pat.cache;
        let n = cache.n_active() + cache.n_pq();
        let nnz = pat.graph.nnz;
        let cs = &pat.graph.col_starts;

        let mut col_offsets = Vec::with_capacity(2 * n + 1);
        let mut row_indices = Vec::with_capacity(3 * nnz + n);

        // δ-columns: [H segment | J segment (rows shifted by n)].
        for c in 0..n {
            col_offsets.push(2 * cs[c]);
            let rows = pat.graph.col_rows(c);
            row_indices.extend_from_slice(rows);
            row_indices.extend(rows.iter().map(|r| r + n));
        }
        // s-columns: [Jᵀ segment | −I entry].
        for c in 0..n {
            col_offsets.push(2 * nnz + cs[c] + c);
            row_indices.extend_from_slice(pat.graph.col_rows(c));
            row_indices.push(n + c);
        }
        col_offsets.push(3 * nnz + n);

        Self {
            n_state: n,
            nnz_flat: 3 * nnz + n,
            col_offsets,
            row_indices,
        }
    }

    /// Stamp the constant `−I` block: the last entry of every s-column.
    /// Call once after allocating the values array; no fill ever rewrites
    /// these slots.
    pub fn stamp_neg_i(&self, values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.nnz_flat);
        for c in 0..self.n_state {
            values[self.col_offsets[self.n_state + c + 1] - 1] = -1.0;
        }
    }
}

/// Block view (§3.4): one values array, four disjoint block slices
/// `[J | H | Jᵀ | −I]`, three numeric fills plus the constant `−I` stamp.
#[allow(clippy::too_many_arguments)]
pub fn fill_kkt(
    ybus: &CscMatrix<Complex64>,
    pat: &KktPattern,
    v: &[Complex64],
    vnorm: &[Complex64],
    scalc: &[Complex64],
    r: &[f64],
    values: &mut [f64],
) {
    debug_assert_eq!(values.len(), pat.nnz_total);
    let cache = &pat.cache;
    let (npq, npv) = (cache.n_pq(), cache.n_active() - cache.n_pq());
    let cs = &pat.graph.col_starts;
    let nnz = pat.graph.nnz;

    let (j_vals, rest) = values.split_at_mut(nnz);
    let (h_vals, rest) = rest.split_at_mut(nnz);
    let (jt_vals, d_vals) = rest.split_at_mut(nnz);

    fill_jacobian_v4::<false>(
        ybus, v, vnorm, scalc,
        cs, cache.pq_ends(), cache.active_ends(), cache.diag_ptrs(),
        npv, npq, j_vals,
    );
    fill_h::<false>(ybus, pat, v, r, h_vals);
    fill_jt::<false>(ybus, pat, j_vals.as_ptr(), jt_vals.as_mut_ptr());
    for x in d_vals.iter_mut() {
        *x = -1.0;
    }
}

/// Flat view: the same three fills writing directly into the global CSC's
/// values array. The `−I` block is not touched — stamp it once with
/// [`FlatLayout::stamp_neg_i`].
#[allow(clippy::too_many_arguments)]
pub fn fill_kkt_flat(
    ybus: &CscMatrix<Complex64>,
    pat: &KktPattern,
    flat: &FlatLayout,
    v: &[Complex64],
    vnorm: &[Complex64],
    scalc: &[Complex64],
    r: &[f64],
    values: &mut [f64],
) {
    debug_assert_eq!(values.len(), flat.nnz_flat);
    let cache = &pat.cache;
    let (npq, npv) = (cache.n_pq(), cache.n_active() - cache.n_pq());
    let cs = &pat.graph.col_starts;

    fill_jacobian_v4::<true>(
        ybus, v, vnorm, scalc,
        cs, cache.pq_ends(), cache.active_ends(), cache.diag_ptrs(),
        npv, npq, values,
    );
    fill_h::<true>(ybus, pat, v, r, values);
    fill_jt::<true>(ybus, pat, values.as_ptr(), values.as_mut_ptr());
}
