//! Layer 0 — `YbusAnalysisCache`: every offset the fill kernels will ever
//! need, computed once from the permuted Ybus CSC.
//!
//! Same idea as `JacobianPattern2` (new_dsdvbus2.rs): buses are pre-ordered
//! `[PQ | PV | slack]` and row indices inside each Ybus column are sorted,
//! so one `partition_point` cuts each column into contiguous type segments
//! and one `binary_search` locates the diagonal. These are cut positions,
//! not search tables — the numeric phase only does start-plus-offset
//! arithmetic with them.
//!
//! Reduced system (PF/LM): only the first `n_active = n_pq + n_pv` buses
//! take part; slack rows are cut away by `active_ends`. Full retention
//! (OPF): `n_pq = n_active = n_bus`, the cuts degenerate to whole columns.

use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

pub struct YbusAnalysisCache {
    n_pq: usize,
    n_active: usize,
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
    /// Per column: offset where PQ rows end (PQ rows are the sorted prefix).
    pq_ends: Vec<usize>,
    /// Per column: offset where active (PQ+PV) rows end; slack rows beyond.
    active_ends: Vec<usize>,
    /// Absolute offset of the diagonal entry in `Ybus.values`
    /// (same convention as `JacobianPattern2::diag_ptrs`).
    diag_ptrs: Vec<usize>,
    /// `diag_ptrs[k] − col_ptrs[k]`: the diagonal's offset inside column k.
    /// Because every block column replicates the Ybus row order (or a prefix
    /// of it), this one offset locates the diagonal in every block.
    diag_off: Vec<usize>,
    /// Per nnz `(i, j)`: offset of the mirror entry `(j, i)`.
    y_trans: Vec<usize>,
}

impl YbusAnalysisCache {
    pub fn build(ybus: &CscMatrix<Complex64>, n_pv: usize, n_pq: usize) -> Self {
        let n_active = n_pv + n_pq;
        let col_ptrs = ybus.col_offsets().to_vec();
        let row_indices = ybus.row_indices().to_vec();

        let mut pq_ends = vec![0usize; n_active];
        let mut active_ends = vec![0usize; n_active];
        let mut diag_ptrs = vec![0usize; n_active];
        let mut diag_off = vec![0usize; n_active];

        for k in 0..n_active {
            let start = col_ptrs[k];
            let rows = &row_indices[start..col_ptrs[k + 1]];
            pq_ends[k] = rows.partition_point(|&r| r < n_pq);
            active_ends[k] = rows.partition_point(|&r| r < n_active);
            let off = rows
                .binary_search(&k)
                .expect("Ybus is missing a structural diagonal entry");
            diag_ptrs[k] = start + off;
            diag_off[k] = off;
        }

        let mut y_trans = vec![0usize; row_indices.len()];
        for j in 0..ybus.ncols() {
            for p in col_ptrs[j]..col_ptrs[j + 1] {
                let i = row_indices[p];
                let mirror = &row_indices[col_ptrs[i]..col_ptrs[i + 1]];
                y_trans[p] = col_ptrs[i]
                    + mirror
                        .binary_search(&j)
                        .expect("Ybus pattern is not structurally symmetric");
            }
        }

        Self {
            n_pq,
            n_active,
            col_ptrs,
            row_indices,
            pq_ends,
            active_ends,
            diag_ptrs,
            diag_off,
            y_trans,
        }
    }

    pub fn n_pq(&self) -> usize {
        self.n_pq
    }
    pub fn n_active(&self) -> usize {
        self.n_active
    }
    pub fn col_ptrs(&self) -> &[usize] {
        &self.col_ptrs
    }
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }
    pub fn pq_ends(&self) -> &[usize] {
        &self.pq_ends
    }
    pub fn active_ends(&self) -> &[usize] {
        &self.active_ends
    }
    pub fn diag_ptrs(&self) -> &[usize] {
        &self.diag_ptrs
    }
    pub fn diag_off(&self) -> &[usize] {
        &self.diag_off
    }
    pub fn y_trans(&self) -> &[usize] {
        &self.y_trans
    }
}
