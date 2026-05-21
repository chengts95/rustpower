//! Second-generation symbolic + fill assuming bus order `[PQ | PV | slack]`.
//!
//! Switching the convention from V1's `[PV | PQ | slack]` removes every
//! conditional from the assembly hot path.
//!
//! Within each Ybus column the row indices split as
//!   `[0, pq_end)` -> PQ rows  (write J11+J21 and J12+J22),
//!   `[pq_end, active_end)` -> PV rows  (write J11 and J12 only),
//!   `[active_end, end)` -> slack rows  (skipped),
//! both boundaries precomputed in the symbolic phase. Across columns the loop
//! splits at `npq`: the first `npq` columns are PQ buses (emit both theta and
//! |V| columns), the next `npv` are PV buses (emit only theta). All four
//! inner loops are branch-free sweeps.

use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

/// Symbolic structure of the reduced Jacobian under `[PQ | PV | slack]` order.
/// Computed once from `Ybus.col_offsets`, `Ybus.row_indices`, and `(npv, npq)`.
pub struct JacobianPattern2 {
    pub nnz_j: usize,
    pub j_col_ptrs: Vec<usize>,
    pub j_row_indices: Vec<usize>,

    // Per-Ybus-column row boundaries (offsets in the column's row slice).
    pub pq_ends: Vec<usize>,     // length n_active; offset where PQ rows end
    pub active_ends: Vec<usize>, // length n_active; offset where active rows end

    // Per-J-column start offsets in j_values (absolute).
    pub j11_starts: Vec<usize>, // length n_active
    pub j21_starts: Vec<usize>, // length n_active
    pub j12_starts: Vec<usize>, // length npq
    pub j22_starts: Vec<usize>, // length npq

    // Absolute offset of the (k,k) diagonal entry in Ybus.values.
    pub diag_ptrs: Vec<usize>, // length n_active
}

impl JacobianPattern2 {
    pub fn build_from_permuted(
        y_col_ptrs: &[usize],
        y_row_indices: &[usize],
        npv: usize,
        npq: usize,
    ) -> Self {
        let n_active = npv + npq;
        let n_j_cols = n_active + npq;

        let mut j_col_ptrs = Vec::with_capacity(n_j_cols + 1);
        let mut j_row_indices = Vec::new();
        j_col_ptrs.push(0);

        let mut pq_ends = vec![0; n_active];
        let mut active_ends = vec![0; n_active];
        let mut j11_starts = vec![0; n_active];
        let mut j21_starts = vec![0; n_active];
        let mut j12_starts = vec![0; npq];
        let mut j22_starts = vec![0; npq];
        let mut diag_ptrs = vec![0; n_active];

        let mut current_nnz = 0;

        // Phase 1: theta columns -- one per active bus (PQ first, then PV).
        for k in 0..n_active {
            let start = y_col_ptrs[k];
            let row_slice = &y_row_indices[start..y_col_ptrs[k + 1]];

            // Two-cut partition of the sorted row indices.
            let idx_pq_end = row_slice.partition_point(|&r| r < npq);
            let idx_active_end = row_slice.partition_point(|&r| r < n_active);

            pq_ends[k] = idx_pq_end;
            active_ends[k] = idx_active_end;

            if let Ok(diag_idx) = row_slice.binary_search(&k) {
                diag_ptrs[k] = start + diag_idx;
            }

            // J11 (rows = all active rows of col k; J_red row = Ybus row).
            j11_starts[k] = current_nnz;
            for offset in 0..idx_active_end {
                j_row_indices.push(row_slice[offset]);
            }
            current_nnz += idx_active_end;

            // J21 (rows = PQ rows of col k; J_red Q-eq row = n_active + Ybus row).
            j21_starts[k] = current_nnz;
            for offset in 0..idx_pq_end {
                let r = row_slice[offset];
                j_row_indices.push(n_active + r);
            }
            current_nnz += idx_pq_end;

            j_col_ptrs.push(current_nnz);
        }

        // Phase 2: |V| columns -- one per PQ bus only.
        for k in 0..npq {
            let start = y_col_ptrs[k];
            let row_slice = &y_row_indices[start..y_col_ptrs[k + 1]];
            let idx_pq_end = pq_ends[k];
            let idx_active_end = active_ends[k];

            // J12 (rows = all active rows of col k).
            j12_starts[k] = current_nnz;
            for offset in 0..idx_active_end {
                j_row_indices.push(row_slice[offset]);
            }
            current_nnz += idx_active_end;

            // J22 (rows = PQ rows of col k).
            j22_starts[k] = current_nnz;
            for offset in 0..idx_pq_end {
                let r = row_slice[offset];
                j_row_indices.push(n_active + r);
            }
            current_nnz += idx_pq_end;

            j_col_ptrs.push(current_nnz);
        }

        Self {
            nnz_j: current_nnz,
            j_col_ptrs,
            j_row_indices,
            pq_ends,
            active_ends,
            j11_starts,
            j21_starts,
            j12_starts,
            j22_starts,
            diag_ptrs,
        }
    }
}

/// Numeric fill under `[PQ | PV | slack]` ordering. Branch-free: the outer
/// column loop splits at `npq`, the inner row loop splits at `pq_ends[k]`.
#[allow(non_snake_case)]
#[inline(never)]
pub fn fill_jacobian_v2(
    Ybus: &CscMatrix<Complex64>,
    v: &[Complex64],
    Vnorm: &[Complex64],
    ibus: &[Complex64],
    pattern: &JacobianPattern2,
    npv: usize,
    npq: usize,
    j_values: &mut [f64],
) {
    let y_col_offsets = Ybus.col_offsets();
    let y_row_indices = Ybus.row_indices();
    let y_vals = Ybus.values();
    let n_active = npv + npq;

    // ===================== PQ bus loop =====================
    // Emit both theta column and |V| column for each PQ bus.
    for k in 0..npq {
        let y_start = y_col_offsets[k];
        let pq_end = pattern.pq_ends[k];
        let active_end = pattern.active_ends[k];

        let ek = v[k].re;
        let fk = v[k].im;
        let enk = Vnorm[k].re;
        let fnk = Vnorm[k].im;
        let Ire_k = ibus[k].re;
        let Iim_k = ibus[k].im;

        let diag_offset = pattern.diag_ptrs[k] - y_start;

        let j_ptr = j_values.as_mut_ptr();
        let out_j11 = unsafe {
            std::slice::from_raw_parts_mut(j_ptr.add(pattern.j11_starts[k]), active_end)
        };
        let out_j21 = unsafe {
            std::slice::from_raw_parts_mut(j_ptr.add(pattern.j21_starts[k]), pq_end)
        };
        let out_j12 = unsafe {
            std::slice::from_raw_parts_mut(j_ptr.add(pattern.j12_starts[k]), active_end)
        };
        let out_j22 = unsafe {
            std::slice::from_raw_parts_mut(j_ptr.add(pattern.j22_starts[k]), pq_end)
        };

        // PQ-row slice: writes J11+J21 and J12+J22 (full participation).
        for offset in 0..pq_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;
            let Vm_re = Y_ik.re * enk - Y_ik.im * fnk;
            let Vm_im = Y_ik.re * fnk + Y_ik.im * enk;

            let ei = v[i].re;
            let fi = v[i].im;

            out_j11[offset] = fi * Va_re - ei * Va_im;
            out_j21[offset] = -(ei * Va_re + fi * Va_im);
            out_j12[offset] = ei * Vm_re + fi * Vm_im;
            out_j22[offset] = fi * Vm_re - ei * Vm_im;
        }

        // PV-row slice: writes J11 and J12 only.
        for offset in pq_end..active_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;
            let Vm_re = Y_ik.re * enk - Y_ik.im * fnk;
            let Vm_im = Y_ik.re * fnk + Y_ik.im * enk;

            let ei = v[i].re;
            let fi = v[i].im;

            out_j11[offset] = fi * Va_re - ei * Va_im;
            out_j12[offset] = ei * Vm_re + fi * Vm_im;
        }

        // Diagonal corrections (k is PQ here; all four blocks).
        unsafe {
            *j_values.get_unchecked_mut(pattern.j11_starts[k] + diag_offset) +=
                ek * Iim_k - fk * Ire_k;
            *j_values.get_unchecked_mut(pattern.j21_starts[k] + diag_offset) +=
                ek * Ire_k + fk * Iim_k;
            *j_values.get_unchecked_mut(pattern.j12_starts[k] + diag_offset) +=
                enk * Ire_k + fnk * Iim_k;
            *j_values.get_unchecked_mut(pattern.j22_starts[k] + diag_offset) +=
                fnk * Ire_k - enk * Iim_k;
        }
    }

    // ===================== PV bus loop =====================
    // Emit only theta column.
    for k in npq..n_active {
        let y_start = y_col_offsets[k];
        let pq_end = pattern.pq_ends[k];
        let active_end = pattern.active_ends[k];

        let ek = v[k].re;
        let fk = v[k].im;
        let Ire_k = ibus[k].re;
        let Iim_k = ibus[k].im;

        let diag_offset = pattern.diag_ptrs[k] - y_start;

        let j_ptr = j_values.as_mut_ptr();
        let out_j11 = unsafe {
            std::slice::from_raw_parts_mut(j_ptr.add(pattern.j11_starts[k]), active_end)
        };
        let out_j21 = unsafe {
            std::slice::from_raw_parts_mut(j_ptr.add(pattern.j21_starts[k]), pq_end)
        };

        // PQ-row slice: writes J11 + J21.
        for offset in 0..pq_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let ei = v[i].re;
            let fi = v[i].im;

            out_j11[offset] = fi * Va_re - ei * Va_im;
            out_j21[offset] = -(ei * Va_re + fi * Va_im);
        }

        // PV-row slice: writes J11 only.
        for offset in pq_end..active_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let ei = v[i].re;
            let fi = v[i].im;

            out_j11[offset] = fi * Va_re - ei * Va_im;
        }

        // Diagonal correction (k is PV here; only J11).
        unsafe {
            *j_values.get_unchecked_mut(pattern.j11_starts[k] + diag_offset) +=
                ek * Iim_k - fk * Ire_k;
        }
    }
}
