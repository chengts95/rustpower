#![allow(unused)]
use std::f64::consts::PI;

use super::new_dsdvbus2::JacobianPattern2;
use super::new_dsdvbus3::fill_jacobian_v3;
use super::solver::Solve;
use super::sparse::slice::*;
use nalgebra::*;
use nalgebra_sparse::*;
use num_complex::Complex64;
use num_traits::Zero;

// Re-export old implementations so the assembly benchmark can still access
// them via `crate::basic::newtonpf::{newton_pf_old, newton_pf_v0, …}`.
#[cfg(feature = "klu")]
pub(crate) use crate::basic::pf_old_impl::{
    assemble_f, build_jacobian, build_jacobian_cached, newton_pf_old, newton_pf_v0, JacobianCache,
};

// ─── Slice trait (public: used by test_jacobian_pattern) ─────────────────────

/// Slices blocks and column spans from a CSC matrix.
pub trait Slice {
    type Mat;
    fn block(&self, start_pos: (usize, usize), shape: (usize, usize)) -> Self::Mat;
    fn columns(&self, start_col: usize, end_col: usize) -> Self::Mat;
}

impl<T: Clone + Zero + Scalar + ClosedAddAssign> Slice for CscMatrix<T> {
    type Mat = CscMatrix<T>;

    #[inline(always)]
    fn block(&self, start_pos: (usize, usize), shape: (usize, usize)) -> Self::Mat {
        slice_csc_matrix_block(self, start_pos, shape)
    }

    #[inline(always)]
    fn columns(&self, start_col: usize, end_col: usize) -> Self::Mat {
        slice_csc_matrix(self, start_col, end_col)
    }
}

// ─── Default solver: newton_pf ────────────────────────────────────────────────

/// Newton-Raphson power flow under the `[PQ | PV | slack]` bus ordering.
/// Branch-free Jacobian assembly via `JacobianPattern2` + `fill_jacobian_v2`.
///
/// Requires `Ybus`, `Sbus`, `v_init` already permuted into `[PQ | PV | slack]`:
/// PQ buses at indices `0..npq`, PV at `npq..npq+npv`, slack at `npq+npv..`.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn newton_pf<Solver: Solve>(
    Ybus: &CscMatrix<Complex64>,
    Sbus: &DVector<Complex64>,
    v_init: &DVector<Complex64>,
    npv: usize,
    npq: usize,
    tolerance: Option<f64>,
    max_iter: Option<usize>,
    solver: &mut Solver,
) -> Result<(DVector<Complex64>, usize), (String, DVector<Complex64>, usize)> {
    let mut v = v_init.clone();
    let max_iter = max_iter.unwrap_or(100);
    let tol = tolerance.unwrap_or(1e-6);

    let j_pattern = JacobianPattern2::build_from_permuted(
        Ybus.col_offsets(),
        Ybus.row_indices(),
        npv,
        npq,
    );
    let n_state = npv + 2 * npq;
    let mut j_values = vec![0.0; j_pattern.nnz_j];

    let n_bus = npv + npq;
    let mut mis = &v.component_mul(&(Ybus * &v).conjugate()) - Sbus;
    let mut F = DVector::zeros(n_state);
    assemble_f_v2(&mut F, n_bus, &mis, n_state, npq);
    if F.norm() < tol {
        return Ok((v, 0));
    }

    let mut v_m = v.map(|e| e.simd_modulus());
    let mut v_a = v.map(|e| e.simd_argument());
    let mut v_norm = v.map(|e| e.simd_signum());

    let Ap = unsafe {
        std::slice::from_raw_parts_mut(
            j_pattern.j_col_ptrs.as_ptr() as *mut usize,
            j_pattern.j_col_ptrs.len(),
        )
    };
    let Ai = unsafe {
        std::slice::from_raw_parts_mut(
            j_pattern.j_row_indices.as_ptr() as *mut usize,
            j_pattern.j_row_indices.len(),
        )
    };

    for it in 0..max_iter {
        let ibus = Ybus * &v;
        let s_calc = v.component_mul(&ibus.map(|e| e.conj()));

        fill_jacobian_v3(
            Ybus,
            v.as_slice(),
            v_norm.as_slice(),
            s_calc.as_slice(),
            &j_pattern,
            npv,
            npq,
            &mut j_values,
        );

        let _ = solver.solve(
            Ap,
            Ai,
            j_values.as_mut_slice(),
            F.data.as_mut_slice(),
            n_state,
        );

        let dx = &F;

        // Angle update: all non-slack buses.
        v_a.rows_range_mut(0..n_bus)
            .zip_apply(&dx.rows_range(0..n_bus), |a, b| {
                *a -= b;
                *a = a.rem_euclid(2.0 * PI);
            });
        // Magnitude update: PQ buses only (at 0..npq in PQ-first ordering).
        let mut vm_pq = v_m.rows_range_mut(0..npq);
        vm_pq.zip_apply(&dx.rows_range(n_bus..n_state), |a, b| *a -= b);

        v_norm.zip_apply(&v_a, |a, va| *a = Complex64::from_polar(1.0, va));
        v.zip_zip_apply(&v_norm, &v_m, |a, e, vm| *a = vm * e);

        v.component_mul(&(Ybus * &v).conjugate())
            .sub_to(Sbus, &mut mis);
        assemble_f_v2(&mut F, n_bus, &mis, n_state, npq);

        if F.norm() < tol {
            return Ok((v, it));
        }
    }

    Err((String::from("Did not converge!"), v, max_iter))
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Mismatch RHS under `[PQ | PV | slack]` ordering.
///
/// `F[0..n_bus]`      = Re(mis[0..n_bus])
/// `F[n_bus..n_state]` = Im(mis[0..npq])`
#[inline(always)]
pub(crate) fn assemble_f_v2(
    f: &mut DVector<f64>,
    n_bus: usize,
    mis: &DVector<Complex64>,
    num_state: usize,
    npq: usize,
) {
    f.rows_range_mut(0..n_bus)
        .zip_apply(&mis.rows_range(0..n_bus), |a, b| *a = b.simd_real());
    f.rows_range_mut(n_bus..num_state)
        .zip_apply(&mis.rows_range(0..npq), |a, b| *a = b.simd_imaginary());
}
