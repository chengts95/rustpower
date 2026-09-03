#![allow(unused)]
use std::f64::consts::PI;

use super::new_dsdvbus2::JacobianPattern2;

use bevy_ecs::prelude::Resource;

#[derive(Resource, Default)]
#[allow(non_snake_case)]
pub struct NewtonCache {
    pub npv: usize,
    pub npq: usize,
    pub j_pattern: Option<JacobianPattern2>,
    pub j_values: Vec<f64>,
    pub ibus: DVector<Complex64>,
    #[allow(non_snake_case)]
    pub F: DVector<f64>,
    pub s_calc: DVector<Complex64>,
}

use super::new_dsdvbus3::fill_jacobian_v3;
use super::solver::Solve;
use super::sparse::slice::*;
use nalgebra::*;
use nalgebra_sparse::*;
use num_complex::Complex64;
use num_traits::Zero;

// Re-export old implementations so the assembly benchmark can still access
// them via `crate::basic::newtonpf::{newton_pf_old, newton_pf_v0, …}`.
#[cfg(any(feature = "klu", feature = "klu_dyn"))]
pub(crate) use crate::basic::pf_old_impl::{
    JacobianCache, assemble_f, build_jacobian, build_jacobian_cached, newton_pf_old, newton_pf_v0,
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
    cache_opt: Option<&mut NewtonCache>,
) -> Result<(DVector<Complex64>, usize), (String, DVector<Complex64>, usize)> {
    let mut v = v_init.clone();
    let max_iter = max_iter.unwrap_or(100);
    let tol = tolerance.unwrap_or(1e-6);

    let n_state = npv + 2 * npq;
    let n_active = npv + npq;
    let n_bus = v.len();
    
    // We will use local variables for pattern and buffers if cache is not available.
    // If cache is available, we will mutably borrow them from the cache.
    // To satisfy borrow checker cleanly without cloning, we use an enum or mutable references.
    let mut local_j_pattern = None;
    let mut local_j_values = Vec::new(); 
    let mut local_ibus = DVector::zeros(0);
    let mut local_F = DVector::zeros(0);
    let mut local_s_calc = DVector::zeros(0);

    let (j_pattern, j_values, ibus, F, s_calc) = if let Some(c) = cache_opt {
        if c.j_pattern.is_none() {
            c.j_pattern = Some(JacobianPattern2::build_from_permuted(Ybus.col_offsets(), Ybus.row_indices(), npv, npq));
            c.j_values = vec![0.0; c.j_pattern.as_ref().unwrap().nnz_j];
            c.ibus = DVector::zeros(n_bus);
            c.F = DVector::zeros(n_state);
            c.s_calc = DVector::zeros(n_bus);
            c.npv = npv;
            c.npq = npq;
        }
        (
            c.j_pattern.as_ref().unwrap(),
            &mut c.j_values,
            &mut c.ibus,
            &mut c.F,
            &mut c.s_calc,
        )
    } else {
        local_j_pattern = Some(JacobianPattern2::build_from_permuted(Ybus.col_offsets(), Ybus.row_indices(), npv, npq));
        local_j_values = vec![0.0; local_j_pattern.as_ref().unwrap().nnz_j];
        local_ibus = DVector::zeros(n_bus);
        local_F = DVector::zeros(n_state);
        local_s_calc = DVector::zeros(n_bus);
        (
            local_j_pattern.as_ref().unwrap(),
            &mut local_j_values,
            &mut local_ibus,
            &mut local_F,
            &mut local_s_calc,
        )
    };
    csc_matvec_and_scalc(
        Ybus.col_offsets(),
        Ybus.row_indices(),
        Ybus.values(),
        v.as_slice(),
        ibus.as_mut_slice(),
        s_calc.as_mut_slice(),
    );

    let norm = fill_f_from_scalc::<false>(
        s_calc.as_slice(),
        Sbus.as_slice(),
        npq,
        n_active,
        F.as_mut_slice(),
    );

    if norm < tol {
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
        fill_jacobian_v3(
            Ybus,
            v.as_slice(),
            v_norm.as_slice(),
            s_calc.as_slice(),
            j_pattern,
            npv,
            npq,
            j_values,
        );

        if let Err(err) = solver.solve(
            Ap,
            Ai,
            j_values.as_mut_slice(),
            F.data.as_mut_slice(),
            n_state,
        ) {
            return Err((format!("Linear solve failed: {err}"), v, it));
        }

        let dx = &F;

        // Angle update: all non-slack buses.
        v_a.rows_range_mut(0..n_active)
            .zip_apply(&dx.rows_range(0..n_active), |a, b| {
                *a -= b;
                // *a = a;
            });
        // Magnitude update: PQ buses only (at 0..npq in PQ-first ordering).
        let mut vm_pq = v_m.rows_range_mut(0..npq);
        vm_pq.zip_apply(&dx.rows_range(n_active..n_state), |a, b| *a -= b);

        v_norm.zip_apply(&v_a, |a, va| *a = Complex64::from_polar(1.0, va));
        v.zip_zip_apply(&v_norm, &v_m, |a, e, vm| *a = vm * e);

        csc_matvec_and_scalc(
            Ybus.col_offsets(),
            Ybus.row_indices(),
            Ybus.values(),
            v.as_slice(),
            ibus.as_mut_slice(),
            s_calc.as_mut_slice(),
        );

        let norm2 = fill_f_from_scalc::<false>(
            s_calc.as_slice(),
            Sbus.as_slice(),
            npq,
            n_active,
            F.as_mut_slice(),
        );

        if norm2 < tol {
            return Ok((v, it + 1));
        }

        if F.norm() < tol {
            return Ok((v, it + 1));
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
#[inline(always)]
pub(crate) fn fill_f_from_scalc<const SPEC_MINUS_CALC: bool>(
    scalc: &[Complex64],
    sbus: &[Complex64],
    npq: usize,
    n_active: usize,
    f: &mut [f64],
) -> f64 {
    let mut max_norm: f64 = 0.0;

    // PQ: P and Q
    for i in 0..npq {
        let mis = if SPEC_MINUS_CALC {
            sbus[i] - scalc[i]
        } else {
            scalc[i] - sbus[i]
        };

        f[i] = mis.re;
        f[n_active + i] = mis.im;

        max_norm = max_norm.max(mis.re.abs()).max(mis.im.abs());
    }

    // PV: P only
    for i in npq..n_active {
        let mis = if SPEC_MINUS_CALC {
            sbus[i] - scalc[i]
        } else {
            scalc[i] - sbus[i]
        };

        f[i] = mis.re;

        max_norm = max_norm.max(mis.re.abs());
    }

    max_norm
}

#[inline(always)]
pub(crate) fn csc_matvec_complex(
    col_ptrs: &[usize],
    row_idx: &[usize],
    y_vals: &[Complex64],
    v: &[Complex64],
    ibus: &mut [Complex64],
) {
    ibus.fill(Complex64::new(0.0, 0.0));

    for k in 0..v.len() {
        let vk = v[k];

        for p in col_ptrs[k]..col_ptrs[k + 1] {
            let i = row_idx[p];
            ibus[i] += y_vals[p] * vk;
        }
    }
}

#[inline(always)]
pub(crate) fn csc_matvec_and_scalc(
    col_ptrs: &[usize],
    row_idx: &[usize],
    y_vals: &[Complex64],
    v: &[Complex64],
    ibus: &mut [Complex64],
    scalc: &mut [Complex64],
) {
    csc_matvec_complex(col_ptrs, row_idx, y_vals, v, ibus);

    for i in 0..v.len() {
        scalc[i] = v[i] * ibus[i].conj();
    }
}
