//! V0 and V1 Newton-PF implementations (kept for the assembly benchmark).
//!
//! Both functions now expect the **`[PQ | PV | slack]`** bus ordering produced
//! by the ECS (same ordering as the current default `newton_pf`).  The previous
//! `npv` offset used inside `build_jacobian` / `update_v` has been replaced by
//! `npq` so the PQ block is addressed at its true position (index 0) in the
//! PQ-first permuted system.

use std::f64::consts::PI;

use crate::basic::dsbus_dv::{dSbus_dV, dSbus_dV_old};
use crate::basic::newtonpf::Slice;
use crate::basic::solver::Solve;
use crate::basic::sparse::{
    conj::RealImage,
    slice::*,
    stack::{csc_hstack, csc_vstack},
};
use nalgebra::*;
use nalgebra_sparse::*;
use num_complex::Complex64;
use num_traits::Zero;

// ─── internal mismatch assembly ──────────────────────────────────────────────

/// Assemble the NR right-hand side for `[PQ | PV | slack]` ordering.
///
/// `F[0..n_bus]`      = Re(mis[0..n_bus])   (P equations, all non-slack)
/// `F[n_bus..n_state]` = Im(mis[0..npq])    (Q equations, PQ buses only)
#[inline(always)]
pub(crate) fn assemble_f(
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

// ─── internal voltage update ─────────────────────────────────────────────────

/// Apply NR step `dx` to voltage vectors under `[PQ | PV | slack]` ordering.
///
/// `npq`: number of PQ buses (Vm update range `0..npq`, angles `0..n_bus`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn update_v(
    v_a: &mut DVector<f64>,
    dx: &DVector<f64>,
    n_bus: usize,
    v_m: &mut DVector<f64>,
    npq: usize,
    num_state: usize,
    v_norm: &mut DVector<Complex64>,
    v: &mut DVector<Complex64>,
) {
    v_a.rows_range_mut(0..n_bus)
        .zip_apply(&dx.rows_range(0..n_bus), |a, b| {
            *a -= b;
            *a = a.rem_euclid(2.0 * PI);
        });
    // PQ buses are at 0..npq in PQ-first ordering.
    let mut vm_pq = v_m.rows_range_mut(0..npq);
    vm_pq.zip_apply(&dx.rows_range(n_bus..num_state), |a, b| *a -= b);

    v_norm.zip_apply(&*v_a, |a, va| *a = Complex64::from_polar(1.0, va));
    v.zip_zip_apply(v_norm, v_m, |a, e, vm| *a = vm * e);
}

// ─── SliceTo (in-place block/column extraction) ───────────────────────────────

trait SliceTo {
    type Mat;
    fn block_to(&self, start_pos: (usize, usize), shape: (usize, usize), mat: &mut Self::Mat);
    fn columns_to(&self, start_col: usize, end_col: usize, mat: &mut Self::Mat);
}

impl<T: Copy + Clone + Zero + Scalar + ClosedAddAssign> SliceTo for CscMatrix<T> {
    type Mat = CscMatrix<T>;

    #[inline(always)]
    fn block_to(&self, start_pos: (usize, usize), shape: (usize, usize), mat: &mut Self::Mat) {
        slice_csc_matrix_block_to(self, start_pos, shape, mat)
    }

    #[inline(always)]
    fn columns_to(&self, start_col: usize, end_col: usize, mat: &mut Self::Mat) {
        slice_csc_matrix_to(self, start_col, end_col, mat)
    }
}

// ─── Jacobian builders ────────────────────────────────────────────────────────

/// Build the NR Jacobian from full `dS_dVm`/`dS_dVa` CSC matrices.
///
/// `npq`: number of PQ buses.  Under `[PQ | PV | slack]` ordering the PQ
/// block starts at index 0, so J12/J21/J22 are extracted from rows/cols
/// `0..npq` (not `npv..n_bus` as in the old PV-first convention).
#[allow(non_snake_case, dead_code)]
pub(crate) fn build_jacobian(
    ds_dvm: &CscMatrix<Complex64>,
    ds_dva: &CscMatrix<Complex64>,
    npq: usize,
    n_ext: usize,
) -> CscMatrix<f64> {
    let (real, imag) = ds_dva
        .block((0, 0), (ds_dva.nrows() - n_ext, ds_dva.ncols() - n_ext))
        .real_imag();
    let (real2, imag2) = ds_dvm
        .block((0, 0), (ds_dvm.nrows() - n_ext, ds_dvm.ncols() - n_ext))
        .real_imag();
    let j11 = real;
    let j12 = real2.columns(0, npq);
    let j21 = imag.block((0, 0), (npq, imag.ncols()));
    let j22 = imag2.block((0, 0), (npq, npq));
    csc_vstack(&[&csc_hstack(&[&j11, &j12]), &csc_hstack(&[&j21, &j22])])
}

/// Build the NR Jacobian with a reusable cache for the sparsity pattern.
///
/// Same PQ-first convention as `build_jacobian`.
#[allow(non_snake_case)]
pub(crate) fn build_jacobian_cached(
    ds_dvm: &CscMatrix<Complex64>,
    ds_dva: &CscMatrix<Complex64>,
    cache: &mut Option<JacobianCache>,
    npq: usize,
    n_ext: usize,
) -> CscMatrix<f64> {
    match cache {
        Some(cache) => {
            ds_dva.block_to(
                (0, 0),
                (ds_dva.nrows() - n_ext, ds_dva.ncols() - n_ext),
                &mut cache.ds_dva,
            );
            let (real, imag) = cache.ds_dva.real_imag();
            cache.j11 = real;
            ds_dvm.block_to(
                (0, 0),
                (ds_dvm.nrows() - n_ext, ds_dvm.ncols() - n_ext),
                &mut cache.ds_dvm,
            );
            let (real2, imag2) = cache.ds_dvm.real_imag();
            real2.columns_to(0, npq, &mut cache.j12);
            imag.block_to((0, 0), (npq, imag.ncols()), &mut cache.j21);
            imag2.block_to((0, 0), (npq, npq), &mut cache.j22);
            csc_vstack(&[
                &csc_hstack(&[&cache.j11, &cache.j12]),
                &csc_hstack(&[&cache.j21, &cache.j22]),
            ])
        }
        None => {
            let ds_dva =
                ds_dva.block((0, 0), (ds_dva.nrows() - n_ext, ds_dva.ncols() - n_ext));
            let ds_dvm =
                ds_dvm.block((0, 0), (ds_dvm.nrows() - n_ext, ds_dvm.ncols() - n_ext));
            let (real, imag) = ds_dva.real_imag();
            let (real2, imag2) = ds_dvm.real_imag();
            let j11 = real;
            let j12 = real2.columns(0, npq);
            let j21 = imag.block((0, 0), (npq, imag.ncols()));
            let j22 = imag2.block((0, 0), (npq, npq));
            let j = csc_vstack(&[&csc_hstack(&[&j11, &j12]), &csc_hstack(&[&j21, &j22])]);
            cache.replace(JacobianCache { ds_dva, ds_dvm, j11, j12, j21, j22 });
            j
        }
    }
}

/// Cached Jacobian block buffers (reused across NR iterations in V1).
pub(crate) struct JacobianCache {
    pub ds_dva: CscMatrix<Complex64>,
    pub ds_dvm: CscMatrix<Complex64>,
    pub j11: CscMatrix<f64>,
    pub j12: CscMatrix<f64>,
    pub j21: CscMatrix<f64>,
    pub j22: CscMatrix<f64>,
}

// ─── V1: newton_pf_old ────────────────────────────────────────────────────────

/// V1 Newton-PF: single-pass `dSbus_dV` + `build_jacobian_cached` under
/// `[PQ | PV | slack]` ordering.  Kept as the "semi-optimised" baseline for
/// the assembly benchmark.
#[allow(non_snake_case, dead_code, clippy::too_many_arguments)]
pub fn newton_pf_old<Solver: Solve>(
    Ybus: &CscMatrix<Complex64>,
    Sbus: &DVector<Complex64>,
    v_init: &DVector<Complex64>,
    npv: usize,
    npq: usize,
    tolerance: Option<f64>,
    max_iter: Option<usize>,
    solver: &mut Solver,
) -> Result<(DVector<Complex64>, usize), (String, DVector<Complex64>)> {
    let mut v = v_init.clone();
    let mut v_norm = v.map(|e| e.simd_signum());
    let max_iter = max_iter.unwrap_or(100);
    let tol = tolerance.unwrap_or(1e-6);

    let mut mis = &v.component_mul(&(Ybus * &v).conjugate()) - Sbus;

    let n_ext = v.len() - npv - npq;
    let n_bus = npq + npv;
    let num_state = npv + 2 * npq;

    let mut F = DVector::zeros(num_state);
    assemble_f(&mut F, n_bus, &mis, num_state, npq);
    if F.norm() < tol {
        return Ok((v, 0));
    }
    let mut v_m = v.map(|e| e.simd_modulus());
    let mut v_a = v.map(|e| e.simd_argument());
    let mut cache: Option<JacobianCache> = None;

    for iterations in 0..max_iter {
        let (dS_dVm, dS_dVa) = dSbus_dV(Ybus, &v, &v_norm);
        let jacobian = build_jacobian_cached(&dS_dVm, &dS_dVa, &mut cache, npq, n_ext);

        let n = jacobian.nrows();
        let (mut Ap, mut Ai, mut Ax) = jacobian.disassemble();
        let _ = solver.solve(
            Ap.as_mut_slice(),
            Ai.as_mut_slice(),
            Ax.as_mut_slice(),
            F.data.as_mut_slice(),
            n,
        );

        let dx = &F;
        update_v(&mut v_a, dx, n_bus, &mut v_m, npq, num_state, &mut v_norm, &mut v);

        v.component_mul(&(Ybus * &v).conjugate())
            .sub_to(Sbus, &mut mis);
        assemble_f(&mut F, n_bus, &mis, num_state, npq);

        if F.norm() < tol {
            return Ok((v, iterations));
        }
    }

    Err((String::from("Did not converge!"), v))
}

// ─── V0: newton_pf_v0 ────────────────────────────────────────────────────────

/// V0 Newton-PF: literal MATPOWER port — `dSbus_dV_old` (diagonal-matrix
/// SpGEMM) + uncached `build_jacobian`.  Un-optimised baseline for the
/// assembly benchmark.
#[allow(non_snake_case, dead_code, clippy::too_many_arguments)]
pub fn newton_pf_v0<Solver: Solve>(
    Ybus: &CscMatrix<Complex64>,
    Sbus: &DVector<Complex64>,
    v_init: &DVector<Complex64>,
    npv: usize,
    npq: usize,
    tolerance: Option<f64>,
    max_iter: Option<usize>,
    solver: &mut Solver,
) -> Result<(DVector<Complex64>, usize), (String, DVector<Complex64>)> {
    let mut v = v_init.clone();
    let mut v_norm = v.map(|e| e.simd_signum());
    let max_iter = max_iter.unwrap_or(100);
    let tol = tolerance.unwrap_or(1e-6);

    let mut mis = &v.component_mul(&(Ybus * &v).conjugate()) - Sbus;

    let n_ext = v.len() - npv - npq;
    let n_bus = npq + npv;
    let num_state = npv + 2 * npq;

    let mut F = DVector::zeros(num_state);
    assemble_f(&mut F, n_bus, &mis, num_state, npq);
    if F.norm() < tol {
        return Ok((v, 0));
    }
    let mut v_m = v.map(|e| e.simd_modulus());
    let mut v_a = v.map(|e| e.simd_argument());

    for iterations in 0..max_iter {
        let (dS_dVm, dS_dVa) = dSbus_dV_old(Ybus, &v, &v_norm);
        let jacobian = build_jacobian(&dS_dVm, &dS_dVa, npq, n_ext);

        let n = jacobian.nrows();
        let (mut Ap, mut Ai, mut Ax) = jacobian.disassemble();
        let _ = solver.solve(
            Ap.as_mut_slice(),
            Ai.as_mut_slice(),
            Ax.as_mut_slice(),
            F.data.as_mut_slice(),
            n,
        );

        let dx = &F;
        update_v(&mut v_a, dx, n_bus, &mut v_m, npq, num_state, &mut v_norm, &mut v);

        v.component_mul(&(Ybus * &v).conjugate())
            .sub_to(Sbus, &mut mis);
        assemble_f(&mut F, n_bus, &mis, num_state, npq);

        if F.norm() < tol {
            return Ok((v, iterations));
        }
    }

    Err((String::from("Did not converge!"), v))
}
