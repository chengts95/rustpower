use std::f64::consts::PI;

use super::dsbus_dv::{dSbus_dV, dSbus_dV_old};
use super::sparse::slice::*;
use super::solver::Solve;
use super::new_dsdvbus2::{JacobianPattern2, fill_jacobian_v2};
use crate::basic::sparse::{
    conj::RealImage,
    stack::{csc_hstack, csc_vstack},
};
use nalgebra::*;
use nalgebra_sparse::*;
use num_complex::Complex64;
use num_traits::Zero;

/// Old Newton-Raphson (kept for reference / fallback).
/// Uses the cached dS_dVm/dS_dVa CSC build & slice path.
#[allow(non_snake_case, dead_code)]
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
    assemble_f(&mut F, n_bus, &mis, num_state, npv);
    if F.norm() < tol {
        return Ok((v, 0));
    }
    let mut v_m = v.map(|e| e.simd_modulus());
    let mut v_a = v.map(|e| e.simd_argument());
    let mut cache: Option<JacobianCache> = None;

    for iterations in 0..max_iter {
        let (dS_dVm, dS_dVa) = dSbus_dV(Ybus, &v, &v_norm);
        let jacobian = build_jacobian_cached(&dS_dVm, &dS_dVa, &mut cache, npv, n_ext);

        let n = jacobian.nrows();
        let (mut Ap, mut Ai, mut Ax) = jacobian.disassemble();

        let _err = solver
            .solve(
                Ap.as_mut_slice(),
                Ai.as_mut_slice(),
                Ax.as_mut_slice(),
                F.data.as_mut_slice(),
                n,
            )
            .unwrap();

        let dx = &F;
        update_v(
            &mut v_a,
            dx,
            n_bus,
            &mut v_m,
            npv,
            num_state,
            &mut v_norm,
            &mut v,
        );

        v.component_mul(&(Ybus * &v).conjugate())
            .sub_to(Sbus, &mut mis);

        assemble_f(&mut F, n_bus, &mis, num_state, npv);

        if F.norm() < tol {
            return Ok((v, iterations));
        }
    }

    Err((String::from("Did not converge!"), v))
}

/// V0 -- the literal MATPOWER port, kept only as the un-optimised baseline for
/// the assembly benchmark. Each iteration recomputes `dSbus_dV_old` (diagonal
/// matrices materialised as identity-pattern CSC, then diagonal-matrix SpGEMM
/// `Ybus * diagV`) and rebuilds the Jacobian with the *uncached* `build_jacobian`
/// (fresh `csc_vstack`/`csc_hstack`, structure regenerated every iteration).
#[allow(non_snake_case, dead_code)]
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
    assemble_f(&mut F, n_bus, &mis, num_state, npv);
    if F.norm() < tol {
        return Ok((v, 0));
    }
    let mut v_m = v.map(|e| e.simd_modulus());
    let mut v_a = v.map(|e| e.simd_argument());

    for iterations in 0..max_iter {
        let (dS_dVm, dS_dVa) = dSbus_dV_old(Ybus, &v, &v_norm);
        let jacobian = build_jacobian(&dS_dVm, &dS_dVa, npv, n_ext);

        let n = jacobian.nrows();
        let (mut Ap, mut Ai, mut Ax) = jacobian.disassemble();

        let _err = solver
            .solve(
                Ap.as_mut_slice(),
                Ai.as_mut_slice(),
                Ax.as_mut_slice(),
                F.data.as_mut_slice(),
                n,
            )
            .unwrap();

        let dx = &F;
        update_v(
            &mut v_a,
            dx,
            n_bus,
            &mut v_m,
            npv,
            num_state,
            &mut v_norm,
            &mut v,
        );

        v.component_mul(&(Ybus * &v).conjugate())
            .sub_to(Sbus, &mut mis);

        assemble_f(&mut F, n_bus, &mis, num_state, npv);

        if F.norm() < tol {
            return Ok((v, iterations));
        }
    }

    Err((String::from("Did not converge!"), v))
}

/// Newton--Raphson power flow under the `[PQ | PV | slack]` bus ordering.
/// Branch-free Jacobian assembly via `JacobianPattern2` + `fill_jacobian_v2`.
///
/// Requires `Ybus`, `Sbus`, `v_init` already permuted into `[PQ | PV | slack]`:
/// PQ buses at indices `0..npq`, PV at `npq..npq+npv`, slack at `npq+npv..`.
#[allow(non_snake_case)]
pub fn newton_pf<Solver: Solve>(
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
    let max_iter = max_iter.unwrap_or(100);
    let tol = tolerance.unwrap_or(1e-6);

    let j_pattern = JacobianPattern2::build_from_permuted(
        Ybus.col_offsets(), Ybus.row_indices(), npv, npq,
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

        fill_jacobian_v2(
            Ybus,
            v.as_slice(),
            v_norm.as_slice(),
            ibus.as_slice(),
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
        // Magnitude update: only PQ buses (V2: PQ at 0..npq).
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

    Err((String::from("Did not converge!"), v))
}

/// V2 mismatch right-hand side under `[PQ | PV | slack]` ordering.
/// `F[0..n_bus] = Re(mis[0..n_bus])`; `F[n_bus..n_state] = Im(mis[0..npq])`.
#[inline(always)]
fn assemble_f_v2(
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

/// Assembles the mismatch vector.
///
/// # Parameters
///
/// * `f` - The mismatch vector to be assembled.
/// * `n_bus` - The number of buses.
/// * `mis` - The current power mismatches.
/// * `num_state` - The number of states.
/// * `npv` - The number of PV buses.
#[inline(always)]
fn assemble_f(
    f: &mut DVector<f64>,
    n_bus: usize,
    mis: &DVector<Complex64>,
    num_state: usize,
    npv: usize,
) {
    f.rows_range_mut(0..n_bus)
        .zip_apply(&mis.rows_range(0..n_bus), |a, b| *a = b.simd_real());
    f.rows_range_mut(n_bus..num_state)
        .zip_apply(&(mis.rows_range(npv..n_bus)), |a, b| {
            *a = b.simd_imaginary()
        });
}

/// Updates the voltage vector.
///
/// # Parameters
///
/// * `v_a` - The voltage angle vector.
/// * `dx` - The state update vector.
/// * `n_bus` - The number of buses.
/// * `v_m` - The voltage magnitude vector.
/// * `npv` - The number of PV buses.
/// * `num_state` - The number of states.
/// * `v_norm` - The normalized voltage vector.
/// * `v` - The voltage vector to be updated.
#[inline(always)]
fn update_v(
    v_a: &mut DVector<f64>,
    dx: &DVector<f64>,
    n_bus: usize,
    v_m: &mut DVector<f64>,
    npv: usize,
    num_state: usize,
    v_norm: &mut DVector<Complex64>,
    v: &mut DVector<Complex64>,
) {
    v_a.rows_range_mut(0..n_bus)
        .zip_apply(&dx.rows_range(0..n_bus), |a, b| {
            (*a) -= b;
            *a = a.rem_euclid(2.0 * PI);
        });
    let mut vm_pq = v_m.rows_range_mut(npv..n_bus);
    vm_pq.zip_apply(&dx.rows_range(n_bus..num_state), |a, b| (*a) -= b);

    v_norm.zip_apply(&*v_a, |a, va| *a = Complex64::from_polar(1.0, va));
    v.zip_zip_apply(v_norm, v_m, |a, e, vm| *a = vm * e);
}

/// Trait for slicing a CSC matrix.
pub trait Slice {
    type Mat;

    /// Slices a block from the CSC matrix.
    fn block(&self, start_pos: (usize, usize), shape: (usize, usize)) -> Self::Mat;

    /// Slices columns from the CSC matrix.
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

/// Trait for slicing a CSC matrix into a destination matrix.
trait SliceTo {
    type Mat;

    /// Slices a block from the CSC matrix into a destination matrix.
    fn block_to(&self, start_pos: (usize, usize), shape: (usize, usize), mat: &mut Self::Mat);

    /// Slices columns from the CSC matrix into a destination matrix.
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

/// Builds the Jacobian matrix.
///
/// # Parameters
///
/// * `ds_dvm` - The partial derivatives of the power injections with respect to voltage magnitudes.
/// * `ds_dva` - The partial derivatives of the power injections with respect to voltage angles.
/// * `npv` - The number of PV buses.
/// * `n_ext` - The number of external elements.
///
/// # Returns
///
/// The Jacobian matrix.
#[allow(non_snake_case)]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn build_jacobian(
    ds_dvm: &CscMatrix<Complex64>,
    ds_dva: &CscMatrix<Complex64>,
    npv: usize,
    n_ext: usize,
) -> CscMatrix<f64> {
    let (real, imag) = ds_dva
        .block((0, 0), (ds_dva.nrows() - n_ext, ds_dva.ncols() - n_ext))
        .real_imag();
    let (real2, imag2) = ds_dvm
        .block((0, 0), (ds_dvm.nrows() - n_ext, ds_dvm.ncols() - n_ext))
        .real_imag();
    let J11 = real;
    let J12 = real2.columns(npv, real2.ncols());
    let J21 = imag.block((npv, 0), (imag.nrows() - npv, imag.ncols()));
    let J22 = imag2.block((npv, npv), (imag2.nrows() - npv, imag2.ncols() - npv));

    let J = csc_vstack(&[&csc_hstack(&[&J11, &J12]), &csc_hstack(&[&J21, &J22])]);
    J
}

/// Builds the Jacobian matrix using a cache.
///
/// # Parameters
///
/// * `ds_dvm` - The partial derivatives of the power injections with respect to voltage magnitudes.
/// * `ds_dva` - The partial derivatives of the power injections with respect to voltage angles.
/// * `cache` - The cache for the Jacobian matrix.
/// * `npv` - The number of PV buses.
/// * `n_ext` - The number of external elements.
///
/// # Returns
///
/// The Jacobian matrix.
#[allow(non_snake_case)]
#[inline(always)]
pub(crate) fn build_jacobian_cached(
    ds_dvm: &CscMatrix<Complex64>,
    ds_dva: &CscMatrix<Complex64>,
    cache: &mut Option<JacobianCache>,
    npv: usize,
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
            real2.columns_to(npv, real2.ncols(), &mut cache.j12);
            imag.block_to((npv, 0), (imag.nrows() - npv, imag.ncols()), &mut cache.j21);
            imag2.block_to(
                (npv, npv),
                (imag2.nrows() - npv, imag2.ncols() - npv),
                &mut cache.j22,
            );

            let J = csc_vstack(&[
                &csc_hstack(&[&cache.j11, &cache.j12]),
                &csc_hstack(&[&cache.j21, &cache.j22]),
            ]);
            J
        }
        None => {
            let ds_dva = ds_dva.block((0, 0), (ds_dva.nrows() - n_ext, ds_dva.ncols() - n_ext));
            let ds_dvm = ds_dvm.block((0, 0), (ds_dvm.nrows() - n_ext, ds_dvm.ncols() - n_ext));
            let (real, imag) = ds_dva.real_imag();
            let (real2, imag2) = ds_dvm.real_imag();
            let j11 = real;
            let j12 = real2.columns(npv, real2.ncols());
            let j21 = imag.block((npv, 0), (imag.nrows() - npv, imag.ncols()));
            let j22 = imag2.block((npv, npv), (imag2.nrows() - npv, imag2.ncols() - npv));
            let J = csc_vstack(&[&csc_hstack(&[&j11, &j12]), &csc_hstack(&[&j21, &j22])]);
            let icache = JacobianCache {
                ds_dva,
                ds_dvm,
                j11,
                j12,
                j21,
                j22,
            };
            cache.replace(icache);

            J
        }
    }
}

/// A cache for the Jacobian matrix components.
pub(crate) struct JacobianCache {
    ds_dva: CscMatrix<Complex64>,
    ds_dvm: CscMatrix<Complex64>,
    j11: CscMatrix<f64>,
    j12: CscMatrix<f64>,
    j21: CscMatrix<f64>,
    j22: CscMatrix<f64>,
}
