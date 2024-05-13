use std::f64::consts::PI;

use super::{dsbus_dv::dSbus_dV, solver::Solve, sparse::slice::*};
use crate::basic::sparse::{
    conj::RealImage,
    stack::{csc_hstack, csc_vstack},
};
use num_traits::Zero;

use nalgebra::*;
use nalgebra_sparse::*;
use num_complex::Complex64;

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
    let mut v_norm = v.map(|e| e.simd_signum());
    let max_iter = max_iter.unwrap_or(100);
    let tol = tolerance.unwrap_or(1e-6);

    let mut mis = &v.component_mul(&(Ybus * &v).conjugate()) - Sbus;

    let n_ext = v.len() - npv - npq;

    let n_bus = npq + npv;
    let num_state = npv + 2 * npq;

    let mut F = DVector::zeros(num_state);
    assemble_f(&mut F, n_bus, &mis, num_state, npv);

    let mut v_m = v.map(|e| e.simd_modulus());
    let mut v_a = v.map(|e| e.simd_argument());

    for iterations in 0..max_iter {
        // let power_mismatch = calc_power_mismatch(Ybus, S_load, &v);

        let (dS_dVm, dS_dVa) = dSbus_dV(Ybus, &v, &v_norm); // Assume Vnorm is just the norm of V here
        let jacobian = build_jacobian(&dS_dVm, &dS_dVa, npv, n_ext); // Need to implement this function

        let n = jacobian.nrows();

        let (mut Ap, mut Ai, mut Ax) = jacobian.disassemble();
        let _err = unsafe {
            solver
                .solve(
                    Ap.as_mut_slice(),
                    Ai.as_mut_slice(),
                    Ax.as_mut_slice(),
                    F.data.as_mut_slice_unchecked(),
                    n,
                )
                .unwrap();
        };

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

// fn slice_csc_matrix_block<T: Clone>(
//     mat: &CscMatrix<T>,
//     start_col: usize,
//     end_col: usize,
//     start_row: usize,
//     end_row: usize
// ) -> CscMatrix<T> {
//     let mut new_values = Vec::new();
//     let mut new_row_indices = Vec::new();
//     let mut new_col_offsets = Vec::new();
//     let mut current_offset = 0;

//     new_col_offsets.push(current_offset);

//     for col in start_col..=end_col {
//         let col_start_idx = mat.col_offsets()[col];
//         let col_end_idx = mat.col_offsets()[col + 1];

//         for idx in col_start_idx..col_end_idx {
//             let row_idx = mat.row_indices()[idx];
//             if row_idx >= start_row && row_idx <= end_row {
//                 new_row_indices.push(row_idx - start_row); // Adjust row indices
//                 new_values.push(mat.values()[idx].clone());
//                 current_offset += 1;
//             }
//         }
//         new_col_offsets.push(current_offset);
//     }

//     CscMatrix::try_from_csc_data(
//         end_row - start_row + 1,
//         end_col - start_col + 1,
//         new_col_offsets,
//         new_row_indices,
//         new_values
//     ).unwrap()
// }

// fn slice_csc_rows<T: Clone>(
//     mat: &CscMatrix<T>,
//     start_row: usize,
//     end_row: usize
// ) -> CscMatrix<T> {
//     let nrows = end_row - start_row + 1;
//     let ncols = mat.ncols();
//     let mut new_values = Vec::new();
//     let mut new_row_indices = Vec::new();
//     let mut new_col_offsets = vec![0; ncols + 1];

//     // 遍历每一列
//     for col in 0..ncols {
//         let col_start_idx = mat.col_offsets()[col];
//         let col_end_idx = mat.col_offsets()[col + 1];

//         // 遍历当前列中的每个元素
//         for idx in col_start_idx..col_end_idx {
//             let row_idx = mat.row_indices()[idx];
//             // 检查行索引是否在给定范围内
//             if row_idx >= start_row && row_idx <= end_row {
//                 new_row_indices.push(row_idx - start_row); // 调整行索引
//                 new_values.push(mat.values()[idx].clone());
//             }
//         }
//         new_col_offsets[col + 1] = new_values.len(); // 更新列偏移
//     }

//     CscMatrix::try_from_csc_data(
//         nrows,
//         ncols,
//         new_col_offsets,
//         new_row_indices,
//         new_values
//     ).unwrap()
// }

trait Slice {
    type Mat;
    fn block(&self, start_pos: (usize, usize), shape: (usize, usize)) -> Self::Mat;
    fn columns(&self, start_col: usize, end_col: usize) -> Self::Mat;
}
impl<T: Clone + Zero + Scalar + ClosedAdd> Slice for CscMatrix<T> {
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

#[allow(non_snake_case)]
#[inline(always)]
fn build_jacobian(
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
