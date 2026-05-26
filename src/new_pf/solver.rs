use bevy_ecs::prelude::*;
use bevy_app::prelude::*;
use nalgebra::*;
use nalgebra_sparse::*;
use num_complex::Complex64;
use std::f64::consts::PI;

use crate::new_pf::systems::{NetworkOperators, BinaryIncidence, PFOrder};
use crate::basic::new_dsdvbus2::{JacobianPattern2};
use crate::basic::new_dsdvbus3::fill_jacobian_v3;
use crate::basic::solver::Solve;

/// Core Newton-Raphson Solver (Free Function)
///
/// This function is decoupled from Bevy systems for maximum performance and testability.
/// It assumes Ybus is already permuted.
pub fn run_newton_pf<S: Solve>(
    ybus: &CscMatrix<Complex64>,
    sbus: &DVector<Complex64>,
    v_init: &DVector<Complex64>,
    npv: usize,
    npq: usize,
    solver: &mut S,
    max_iter: usize,
    tol: f64,
) -> Result<(DVector<Complex64>, usize), String> {
    let mut v = v_init.clone();
    let n_bus = npv + npq;
    let n_state = npv + 2 * npq;
    
    let j_pattern = JacobianPattern2::build_from_permuted(
        ybus.col_offsets(), ybus.row_indices(), npv, npq,
    );
    let mut j_values = vec![0.0; j_pattern.nnz_j];
    let mut f_vec = DVector::zeros(n_state);
    
    let mut v_m = v.map(|e| e.norm());
    let mut v_a = v.map(|e| e.arg());
    let mut v_norm = v.map(|e| Complex64::from_polar(1.0, e.arg()));

    for it in 0..max_iter {
        let ibus = ybus * &v;
        let s_calc = v.component_mul(&ibus.map(|e| e.conj()));
        let mis = &s_calc - sbus;
        
        // Assemble mismatch vector F
        for i in 0..n_bus { f_vec[i] = mis[i].re; }
        for i in 0..npq { f_vec[n_bus + i] = mis[i].im; }

        if f_vec.norm() < tol {
            return Ok((v, it));
        }

        fill_jacobian_v3(
            ybus,
            v.as_slice(),
            v_norm.as_slice(),
            s_calc.as_slice(),
            &j_pattern,
            npv,
            npq,
            &mut j_values,
        );

        solver.solve(
            &mut j_pattern.j_col_ptrs.clone(), // This clone is bad for perf, but Solve trait requires mut
            &mut j_pattern.j_row_indices.clone(),
            &mut j_values,
            f_vec.data.as_mut_slice(),
            n_state,
        ).map_err(|e| e.to_string())?;

        let dx = &f_vec;
        
        // Update x
        for i in 0..n_bus {
            v_a[i] -= dx[i];
            v_a[i] = v_a[i].rem_euclid(2.0 * PI);
        }
        for i in 0..npq {
            v_m[i] -= dx[n_bus + i];
        }

        // Reconstruct V
        for i in 0..v.len() {
            v_norm[i] = Complex64::from_polar(1.0, v_a[i]);
            v[i] = v_m[i] * v_norm[i];
        }
    }

    Err("Newton-Raphson failed to converge".to_string())
}

/// Thin Bevy System Wrapper
pub fn newton_pf_system(
    mut ops: ResMut<NetworkOperators>,
    order: Res<PFOrder>,
    // TODO: Add queries for current voltage and setpoints
) {
    let Some(_ybus) = &ops.ybus else { return };
    // Integration with ECS components would go here.
}
