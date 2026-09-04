use bevy_ecs::prelude::Resource;
use nalgebra::DVector;
use num_complex::Complex64;

use super::newtonpf::{newton_pf, NewtonCache};
use super::solver::Solve;
use nalgebra_sparse::CscMatrix;

/// Pre-assembled DCPF model in `[PQ | PV | Slack]` permuted space.
/// Holds raw CSC slices ready for in-place linear solve without any data copying.
#[derive(Resource, Clone, Debug)]
pub struct DcpfModel {
    pub col_ptrs: Vec<usize>,
    pub row_indices: Vec<usize>,
    pub values: Vec<f64>,
    pub p_shift: DVector<f64>,
    pub n_active: usize,
}

/// Solves the DC power flow linear system in-place into `theta_buf`:
/// B_active * theta_active = P_inj - P_shift
///
/// ZERO heap allocation: uses caller-provided workspace buffer `theta_buf`.
pub fn dcpf_solve<Solver: Solve>(
    col_ptrs: &mut [usize],
    row_indices: &mut [usize],
    values: &mut [f64],
    p_inj: &[f64],
    p_shift: &[f64],
    theta_buf: &mut [f64],
    solver: &mut Solver,
) -> Result<(), &'static str> {
    let n = theta_buf.len();
    for i in 0..n {
        theta_buf[i] = p_inj[i] - p_shift[i];
    }
    solver.solve(col_ptrs, row_indices, values, theta_buf, n)
}

/// Computes the DCPF initial voltage vector (in permuted space) using caller-provided workspace buffer.
/// V_i = |V_init_i| * exp(j * theta_i) for active buses (0..n_active).
/// Slack buses retain their original complex voltage from v_init.
pub fn dcpf_initial_v<Solver: Solve>(
    model: &mut DcpfModel,
    s_bus: &DVector<Complex64>,
    v_init: &DVector<Complex64>,
    theta_workspace: &mut [f64],
    solver: &mut Solver,
) -> Result<DVector<Complex64>, &'static str> {
    let n_active = model.n_active;
    assert!(
        theta_workspace.len() >= n_active,
        "theta_workspace buffer too small"
    );

    for i in 0..n_active {
        theta_workspace[i] = s_bus[i].re - model.p_shift[i];
    }

    solver.solve(
        &mut model.col_ptrs,
        &mut model.row_indices,
        &mut model.values,
        &mut theta_workspace[..n_active],
        n_active,
    )?;

    let mut v = v_init.clone();
    for i in 0..n_active {
        let mag = v_init[i].norm();
        let angle = theta_workspace[i];
        v[i] = Complex64::from_polar(mag, angle);
    }
    Ok(v)
}

/// Serial Newton-Raphson ACPF initialized by in-place DCPF.
/// Reuses standard `newton_pf` kernel without code duplication or extra allocations.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn newton_pf_dcpf_serial<Solver: Solve>(
    Ybus: &CscMatrix<Complex64>,
    Sbus: &DVector<Complex64>,
    v_init: &DVector<Complex64>,
    dcpf_model: &mut DcpfModel,
    dcpf_workspace: &mut [f64],
    npv: usize,
    npq: usize,
    tolerance: Option<f64>,
    max_iter: Option<usize>,
    solver: &mut Solver,
    dcpf_solver: &mut Solver,
    cache_opt: Option<&mut NewtonCache>,
) -> Result<(DVector<Complex64>, usize), (String, DVector<Complex64>, usize)> {
    let v_dcpf = dcpf_initial_v(dcpf_model, Sbus, v_init, dcpf_workspace, dcpf_solver)
        .map_err(|e| (e.to_string(), v_init.clone(), 0))?;

    newton_pf(
        Ybus,
        Sbus,
        &v_dcpf,
        npv,
        npq,
        tolerance,
        max_iter,
        solver,
        cache_opt,
    )
}
