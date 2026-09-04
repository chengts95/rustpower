use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use nalgebra::DVector;

use crate::basic::dcpf::{newton_pf_dcpf_serial, DcpfModel};
use crate::basic::ecs::elements::*;
use crate::basic::ecs::network::{PowerFlowSolver, SolverStage};
use crate::basic::ecs::plugin::{CustomSolverActive, DefaultSolverSet, PowerFlowSolverSet};
use crate::basic::ecs::powerflow::systems::{PowerFlowConfig, PowerFlowMat, PowerFlowResult};
use crate::basic::newtonpf::NewtonCache;
use crate::basic::solver::DefaultSolver;

/// Marker resource to flag that the DCPF-initialized Newton solver is active.
#[derive(Resource, Default)]
pub struct DcpfSolverActive;

/// Pre-allocated workspace buffer for DCPF in-place solve to avoid dynamic allocation.
#[derive(Resource, Default)]
pub struct DcpfWorkspace {
    pub buffer: Vec<f64>,
    pub solver: DefaultSolver,
}

/// Builds the DCPF susceptance matrix B_aa and phase-shifter / Slack boundary injections
/// directly from the existing `PowerFlowMat::y_bus` in O(nnz) time without rebuilding COO.
pub fn build_dcpf_model(
    common: Res<PFCommonData>,
    mat: Res<PowerFlowMat>,
    trafos: Query<(&Port4MatPatch, &TransformerDevice, &FromBus, &ToBus), Without<OutOfService>>,
) -> DcpfModel {
    let nodes = mat.v_bus_init.len();
    let n_active = mat.npv + mat.npq;
    let y_bus = &mat.y_bus;

    // 1. Directly construct B_aa in CSC format from Ybus left-upper block
    let mut col_ptrs = Vec::with_capacity(n_active + 1);
    col_ptrs.push(0);
    let mut row_indices = Vec::with_capacity(y_bus.row_indices().len());
    let mut values = Vec::with_capacity(y_bus.values().len());

    let mut p_slack_inj = vec![0.0; n_active];

    for col in 0..n_active {
        let start = y_bus.col_offsets()[col];
        let end = y_bus.col_offsets()[col + 1];

        let mut diag_sum = 0.0;
        let mut diag_idx = None;

        for idx in start..end {
            let row = y_bus.row_indices()[idx];
            if row == col {
                // Diagonal placeholder: row indices in Ybus column are already sorted,
                // so keeping `col` at its original slot maintains sorted CSC row indices.
                diag_idx = Some(values.len());
                row_indices.push(col);
                values.push(0.0);
                continue;
            }

            let b = y_bus.values()[idx].norm();
            diag_sum += b;

            if row < n_active {
                // Active neighbor in B_aa
                row_indices.push(row);
                values.push(-b);
            } else {
                // Slack neighbor: Dirichlet boundary injection (b * theta_slack)
                let theta_slack = mat.v_bus_init[row].arg();
                p_slack_inj[col] += b * theta_slack;
            }
        }

        if let Some(d_idx) = diag_idx {
            values[d_idx] = diag_sum;
        } else {
            row_indices.push(col);
            values.push(diag_sum);
        }

        col_ptrs.push(row_indices.len());
    }

    // 2. Transformer Phase Shifters (only scan non-zero shift degree trafos)
    let s_base = common.sbase;
    let mut p_shift_orig = vec![0.0; nodes];
    for (patch, dev, from, to) in trafos.iter() {
        let shift_deg = dev.effective_shift_degree();
        if shift_deg.abs() > 1e-6 {
            let vn = dev.vn_lv_kv;
            let b = patch.0[(0, 1)].norm() * (vn * vn) / s_base;
            let p_inj = b * (-shift_deg.to_radians());
            let f = from.0;
            let t = to.0;
            if f >= 0 && (f as usize) < nodes {
                p_shift_orig[f as usize] += p_inj;
            }
            if t >= 0 && (t as usize) < nodes {
                p_shift_orig[t as usize] -= p_inj;
            }
        }
    }

    // 3. Assemble active RHS shift vector: P_shift_active = P_shift - P_slack_inj
    let mut p_shift_active = DVector::zeros(n_active);
    for orig in 0..nodes {
        let perm = mat.to_perm[orig];
        if perm < n_active {
            p_shift_active[perm] = p_shift_orig[orig] - p_slack_inj[perm];
        }
    }

    DcpfModel {
        col_ptrs,
        row_indices,
        values,
        p_shift: p_shift_active,
        n_active,
    }
}

/// ECS system that executes the serial DCPF-initialized AC power flow.
pub fn ecs_run_dcpf_pf(
    mut cmd: Commands,
    mat: Res<PowerFlowMat>,
    cfg: Res<PowerFlowConfig>,
    mut dcpf_model: ResMut<DcpfModel>,
    mut workspace: ResMut<DcpfWorkspace>,
    mut solver_res: ResMut<PowerFlowSolver>,
    mut cache: Option<ResMut<NewtonCache>>,
) {
    if mat.npv + mat.npq >= mat.v_bus_init.len() {
        cmd.insert_resource(PowerFlowResult {
            v: mat.v_bus_init.clone_owned(),
            iterations: 0,
            converged: false,
        });
        return;
    }

    let n_active = dcpf_model.n_active;
    if workspace.buffer.len() < n_active {
        workspace.buffer.resize(n_active, 0.0);
    }

    let v_init = &mat.v_bus_init;
    let max_it = cfg.max_it;
    let tol = cfg.tol;

    let DcpfWorkspace {
        buffer,
        solver,
    } = &mut *workspace;

    let v = newton_pf_dcpf_serial(
        &mat.y_bus,
        &mat.s_bus,
        v_init,
        &mut dcpf_model,
        buffer.as_mut_slice(),
        mat.npv,
        mat.npq,
        tol,
        max_it,
        &mut solver_res.solver,
        solver,
        cache.as_deref_mut(),
    );

    let n = mat.v_bus_init.len();
    match v {
        Ok((v_perm, iterations)) => {
            let mut v_orig = nalgebra::DVector::from_element(n, num_complex::Complex64::new(0.0, 0.0));
            for (new_idx, &orig_idx) in mat.from_perm.iter().enumerate() {
                v_orig[orig_idx] = v_perm[new_idx];
            }
            cmd.insert_resource(PowerFlowResult {
                v: v_orig,
                iterations,
                converged: true,
            });
        }
        Err((_err, v_perm_err, its)) => {
            let mut v_orig = nalgebra::DVector::from_element(n, num_complex::Complex64::new(0.0, 0.0));
            for (new_idx, &orig_idx) in mat.from_perm.iter().enumerate() {
                v_orig[orig_idx] = v_perm_err[new_idx];
            }
            cmd.insert_resource(PowerFlowResult {
                v: v_orig,
                iterations: its,
                converged: false,
            });
        }
    }
}

/// Alternative solver plugin that activates serial DCPF initialization.
#[derive(Default)]
pub struct DcpfNewtonPfPlugin;

impl Plugin for DcpfNewtonPfPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<DcpfWorkspace>() {
            app.world_mut().insert_resource(DcpfWorkspace::default());
        }
        app.configure_sets(
            Update,
            DefaultSolverSet.run_if(
                not(resource_exists::<CustomSolverActive>)
                    .and_then(not(resource_exists::<DcpfSolverActive>)),
            ),
        );
        app.add_systems(
            Update,
            ecs_run_dcpf_pf
                .in_set(SolverStage::Solve)
                .in_set(PowerFlowSolverSet)
                .run_if(resource_exists::<DcpfSolverActive>),
        );
    }
}
