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

/// Pre-allocated workspace buffer and solver for DCPF in-place solve.
#[derive(Resource, Default)]
pub struct DcpfWorkspace {
    pub buffer: Vec<f64>,
    pub solver: DefaultSolver,
}

impl DcpfWorkspace {
    #[inline]
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.buffer.len() < n {
            self.buffer.resize(n, 0.0);
        }
    }
}

/// Builds the DCPF model by querying phase shifters and delegating matrix extraction to `DcpfModel::from_ybus`.
pub fn build_dcpf_model(
    common: Res<PFCommonData>,
    mat: Res<PowerFlowMat>,
    trafos: Query<(&Port4MatPatch, &TransformerDevice, &FromBus, &ToBus), Without<OutOfService>>,
) -> DcpfModel {
    let mut p_shift_orig = vec![0.0; mat.v_bus_init.len()];
    for (patch, dev, from, to) in trafos.iter() {
        let shift_deg = dev.effective_shift_degree();
        if shift_deg.abs() > 1e-6 {
            let vn = dev.vn_lv_kv;
            let b = patch.0[(0, 1)].norm() * (vn * vn) / common.sbase;
            let p_inj = b * (-shift_deg.to_radians());
            if from.0 >= 0 && (from.0 as usize) < p_shift_orig.len() {
                p_shift_orig[from.0 as usize] += p_inj;
            }
            if to.0 >= 0 && (to.0 as usize) < p_shift_orig.len() {
                p_shift_orig[to.0 as usize] -= p_inj;
            }
        }
    }

    DcpfModel::from_ybus(
        &mat.y_bus,
        &mat.v_bus_init,
        mat.npv + mat.npq,
        &mat.to_perm,
        &p_shift_orig,
    )
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
    let ws = &mut *workspace;
    ws.ensure_capacity(n_active);

    let res = newton_pf_dcpf_serial(
        &mat.y_bus,
        &mat.s_bus,
        &mat.v_bus_init,
        &mut dcpf_model,
        &mut ws.buffer[..n_active],
        mat.npv,
        mat.npq,
        cfg.tol,
        cfg.max_it,
        &mut solver_res.solver,
        &mut ws.solver,
        cache.as_deref_mut(),
    );

    let (v_perm, iterations, converged) = match res {
        Ok((v, iters)) => (v, iters, true),
        Err((_, v, iters)) => (v, iters, false),
    };

    let mut v_orig = DVector::from_element(mat.v_bus_init.len(), num_complex::Complex64::new(0.0, 0.0));
    for (new_idx, &orig_idx) in mat.from_perm.iter().enumerate() {
        v_orig[orig_idx] = v_perm[new_idx];
    }

    cmd.insert_resource(PowerFlowResult {
        v: v_orig,
        iterations,
        converged,
    });
}

/// Ensures DcpfModel resource exists before solve if DcpfSolverActive is set.
pub fn ensure_dcpf_model(
    mut cmd: Commands,
    model: Option<Res<DcpfModel>>,
    common: Res<PFCommonData>,
    mat: Res<PowerFlowMat>,
    trafos: Query<(&Port4MatPatch, &TransformerDevice, &FromBus, &ToBus), Without<OutOfService>>,
) {
    if model.is_none() {
        cmd.insert_resource(build_dcpf_model(common, mat, trafos));
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
            (
                ensure_dcpf_model.run_if(
                    resource_exists::<DcpfSolverActive>
                        .and_then(not(resource_exists::<DcpfModel>)),
                ),
                ecs_run_dcpf_pf.run_if(
                    resource_exists::<DcpfSolverActive>
                        .and_then(resource_exists::<DcpfModel>),
                ),
            )
                .chain()
                .in_set(SolverStage::Solve)
                .in_set(PowerFlowSolverSet),
        );
    }
}
