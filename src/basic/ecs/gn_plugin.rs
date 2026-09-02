//! Classical Gauss–Newton LM power flow as an ECS plugin — the same swap-in
//! pattern as [`super::plugin::IwamotoPlugin`]: a `*_run_pf` system plus a
//! plugin that takes over the solve stage when [`ActiveSolver`] selects it
//! (the default NR system only runs for `ActiveSolver::NewtonRaphson`).
//!
//! Pure-Rust usage:
//! ```ignore
//! let mut app = default_app();
//! app.add_plugins(GnPlugin);
//! app.world_mut().insert_resource(PPNetwork(net));
//! app.world_mut().insert_resource(ActiveSolver::GaussNewtonLm);
//! app.update();
//! ```

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::lm::gn_flat::newton_pf_gn;

use super::network::SolverStage;
use super::plugin::{ActiveSolver, PowerFlowSolverSet};
use super::powerflow::systems::{PowerFlowConfig, PowerFlowMat, PowerFlowResult};
use crate::basic::solver::DefaultLmSolver;

/// Solver state owned by [`GnPlugin`]; see [`super::lm_plugin::LmSolverState`]
/// for why the LM family does not share the NR path's `PowerFlowSolver`.
#[derive(Resource, Default)]
pub struct GnSolverState {
    pub solver: DefaultLmSolver,
}

/// ECS system: classical Gauss–Newton LM (no Hessian term).
pub fn gn_run_pf(
    mut cmd: Commands,
    mat: Res<PowerFlowMat>,
    cfg: Res<PowerFlowConfig>,
    mut state: ResMut<GnSolverState>,
) {
    if mat.npv + mat.npq >= mat.v_bus_init.len() {
        cmd.insert_resource(PowerFlowResult {
            v: mat.v_bus_init.clone_owned(),
            iterations: 0,
            converged: false,
        });
        return;
    }

    let v = newton_pf_gn(
        &mat.y_bus,
        &mat.s_bus,
        &mat.v_bus_init,
        mat.npv,
        mat.npq,
        cfg.tol,
        cfg.max_it,
        &mut state.solver,
    );

    match v {
        Ok((v, iterations)) => {
            cmd.insert_resource(PowerFlowResult {
                v,
                iterations,
                converged: true,
            });
        }
        Err((_err, v_err, its)) => {
            cmd.insert_resource(PowerFlowResult {
                v: v_err,
                iterations: its,
                converged: false,
            });
        }
    }
}

/// Plugin for running power flow with classical Gauss–Newton LM.
#[derive(Default)]
pub struct GnPlugin;

impl Plugin for GnPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GnSolverState>();
        app.add_systems(
            Update,
            gn_run_pf
                .in_set(SolverStage::Solve)
                .in_set(PowerFlowSolverSet)
                .run_if(resource_equals(ActiveSolver::GaussNewtonLm)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::ecs::elements::PPNetwork;
    use crate::basic::ecs::network::PowerFlow;
    use crate::basic::ecs::plugin::default_app;
    use crate::io::pandapower::Network;

    /// 经典 GN-LM 插件路径：IEEE39 标准算例。
    #[test]
    fn gn_plugin_ieee39_standard_case() {
        let net: Network = serde_json::from_str(crate::testcases::case_ieee39::IEEE_39).unwrap();
        let mut app_gn = default_app();
        app_gn.add_plugins(GnPlugin);
        app_gn.world_mut().insert_resource(PPNetwork(net));
        app_gn.world_mut().insert_resource(ActiveSolver::GaussNewtonLm);
        app_gn.update();
        let r_gn = app_gn
            .world()
            .get_resource::<PowerFlowResult>()
            .expect("no PowerFlowResult after GN run")
            .clone();
        println!(
            "IEEE39 插件路径: GN-LM converged={} it={}",
            r_gn.converged, r_gn.iterations
        );
        assert!(r_gn.converged, "GN plugin failed to converge on IEEE39");
    }
}
