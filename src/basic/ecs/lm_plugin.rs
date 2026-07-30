//! Exact-LM second-order power flow as an ECS plugin — the same swap-in
//! pattern as [`super::plugin::IwamotoPlugin`]: a `*_run_pf` system plus a
//! plugin that takes over the solve stage when `CustomSolverActive` is
//! present (the default NR system is bypassed by its `run_if`).
//!
//! Pure-Rust usage:
//! ```ignore
//! let mut app = default_app();
//! app.add_plugins(LmPlugin);
//! app.world_mut().insert_resource(PPNetwork(net));
//! app.world_mut().insert_resource(CustomSolverActive);
//! app.update();
//! ```

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::basic::kkt::exact::driver::newton_pf_lm;

use super::network::{PowerFlowSolver, SolverStage};
use super::plugin::{CustomSolverActive, DefaultSolverSet, PowerFlowSolverSet};
use super::powerflow::systems::{PowerFlowConfig, PowerFlowMat, PowerFlowResult};

/// ECS system: exact-LM power flow on the flat augmented KKT system
/// (mirrors [`super::network::iwamoto_run_pf`] line for line).
pub fn lm_run_pf(
    mut cmd: Commands,
    mat: Res<PowerFlowMat>,
    cfg: Res<PowerFlowConfig>,
    mut solver: ResMut<PowerFlowSolver>,
) {
    if mat.npv + mat.npq >= mat.v_bus_init.len() {
        cmd.insert_resource(PowerFlowResult {
            v: mat.v_bus_init.clone_owned(),
            iterations: 0,
            converged: false,
        });
        return;
    }

    let v = newton_pf_lm(
        &mat.y_bus,
        &mat.s_bus,
        &mat.v_bus_init,
        mat.npv,
        mat.npq,
        cfg.tol,
        cfg.max_it,
        &mut solver.solver,
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

/// Plugin for running power flow with the exact-LM second-order method.
#[derive(Default)]
pub struct LmPlugin;

impl Plugin for LmPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            Update,
            DefaultSolverSet.run_if(not(resource_exists::<CustomSolverActive>)),
        );
        app.add_systems(
            Update,
            lm_run_pf
                .in_set(SolverStage::Solve)
                .in_set(PowerFlowSolverSet)
                .run_if(resource_exists::<CustomSolverActive>),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::ecs::elements::PPNetwork;
    use crate::basic::ecs::network::{DataOps, PowerFlow};
    use crate::basic::ecs::plugin::default_app;
    use crate::io::pandapower::Network;

    /// 标准算例走插件路径：default_app + LmPlugin，与默认 NR 逐点对比。
    #[test]
    fn lm_plugin_ieee39_standard_case() {
        // LM 插件路径
        let net: Network = serde_json::from_str(crate::testcases::case_ieee39::IEEE_39).unwrap();
        let mut app_lm = default_app();
        app_lm.add_plugins(LmPlugin);
        app_lm.world_mut().insert_resource(PPNetwork(net));
        app_lm.world_mut().insert_resource(CustomSolverActive);
        app_lm.update();
        let r_lm = app_lm
            .world()
            .get_resource::<PowerFlowResult>()
            .expect("no PowerFlowResult after LM run")
            .clone();

        // 默认 NR 路径（同一个 app 模板，不加插件）
        let net2: Network = serde_json::from_str(crate::testcases::case_ieee39::IEEE_39).unwrap();
        let mut app_nr = default_app();
        app_nr.world_mut().insert_resource(PPNetwork(net2));
        app_nr.update();
        let r_nr = app_nr
            .world()
            .get_resource::<PowerFlowResult>()
            .expect("no PowerFlowResult after NR run")
            .clone();

        println!(
            "IEEE39 插件路径: LM converged={} it={} | 默认NR converged={} it={}",
            r_lm.converged, r_lm.iterations, r_nr.converged, r_nr.iterations
        );
        assert!(r_lm.converged, "LM plugin failed to converge on IEEE39");
        assert!(r_nr.converged, "default NR failed to converge on IEEE39");

        let max_diff = r_lm
            .v
            .iter()
            .zip(r_nr.v.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).norm()));
        println!("max |v_LM - v_NR| = {max_diff:.3e}");
        assert!(max_diff < 1e-4, "LM and NR solutions disagree on IEEE39");
    }
}
