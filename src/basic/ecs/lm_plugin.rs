//! Exact-LM second-order power flow as an ECS plugin — the same swap-in
//! pattern as [`super::plugin::IwamotoPlugin`]: a `*_run_pf` system plus a
//! plugin that takes over the solve stage when [`ActiveSolver`] selects it
//! (the default NR system only runs for `ActiveSolver::NewtonRaphson`).
//!
//! Pure-Rust usage:
//! ```ignore
//! let mut app = default_app();
//! app.add_plugins(LmPlugin);
//! app.world_mut().insert_resource(PPNetwork(net));
//! app.world_mut().insert_resource(ActiveSolver::ExactLm);
//! app.update();
//! ```

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::lm::exact::driver::newton_pf_lm;

use super::network::SolverStage;
use super::plugin::{ActiveSolver, PowerFlowSolverSet};
use super::powerflow::systems::{PowerFlowConfig, PowerFlowMat, PowerFlowResult};
use crate::basic::solver::DefaultLmSolver;

/// Solver state owned by [`LmPlugin`].
///
/// Deliberately separate from `PowerFlowSolver` (the NR path's LU backend):
/// the LM augmented KKT system is symmetric indefinite, so its default
/// backend is an LDLᵀ factorization ([`DefaultLmSolver`] — SuiteSparse LDL
/// with feature `ldl`, pure-Rust QDLDL otherwise). Each LM plugin holds its
/// own instance so selector switches never thrash a shared symbolic.
#[derive(Resource, Default)]
pub struct LmSolverState {
    pub solver: DefaultLmSolver,
}

/// ECS system: exact-LM power flow on the flat augmented KKT system
/// (mirrors [`super::network::iwamoto_run_pf`] line for line).
pub fn lm_run_pf(
    mut cmd: Commands,
    mat: Res<PowerFlowMat>,
    cfg: Res<PowerFlowConfig>,
    mut state: ResMut<LmSolverState>,
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

/// Plugin for running power flow with the exact-LM second-order method.
#[derive(Default)]
pub struct LmPlugin;

impl Plugin for LmPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LmSolverState>();
        app.add_systems(
            Update,
            lm_run_pf
                .in_set(SolverStage::Solve)
                .in_set(PowerFlowSolverSet)
                .run_if(resource_equals(ActiveSolver::ExactLm)),
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
        app_lm.world_mut().insert_resource(ActiveSolver::ExactLm);
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

    /// 两个 LM 插件同时注册在同一个 app 上：ActiveSolver 枚举必须精确选中
    /// 一个，另一家不运行。布尔劫持时代这会双跑 + 结果互相覆盖。
    #[test]
    fn active_solver_disambiguates_coexisting_plugins() {
        use crate::basic::ecs::gn_plugin::GnPlugin;

        let net: Network = serde_json::from_str(crate::testcases::case_ieee39::IEEE_39).unwrap();
        let mut app = default_app();
        app.add_plugins((GnPlugin, LmPlugin));
        app.world_mut().insert_resource(PPNetwork(net));

        // 选中 exact-LM：跑出的迭代数应与单独 LmPlugin 一致
        app.world_mut().insert_resource(ActiveSolver::ExactLm);
        app.update();
        let r_exact = app
            .world()
            .get_resource::<PowerFlowResult>()
            .expect("no PowerFlowResult with ExactLm selected")
            .clone();
        assert!(r_exact.converged, "ExactLm selected but did not converge");

        // 运行中切换 selector：下一帧应走 GN-LM
        app.world_mut().insert_resource(ActiveSolver::GaussNewtonLm);
        app.update();
        let r_gn = app
            .world()
            .get_resource::<PowerFlowResult>()
            .expect("no PowerFlowResult after switching to GaussNewtonLm")
            .clone();
        assert!(r_gn.converged, "GaussNewtonLm selected but did not converge");

        // 两家都收敛到同一解（标准算例），但迭代数一般不同——
        // 证明确实是两个不同的 solver 各自跑了一次，而不是某家空转。
        let max_diff = r_exact
            .v
            .iter()
            .zip(r_gn.v.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).norm()));
        println!(
            "共存测试: exact-LM it={} | GN-LM it={} | max|Δv|={max_diff:.3e}",
            r_exact.iterations, r_gn.iterations
        );
        assert!(max_diff < 1e-4, "coexisting LM plugins disagree on IEEE39");
    }
}
