use bevy_app::prelude::*;
use bevy_ecs::{prelude::*, system::SystemParam};
use nalgebra::DVector;

use crate::{
    basic::{
        ecs::{
            elements::*,
            powerflow::init::{PQBus, PVBus, SlackBus},
        },
        sparse::cast::Cast,
    },
    prelude::ecs::network::SolverStage::*,
};

use super::{
    nonlinear_schedule::*,
    structure_update::{NodeTypeChangeEvent, StructureUpdatePlugin},
    systems::{PowerFlowMat, PowerFlowResult},
};
#[derive(SystemParam)]
pub struct QLimEnv<'w, 's> {
    buses: Res<'w, NodeLookup>,
    common: Res<'w, PFCommonData>,
    res: Res<'w, PowerFlowResult>,
    mat: Res<'w, PowerFlowMat>,
    res_convergence: ResMut<'w, ConvergedResult>,
    pass: ResMut<'w, super::prepare::SolvePass>,
    node_agg: Option<Res<'w, NodeAggRes>>,
    cache: Option<Res<'w, crate::basic::newtonpf::NewtonCache>>,
    cached_result: Option<Res<'w, super::prepare::CachedNewtonResult>>,
    generators: Query<
        'w,
        's,
        (&'static TargetBus, Option<&'static PQLim>),
        (With<TargetPMW>, With<TargetVmPu>, Without<OutOfService>),
    >,
    pf_bus: Query<'w, 's, &'static SBusInjPu, (With<PVBus>, Without<SlackBus>)>,
}

/// Solver constraint state; the device injection remains unchanged.
/// Retained across parameter updates; full initialization resets bus types.
#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct QLimited {
    pub q_pu: f64,
}

#[derive(Clone, Copy)]
struct BusQLimits {
    min: f64,
    max: f64,
}

/// Checks if reactive power output of PV buses exceeds their generator Q limits.
/// If so, downgrades the bus from PV to PQ type, clamps the Q value, and triggers structural update.
///
/// # Behavior:
/// - Computes current injected Q at each PV bus based on YBus and VBus.
/// - For each  PV-node generator:
///   - If `Q` is out of bounds, switch bus to PQ and update injection value.
///   - Sets `ConvergedResult` to `Continue` to trigger further NR iterations.
///   - Emits `NodeTypeChangeEvent` to notify matrix structure update.
///   
/// # Dependencies:
/// - Requires PV bus tags, target voltage/magnitude, and generator Q limits.
/// - Must be scheduled **after** each nonlinear solve attempt.
/// - This relies on [`NonLinearSchedulePlugin`] and `[StructureUpdatePlugin]`.
///
/// # Notes:
/// - Combines the Q limits of in-service generators sharing a PV bus.
/// - Requires consistent ordering with matrix reordering / aggregation structure.
fn modify_qlim_system(
    mut cmd: Commands,
    mut event: MessageWriter<NodeTypeChangeEvent>,
    env: QLimEnv,
    mut limits: Local<Vec<Option<BusQLimits>>>,
) {
    let QLimEnv {
        buses,
        common,
        res,
        mat,
        mut res_convergence,
        mut pass,
        node_agg,
        cache,
        cached_result,
        generators,
        pf_bus,
    } = env;

    if !res.converged {
        return;
    }
    let n = res.v.len();
    let mut sbus_res = DVector::zeros(n);
    if let Some(c) = cache
        .as_ref()
        .filter(|c| cached_result.is_some() && c.s_calc.len() == n)
    {
        for (perm_idx, &orig_idx) in mat.from_perm.iter().enumerate() {
            sbus_res[orig_idx] = c.s_calc[perm_idx];
        }
    } else {
        // Fallback when NewtonCache is empty or not enabled:
        let mut v_perm = DVector::zeros(n);
        for (new_idx, &orig_idx) in mat.from_perm.iter().enumerate() {
            v_perm[new_idx] = res.v[orig_idx];
        }
        let mis_perm = v_perm.component_mul(&(&mat.y_bus * &v_perm).conjugate());
        for (new_idx, &orig_idx) in mat.from_perm.iter().enumerate() {
            sbus_res[orig_idx] = mis_perm[new_idx];
        }
    }

    let sbus_res = match &node_agg {
        Some(node_agg) => &node_agg.expand_mat.cast() * &sbus_res,
        None => sbus_res,
    };
    limits.resize(sbus_res.len(), None);
    limits.fill(None);
    for (bus, lim) in &generators {
        let Some(entry) = limits.get_mut(bus.0 as usize) else {
            continue;
        };
        let (min, max) = lim.map_or((f64::NEG_INFINITY, f64::INFINITY), |l| (l.q.min, l.q.max));
        let total = entry.get_or_insert(BusQLimits { min: 0.0, max: 0.0 });
        total.min += min;
        total.max += max;
    }
    let mut structure_change = false;
    for (bus, limit) in limits.iter().enumerate() {
        let Some(limit) = limit else {
            continue;
        };
        let Some(entity) = buses.get_entity(bus as i64) else {
            continue;
        };
        let Ok(injection) = pf_bus.get(entity) else {
            continue;
        };
        let q_mvar = (sbus_res[bus].im - injection.0.im) * common.sbase;
        let bound = if q_mvar < limit.min {
            Some(limit.min)
        } else if q_mvar > limit.max {
            Some(limit.max)
        } else {
            None
        };
        if let Some(q_mvar) = bound {
            // Keep the clamp separate so injection updates do not count it twice.
            cmd.entity(entity).remove::<PVBus>().insert((
                PQBus,
                QLimited {
                    q_pu: q_mvar / common.sbase,
                },
            ));
            structure_change = true;
        }
    }
    if structure_change {
        pass.continuation = true;
        pass.voltage = Some(res.v.clone());
        res_convergence.converged = NonlinearConvType::Continue;
        event.write(NodeTypeChangeEvent);
    }
}

/// Plugin responsible for enforcing generator reactive power limits (Q-limits)
/// by converting PV buses to PQ when violations are detected during power flow simulation.
///
/// # Responsibilities
/// - Checks generator Q output after each complete AC solve.
/// - Ensures compatibility with the power flow structure update system and nonlinear solver schedule.
///
/// # Plugin Dependencies
/// This plugin automatically adds the following if not already present:
/// - [`StructureUpdatePlugin`] – for handling matrix structure rebuilds when node types change.
/// - [`NonLinearSchedulePlugin`] – to support iterative NR-based solving with convergence checks.
///
/// # System Scheduling
/// - [`modify_qlim_system`] is added to the `Update` stage under the `AfterSolve` system set.
///   It must run *after* the main power flow solve step and *before* structure rebuild.
#[derive(Default)]
pub struct QLimPlugin;

impl Plugin for QLimPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<StructureUpdatePlugin>() {
            //panic!("QLimPlugin requires StructureUpdatePlugin to be added before it.");
            app.add_plugins(StructureUpdatePlugin);
        }
        if !app.is_plugin_added::<NonLinearSchedulePlugin>() {
            //panic!("QLimPlugin requires StructureUpdatePlugin to be added before it.");
            app.add_plugins(NonLinearSchedulePlugin);
        }
        app.add_systems(Update, modify_qlim_system.in_set(AfterSolve));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::ecs::plugin::default_app;
    use crate::basic::ecs::post_processing::PostProcessing;
    use crate::io::pandapower::load_csv_zip;
    use crate::prelude::PPNetwork;
    use std::env;

    #[test]
    fn test_qlim_118_outer_iteration() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let name = format!("{}/cases/IEEE118/data.zip", dir);
        let net = load_csv_zip(&name).unwrap();

        let mut app = default_app();
        app.add_plugins(QLimPlugin);
        app.world_mut().insert_resource(PPNetwork(net));
        app.update();

        let res = app
            .world()
            .get_resource::<super::super::systems::PowerFlowResult>()
            .unwrap();
        assert!(res.converged, "qlim outer iteration must converge");
        app.post_process();
        app.print_res_bus();
    }
}

#[cfg(test)]
mod tests_python_composition {
    use super::*;
    use crate::basic::ecs::plugin::BasePFPlugin;
    use crate::basic::ecs::powerflow::pf_init::PFInit;
    use crate::basic::ecs::powerflow::result_extract::VBusUpdatePlugin;
    use crate::io::pandapower::load_csv_zip;
    use crate::prelude::PPNetwork;
    use std::env;

    /// Mirrors the plugin set and init flow of the Python PowerGrid.
    #[test]
    fn test_qlim_118_python_plugin_set() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let name = format!("{}/cases/IEEE118/data.zip", dir);
        let net = load_csv_zip(&name).unwrap();

        let mut app = App::new();
        app.world_mut().insert_resource(PFCommonData {
            sbase: 100.0,
            f_hz: 50.0,
            wbase: 2.0 * std::f64::consts::PI * 50.0,
        });
        app.add_plugins((BasePFPlugin, StructureUpdatePlugin, VBusUpdatePlugin));
        app.add_plugins(QLimPlugin);

        app.world_mut().insert_resource(PPNetwork(net));
        let _ = app.world_mut().try_run_schedule(PFInit); // python init_pf()
        app.update(); // python solve()

        let res = app
            .world()
            .get_resource::<super::super::systems::PowerFlowResult>()
            .unwrap();
        assert!(
            res.converged,
            "qlim with the python plugin set must converge"
        );
    }
}
