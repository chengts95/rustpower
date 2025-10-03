use std::ops::DerefMut;

use bevy_app::prelude::*;
use bevy_ecs::{prelude::*, system::SystemParam};

use crate::{
    basic::{
        ecs::{
            elements::*,
            powerflow::init::{PQBus, PVBus},
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
    mat: ResMut<'w, PowerFlowMat>,
    res_convergence: ResMut<'w, ConvergedResult>,
    node_agg: Option<Res<'w, NodeAggRes>>,
    generators:
        Query<'w, 's, (&'static TargetBus, &'static PQLim), (With<TargetPMW>, With<TargetVmPu>)>,
    pf_bus: Query<'w, 's, &'static mut SBusInjPu, With<PVBus>>,
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
/// - Assumes only **one generator per bus**, or at least uses the first found.
/// - Requires consistent ordering with matrix reordering / aggregation structure.
fn modify_qlim_system(
    mut cmd: Commands,
    mut event: MessageWriter<NodeTypeChangeEvent>,
    env: QLimEnv,
) {
    let QLimEnv {
        buses,
        common,
        res,
        mut mat,
        mut res_convergence,
        node_agg,
        generators,
        mut pf_bus,
    } = env;
    // This system may have trouble since multiple generators can be connected to the same bus.
    let cv = &res.v;
    let mis = &cv.component_mul(&(&mat.y_bus * cv).conjugate());
    let sbus_res = mis;
    let inv_order = &mat.reorder.transpose();
    let sbus_res = inv_order * sbus_res;

    let sbus_res = match &node_agg {
        Some(node_agg) => &node_agg.expand_mat.cast() * &sbus_res,
        None => sbus_res,
    };
    let mut structure_change = false;
    generators
        .iter()
        .map(|d| {
            let bus = d.0.0;
            let e = buses.get_entity(bus).unwrap();
            (e, bus, d.1)
        })
        .for_each(|(e, bus, lim)| {
            if !pf_bus.contains(e) {
                return;
            }
            let mut q_target = pf_bus.get_mut(e).unwrap();
            let q_mvar = (sbus_res[bus as usize].im - q_target.0.im) * common.sbase;
            let qlim = &lim.q;
            if q_mvar < qlim.min {
                structure_change = true;
                cmd.entity(e).remove::<PVBus>().insert(PQBus);
                q_target.deref_mut().0.im = qlim.min / common.sbase;
            }
            if q_mvar > qlim.max {
                structure_change = true;
                cmd.entity(e).remove::<PVBus>().insert(PQBus);
                q_target.deref_mut().0.im = qlim.max / common.sbase;
            }
        });
    if structure_change { 
        mat.v_bus_init.clone_from(&res.v);
        res_convergence.converged = NonlinearConvType::Continue;
        event.write(NodeTypeChangeEvent);
    }
}

/// Plugin responsible for enforcing generator reactive power limits (Q-limits)
/// by converting PV buses to PQ when violations are detected during power flow simulation.
///
/// # Responsibilities
/// - Adds the [`modify_qlim_system`] to check generator Q output after each NR iteration.
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
