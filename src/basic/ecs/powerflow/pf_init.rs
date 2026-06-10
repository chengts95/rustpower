//! The unified, re-runnable full-initialization pipeline: the `PFInit`
//! schedule. Single definition of "full rebuild" shared by every caller
//! (Python `init_pf`, `FullRebuildEvent` consumers, future native paths).
//!
//! Rebuild semantics mirror the incremental path exactly: `init_node_lookup`
//! zeroes `SBusInjPu`/`VBusPu`, then the injection systems re-accumulate from
//! the case data — i.e. "zero, then consume all diffs from scratch". The
//! incremental path is the same accumulation without the zeroing.

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ScheduleLabel;

use crate::basic::ecs::elements::bus::bus_systems::init_node_lookup;
use crate::basic::ecs::elements::line::line_systems::setup_line_systems;
use crate::basic::ecs::elements::shunt::shunt_systems::setup_shunt_systems;
use crate::basic::ecs::elements::trans::trans_systems::setup_transformer;
use crate::basic::ecs::network::apply_permutation;
use crate::io::pandapower::ecs_net_conv::pandapower_init_system;

use super::init::{
    PQBus, PVBus, SlackBus, label_pq_nodes, label_pv_nodes, label_slack_nodes, p_mw_inj,
    q_mvar_inj, v_inj,
};
use super::mutation::ParamDiff;
use super::structure_update::reset_solvers;
use super::systems::{PowerFlowMat, PowerFlowResult, init_states};

/// Label of the re-runnable full-initialization schedule.
#[derive(ScheduleLabel, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PFInit;

/// Drop stale projections and pending diff messages; the rebuild re-derives
/// everything from the case data, so applying old diffs afterwards would
/// double-count.
pub fn cleanup_solver_state(world: &mut World) {
    world.remove_resource::<PowerFlowMat>();
    world.remove_resource::<PowerFlowResult>();
    if let Some(mut msgs) = world.get_resource_mut::<bevy_ecs::message::Messages<ParamDiff>>() {
        msgs.clear();
    }
}

/// Clear stale PQ/PV/Slack tags so relabeling reflects the current service
/// state (e.g. a generator switched out of service demotes its bus to PQ).
pub fn clear_node_type_tags(
    mut cmd: Commands,
    q: Query<Entity, Or<(With<PQBus>, With<PVBus>, With<SlackBus>)>>,
) {
    for e in &q {
        cmd.entity(e).remove::<(PQBus, PVBus, SlackBus)>();
    }
}

/// Build the `PFInit` schedule. Registered by `StructureUpdatePlugin`; run
/// via `world.try_run_schedule(PFInit)`.
pub fn build_pf_init_schedule() -> Schedule {
    let mut s = Schedule::new(PFInit);
    s.add_systems(
        (
            // 0. Ingest a pending pandapower network, if any (consumed once)
            pandapower_init_system,
            // 1. Invalidate projections + pending diffs
            cleanup_solver_state,
            // 2. Node lookup; zeroes SBusInjPu / VBusPu ("0 启动")
            init_node_lookup,
            // 3. Physics-ready element state (all idempotent)
            setup_transformer,
            setup_line_systems,
            setup_shunt_systems,
            // 4. Node classification from current service state
            clear_node_type_tags,
            label_pv_nodes,
            label_slack_nodes,
            label_pq_nodes,
            // 5. Consume the case data as diffs from zero -> sums
            (p_mw_inj, q_mvar_inj, v_inj),
            // 6. Rebuild the projection in solver ordering
            init_states,
            apply_permutation,
            // 7. Structure changed: drop cached factorizations
            reset_solvers,
        )
            .chain(),
    );
    s
}
