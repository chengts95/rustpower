//! Shared preparation boundary for AC solves and standalone DC queries.
//! Dependency changes arrive through messages; no component change ticks are used.

use bevy_ecs::{prelude::*, schedule::ScheduleLabel};
use nalgebra::DVector;
use num_complex::Complex64;

use super::{mutation::consume_param_diffs, structure_update::structure_update};
use crate::basic::{ecs::dcpf::DcpfWorkspace, newtonpf::NewtonCache, solver::Solve};

#[derive(ScheduleLabel, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PFPrepare;

/// Cache policy survives invalidation and full model rebuilds.
#[derive(Resource)]
pub struct SolverCachePolicy {
    pub enabled: bool,
}

impl Default for SolverCachePolicy {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Changes whenever a dependency of the DC model changes, including bus order
/// and slack voltage. Standalone DC workspaces use this to validate their model.
#[derive(Resource, Default, Clone, Copy)]
pub struct DcpfModelRevision(pub u64);

/// Present only when the current result was produced with NewtonCache.
/// Other solver plugins can publish results without exposing stale NR buffers.
#[derive(Resource)]
pub struct CachedNewtonResult;

/// A continuation within one external solve must retain the last AC voltage,
/// rather than run the initial-voltage strategy again.
#[derive(Resource, Default)]
pub struct SolvePass {
    pub continuation: bool,
    pub(crate) voltage: Option<DVector<Complex64>>,
}

pub fn begin_solve(mut pass: ResMut<SolvePass>) {
    *pass = SolvePass::default();
}

pub fn ensure_newton_cache(world: &mut World) {
    world.remove_resource::<CachedNewtonResult>();
    let enabled = world
        .get_resource::<SolverCachePolicy>()
        .is_none_or(|p| p.enabled);
    if enabled {
        world.init_resource::<NewtonCache>();
    } else {
        world.remove_resource::<NewtonCache>();
    }
}

pub fn invalidate_dcpf_model(world: &mut World) {
    world.init_resource::<DcpfModelRevision>();
    world.resource_mut::<DcpfModelRevision>().0 += 1;
    world.remove_resource::<crate::basic::dcpf::DcpfModel>();
    if let Some(mut ws) = world.get_resource_mut::<DcpfWorkspace>() {
        ws.solver.reset();
    }
}

/// Explicit invalidation leaves the default-on policy intact. Numerical caches
/// are recreated lazily at the next solve; the physical network is retained.
pub fn clear_solver_caches(world: &mut World) {
    world.remove_resource::<CachedNewtonResult>();
    world.remove_resource::<NewtonCache>();
    super::structure_update::reset_solvers(world);
}

pub fn build_prepare_schedule() -> Schedule {
    let mut schedule = Schedule::new(PFPrepare);
    schedule.add_systems((consume_param_diffs, structure_update).chain());
    schedule
}

pub fn prepare_solver_inputs(world: &mut World) {
    world.run_schedule(PFPrepare);
}

#[cfg(test)]
#[path = "prepare_tests.rs"]
mod tests;
