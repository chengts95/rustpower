//! Explicit addon for Ybus ordering. Legacy solvers and default schedules are unchanged.
use super::ecs::powerflow::systems::PowerFlowMat;
use super::ecs::{
    network::{PowerFlowSolver, SolverStage},
    plugin::{CustomSolverActive, PowerFlowSolverSet},
};
use super::{newtonpf::NewtonCache, solver::KLUSolver};
use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use rustpower_amd::klu::{KluOrdering, YbusOrderingBackend};

pub fn default_backend() -> YbusOrderingBackend {
    if cfg!(feature = "klu_ybus_amd_rust") {
        YbusOrderingBackend::Rust
    } else {
        YbusOrderingBackend::C
    }
}
#[derive(Resource)]
pub struct AmdOrderingState {
    pub ordering: KluOrdering,
}
impl Default for AmdOrderingState {
    fn default() -> Self {
        Self {
            ordering: KluOrdering::new(default_backend()),
        }
    }
}
#[derive(Default)]
pub struct AmdOrderingPlugin;
impl Plugin for AmdOrderingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AmdOrderingState>();
        app.add_systems(
            Update,
            prepare_amd
                .in_set(SolverStage::Solve)
                .after(super::ecs::powerflow::prepare::ensure_newton_cache)
                .before(PowerFlowSolverSet)
                .run_if(|custom: Option<Res<CustomSolverActive>>| custom.is_none()),
        );
    }
}
fn prepare_amd(
    mat: Res<PowerFlowMat>,
    mut solver: ResMut<PowerFlowSolver>,
    mut state: ResMut<AmdOrderingState>,
    mut cache: Option<ResMut<NewtonCache>>,
) {
    prepare_klu(
        &mut state.ordering,
        &mut solver.solver,
        &mat.y_bus,
        mat.npv,
        mat.npq,
        cache.as_deref_mut(),
    );
}
pub(crate) fn prepare_klu(
    ordering: &mut KluOrdering,
    solver: &mut KLUSolver,
    ybus: &CscMatrix<Complex64>,
    npv: usize,
    npq: usize,
    cache: Option<&mut NewtonCache>,
) {
    if ordering.prepare_ybus(
        &mut solver.0,
        ybus.col_offsets(),
        ybus.row_indices(),
        npv,
        npq,
    ) {
        if let Some(cache) = cache {
            cache.j_pattern = None;
        }
    }
}
