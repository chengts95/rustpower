use bevy_app::prelude::*;
use bevy_ecs::{prelude::*, system::RunSystemOnce};
use nalgebra_sparse::CsrMatrix;
use nalgebra::DVector;
use num_complex::Complex64;
use crate::basic::ecs::powerflow::systems::create_y_bus;

/// Incremental resource to store branch-related matrices for analysis.
/// Only populated if BranchAnalysisPlugin is added.
#[derive(Debug, Resource, Clone)]
pub struct BranchAnalysisRes {
    pub incidence: CsrMatrix<Complex64>,
    pub branch_y: DVector<Complex64>,
}

pub struct BranchAnalysisPlugin;

impl Plugin for BranchAnalysisPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostStartup, capture_branch_data_system);
    }
}

fn capture_branch_data_system(world: &mut World) {
    if let Ok((incidence, _ybus)) = world.run_system_once(create_y_bus) {
        world.insert_resource(BranchAnalysisRes {
            incidence,
            branch_y: DVector::zeros(0), // Placeholder
        });
    }
}
