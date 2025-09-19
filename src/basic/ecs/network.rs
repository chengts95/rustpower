#![allow(deprecated)]
use std::fmt;

use bevy_app::prelude::*;
use bevy_ecs::{component::Mutable, prelude::*, world::error::EntityMutableFetchError};

use crate::basic::{newton_pf, solver::DefaultSolver};

use super::{
    plugin::DefaultPlugins,
    powerflow::{init::BasePFInitPlugins, systems::*},
};
#[derive(Clone, SystemSet, Debug, Hash, PartialEq, Eq)]
pub enum SolverStage {
    BeforeSolve,
    Solve,
    AfterSolve,
}
#[derive(Default, Resource)]
pub struct PowerFlowSolver {
    pub solver: DefaultSolver,
}

/// Represents the ground node in the network.
pub const GND: i64 = -1;

/// Represents the power grid, managing the ECS world for power flow calculations.
#[derive(Default)]
pub struct PowerGrid {
    data_storage: App,
}

/// Trait for performing operations on ECS data, such as getting and mutating components of entities.
pub trait DataOps {
    fn get_entity_mut(
        &mut self,
        entity: Entity,
    ) -> Result<EntityWorldMut<'_>, EntityMutableFetchError>;
    fn get_mut<T>(&'_ mut self, entity: Entity) -> Option<Mut<'_, T>>
    where
        T: Component<Mutability = Mutable>;
    fn get<T>(&self, entity: Entity) -> Option<&T>
    where
        T: Component;
    fn world_mut(&mut self) -> &mut World;
    fn world(&self) -> &World;
}

/// Trait for defining power flow operations, such as initializing and running the power flow calculation.
pub trait PowerFlow {
    /// Initializes the power flow network by preparing matrices and resources required for the computation.
    fn init_pf_net(&mut self);

    /// Runs the power flow calculation using the Newton-Raphson method.
    fn run_pf(&mut self);
}

impl PowerFlow for PowerGrid {
    fn init_pf_net(&mut self) {
        // Initialize the power flow network, prepare matrices, and store them as ECS resources.

        self.world_mut().insert_resource(PowerFlowConfig {
            max_it: None,
            tol: None,
        });

        self.app_mut()
            .add_plugins((BasePFInitPlugins, DefaultPlugins));
        let world = self.world_mut();

        let mut schedules = world.get_resource_mut::<Schedules>().unwrap();

        let mut s = schedules.remove(Startup).unwrap();
        s.run(world);

        //let mut schedules = world.get_resource_mut::<Schedules>().unwrap();
        //.schedules.insert(s); termporarily removed to avoid double insertion
 
    }

    fn run_pf(&mut self) {
        self.app_mut().update();
    }
}

pub fn apply_permutation(mut mat: ResMut<PowerFlowMat>) {
    let reorder = &mat.reorder.clone().transpose_as_csc();
    let y_bus = &mat.y_bus;
    let rt = reorder.transpose();
    let reordered_y_bus = &rt * y_bus * reorder;
    mat.s_bus = &rt * &mat.s_bus;
    mat.v_bus_init = &rt * &mat.v_bus_init;
    mat.y_bus = reordered_y_bus;
}
#[allow(unused)]
fn apply_inversed_permutation(mut mat: ResMut<PowerFlowMat>) {
    let reorder = &mat.reorder.clone().transpose_as_csc();
    let y_bus = &mat.y_bus;
    let rt = reorder.transpose();
    let reordered_y_bus = reorder * y_bus * &rt;
    mat.s_bus = reorder * &mat.s_bus;
    mat.v_bus_init = reorder * &mat.v_bus_init;
    mat.y_bus = reordered_y_bus;
}
/// ECS system that runs the p ower flow calculation based on the current configuration and matrices.
///
/// # Parameters
/// - `cmd`: Command buffer to insert the result resource.
/// - `mat`: Power flow matrices resource.
/// - `cfg`: Power flow configuration resource.
pub fn ecs_run_pf(
    mut cmd: Commands,
    mat: Res<PowerFlowMat>,
    cfg: Res<PowerFlowConfig>,
    mut solver: ResMut<PowerFlowSolver>,
) {
    let v_init = &mat.v_bus_init;
    let max_it = cfg.max_it;
    let tol = cfg.tol;
    let v = newton_pf(
        &mat.y_bus,
        &mat.s_bus,
        &v_init,
        mat.npv,
        mat.npq,
        tol,
        max_it,
        &mut solver.solver,
    );

    // Handle the results of the power flow calculation.
    match v {
        Ok((v, iterations)) => {
            //let v = mat.reorder.transpose() * v;
            let v = v;
            cmd.insert_resource(PowerFlowResult {
                v,
                iterations,
                converged: true,
            });
        }
        Err((_err, v_err)) => {
            // let v = mat.reorder.transpose() * v_err;
            let v = v_err;
            cmd.insert_resource(PowerFlowResult {
                v,
                iterations: 0,
                converged: false,
            });
        }
    }
}
impl PowerGrid {
    pub fn app(&self) -> &App {
        &self.data_storage
    }
    pub fn app_mut(&mut self) -> &mut App {
        &mut self.data_storage
    }
}
impl DataOps for PowerGrid {
    fn world(&self) -> &World {
        self.app().world()
    }
    fn world_mut(&mut self) -> &mut World {
        self.app_mut().world_mut()
    }
    fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.world().get(entity)
    }
    fn get_mut<T: Component>(&'_ mut self, entity: Entity) -> Option<Mut<'_, T>>
    where
        T: Component<Mutability = Mutable>,
    {
        self.world_mut().get_mut(entity)
    }
    fn get_entity_mut(
        &mut self,
        entity: Entity,
    ) -> Result<EntityWorldMut<'_>, EntityMutableFetchError> {
        self.world_mut().get_entity_mut(entity)
    }
}

#[derive(Debug)]
pub enum ParseError {
    InvalidData,
    ConversionError(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::InvalidData => write!(f, "Invalid input data"),
            ParseError::ConversionError(msg) => write!(f, "Conversion failed: {}", msg),
        }
    }
}
impl std::error::Error for ParseError {}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use bevy_ecs::system::RunSystemOnce;
    use nalgebra::ComplexField;

    use crate::{
        basic::{self},
        io::pandapower::load_csv_zip,
        prelude::PPNetwork,
    };

    use super::*;
    use std::env;

    /// Test case for running power flow in the ECS system.
    #[test]
    fn test_ecs_pf() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        let net = load_csv_zip(&name).unwrap();

        let mut pf_net = PowerGrid::default();
        pf_net.world_mut().insert_resource(PPNetwork(net));
        pf_net.init_pf_net();
        pf_net.run_pf();
        assert_eq!(
            pf_net
                .world()
                .get_resource::<PowerFlowResult>()
                .unwrap()
                .converged,
            true
        );
    }
}
