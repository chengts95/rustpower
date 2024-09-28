use std::fmt;

use bevy_app::prelude::*;
use bevy_ecs::{prelude::*, system::RunSystemOnce};
use nalgebra::*;
use nalgebra_sparse::*;
use num_complex::Complex64;

use crate::{basic::{
    self, newton_pf, solver::RSparseSolver, system::{PFNetwork, RunPF}
}, io::pandapower::ecs_net_conv::init_pf};

use super::elements::*;

/// Represents the ground node in the network.
pub const GND: i64 = -1;

/// Extracts nodes from the ECS world based on a given extractor function.
///
/// # Parameters
/// - `world`: The ECS world containing the nodes.
/// - `extractor`: A closure that defines how to extract specific node information from `NodeType`.
///
/// # Returns
/// A vector of extracted node information based on the provided extractor.
fn extract_node<T, F>(world: &mut World, extractor: F) -> Vec<T>
where
    F: Fn(&NodeType) -> Option<T>,
{
    world
        .query::<&NodeType>()
        .iter(world)
        .filter_map(extractor)
        .collect()
}

/// Represents the power grid, managing the ECS world for power flow calculations.
#[derive(Default)]
pub struct PowerGrid {
    data_storage: App,
}

/// Resource that wraps the power flow network (PFNetwork).
#[derive(Debug, Resource, Clone)]
pub struct ResPFNetwork(pub PFNetwork);



/// Resource that holds the power flow configuration options, such as the initial voltage guess,
/// maximum iterations, and tolerance for convergence.
#[derive(Debug, Resource, Clone)]
pub struct PowerFlowConfig {
    pub v_init: DVector<Complex64>, // Initial voltage vector
    pub max_it: Option<usize>,      // Maximum number of iterations
    pub tol: Option<f64>,           // Tolerance for convergence
}

/// Resource for storing the results of power flow calculation, including the final voltage vector,
/// number of iterations taken, and whether the solution converged.
#[derive(Debug, Default, Resource, Clone)]
pub struct PowerFlowResult {
    pub v: DVector<Complex64>, // Final voltage vector after convergence
    pub iterations: usize,     // Number of iterations taken
    pub converged: bool,       // Convergence status
}

/// Resource holding various matrices required for power flow calculations, including the reordered
/// matrix, admittance matrix (Y-bus), and the power injection vector (S-bus).
#[derive(Debug, Resource, Clone)]
pub struct PowerFlowMat {
    pub reorder: CsrMatrix<Complex<f64>>, // Reordering matrix
    pub y_bus: CscMatrix<Complex<f64>>,   // Y-bus admittance matrix
    pub s_bus: DVector<Complex64>,        // S-bus power injections
    pub npv: usize,                       // Number of PV buses
    pub npq: usize,                       // Number of PQ buses
}

/// Trait for performing operations on ECS data, such as getting and mutating components of entities.
pub trait DataOps {
    fn get_entity_mut(&mut self, entity: Entity) -> Option<EntityWorldMut<'_>>;
    fn get_mut<T>(&mut self, entity: Entity) -> Option<Mut<T>>
    where
        T: Component;
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
        let pf_net: PFNetwork = self.try_into().unwrap();
        let v_init_bak = pf_net.create_v_init();
        let (reorder, y_bus, s_bus, _, npv, npq) = pf_net.prepare_matrices(v_init_bak.clone());
        let mat = PowerFlowMat {
            reorder,
            y_bus,
            s_bus,
            npv,
            npq,
        };
        self.world_mut().insert_resource(mat);
        self.world_mut().insert_resource(PowerFlowConfig {
            v_init: v_init_bak,
            max_it: None,
            tol: None,
        });
        self.world_mut().insert_resource(ResPFNetwork(pf_net));
    }

    fn run_pf(&mut self) {
        // Executes the power flow system once within the ECS world.
        self.world_mut().run_system_once(ecs_run_pf);
    }
}

/// ECS system that runs the power flow calculation based on the current configuration and matrices.
///
/// # Parameters
/// - `cmd`: Command buffer to insert the result resource.
/// - `mat`: Power flow matrices resource.
/// - `cfg`: Power flow configuration resource.
fn ecs_run_pf(mut cmd: Commands, mat: Res<PowerFlowMat>, cfg: Res<PowerFlowConfig>) {
    let v_init = &mat.reorder * &cfg.v_init;
    let max_it = cfg.max_it;
    let tol = cfg.tol;

    #[cfg(feature = "klu")]
    let mut solver = KLUSolver::default();
    #[cfg(not(feature = "klu"))]
    let mut solver = RSparseSolver {};

    let v = newton_pf(
        &mat.y_bus,
        &mat.s_bus,
        &v_init,
        mat.npv,
        mat.npq,
        tol,
        max_it,
        &mut solver,
    );

    // Handle the results of the power flow calculation.
    match v {
        Ok((v, iterations)) => {
            let v = mat.reorder.transpose() * v;
            cmd.insert_resource(PowerFlowResult {
                v,
                iterations,
                converged: true,
            });
        }
        Err((_err, v_err)) => {
            let v = mat.reorder.transpose() * v_err;
            cmd.insert_resource(PowerFlowResult {
                v,
                iterations: 0,
                converged: false,
            });
        }
    }
}

impl DataOps for PowerGrid {
    fn world(&self) -> &World {
        self.data_storage.world()
    }
    fn world_mut(&mut self) -> &mut World {
        self.data_storage.world_mut()
    }
    fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.world().get(entity)
    }
    fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<Mut<T>> {
        self.world_mut().get_mut(entity)
    }
    fn get_entity_mut(&mut self, entity: Entity) -> Option<EntityWorldMut<'_>> {
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

impl TryFrom<&mut PowerGrid> for PFNetwork {
    type Error = ParseError;

    fn try_from(value: &mut PowerGrid) -> Result<Self, Self::Error> {
        use crate::basic::new_ecs::network::DataOps;
        let world = value.world_mut();
        if world.get_resource::<PPNetwork>().is_none() {
            return Err(ParseError::ConversionError(
                "Net resource not found".to_string(),
            ));
        }
        world.run_system_once(init_pf);
        let net = &world.get_resource::<PPNetwork>().unwrap();
        let buses = net.bus.clone();
        let v_base = net.bus[0].vn_kv;
        let s_base = net.sn_mva;
        let pq_loads = extract_node(world, |x| {
            if let NodeType::PQ(v) = x {
                Some(v.clone())
            } else {
                None
            }
        });
        let pv_nodes = extract_node(world, |x| {
            if let NodeType::PV(v) = x {
                Some(v.clone())
            } else {
                None
            }
        });
        let binding = extract_node(world, |x| {
            if let NodeType::EXT(v) = x {
                Some(v.clone())
            } else {
                None
            }
        });
        let ext = binding
            .get(0)
            .ok_or_else(|| ParseError::ConversionError("No external node found".to_string()))?;
        let ext = ext.clone();
        let y_br: Vec<_> = world
            .query::<(&Admittance, &Port2, &VBase)>()
            .iter(world)
            .map(|(a, p, vb)| basic::system::AdmittanceBranch {
                y: basic::system::Admittance(a.0),
                port: basic::system::Port2(p.0.cast()),
                v_base: vb.0,
            })
            .collect();

        let net = PFNetwork {
            v_base,
            s_base,
            buses,
            pq_loads,
            pv_nodes,
            ext,
            y_br,
        };
        Ok(net)
    }
}
#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use bevy_ecs::system::RunSystemOnce;
    use nalgebra::ComplexField;

    use crate::{
        basic::{
            self,
            system::{PFNetwork, RunPF},
        },
        io::pandapower::load_csv_zip,
    };

    use super::*;
    use std::env;
    #[test]
    fn test_to_pf_net() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        let net = load_csv_zip(&name).unwrap();

        let mut pf_net = PowerGrid::default();
        pf_net.world_mut().insert_resource(PPNetwork(net));
        let net = PFNetwork::try_from(&mut pf_net).unwrap();
        let v_init = net.create_v_init();
        let tol = Some(1e-8);
        let max_it = Some(10);
        let (_v, _iter) = net.run_pf(v_init.clone(), max_it, tol);
    }


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
