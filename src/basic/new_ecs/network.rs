
use bevy_app::prelude::*;
use bevy_ecs::{prelude::*, system::RunSystemOnce};
use nalgebra::*;
use nalgebra_sparse::*;
use num_complex::Complex64;

use crate::basic::{
    newton_pf,
    solver::RSparseSolver,
    system::{PFNetwork, RunPF},
};

use super::elements::*;

/// Represents the ground node in the network.
pub const GND: i64 = -1;

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

#[derive(Default)]
pub struct PowerGrid {
    data_storage: App,
}

#[derive(Debug, Resource, Clone)]
pub struct ResPFNetwork(pub PFNetwork);

#[derive(Debug, Component, Clone)]
pub struct SBusResult(pub Complex64);
#[derive(Debug, Component, Clone)]
pub struct VBusResult(pub Complex64);



#[derive(Debug, Resource, Clone)]
pub struct PowerFlowConfig {
    pub v_init: DVector<Complex64>,
    pub max_it: Option<usize>,
    pub tol: Option<f64>,
}
#[derive(Debug, Default, Resource, Clone)]
pub struct PowerFlowResult {
    pub v: DVector<Complex64>,
    pub iterations: usize,
    pub converged: bool,
}
#[derive(Debug, Resource, Clone)]
pub struct PowerFlowMat {
    pub reorder: CsrMatrix<Complex<f64>>,
    pub y_bus: CscMatrix<Complex<f64>>,
    pub s_bus: DVector<Complex64>,
    pub npv: usize,
    pub npq: usize,
}
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


pub trait PowerFlow {
    fn run_pf(&mut self);
    fn init_pf_net(&mut self);
}
impl PowerFlow for PowerGrid {
    fn init_pf_net(&mut self) {
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
        self.world_mut().run_system_once(ecs_run_pf);
    }
}
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
        assert_eq!(pf_net.world().get_resource::<PowerFlowResult>().unwrap().converged,true);
    }
}
