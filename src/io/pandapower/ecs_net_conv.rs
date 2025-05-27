

use crate::basic::ecs::defer_builder::DeferBundleSpawner;
use crate::basic::ecs::network::DataOps;
use crate::basic::ecs::network::PowerGrid;
use crate::basic::ecs::*;

use crate::prelude::pandapower::*;
use bevy_ecs::prelude::*;
use elements::*;

use std::f64::consts::PI;


trait IntoBundleVec<T, U> {
    fn to_bundle_vec(self) -> Vec<U>;
}

impl<T, U> IntoBundleVec<Option<Vec<T>>, U> for Option<Vec<T>>
where
    for<'a> &'a T: Into<U>,
{
    fn to_bundle_vec(self) -> Vec<U> {
        self.unwrap_or_default().iter().map(Into::into).collect()
    }
}

fn inital_setup(mut cmd: Commands, net: Res<PPNetwork>) {
    cmd.insert_resource(PFCommonData {
        wbase: 2.0 * PI * net.f_hz,
        f_hz: net.f_hz,
        sbase: net.sn_mva,
    });
}

pub trait LoadPandapowerNet {
    fn load_pandapower_net(&mut self, net: &Network);
}

impl LoadPandapowerNet for PowerGrid {
    fn load_pandapower_net(&mut self, net: &Network) {
        let world = self.world_mut();
        world.load_pandapower_net(net);
    }
}
impl LoadPandapowerNet for World {
    fn load_pandapower_net(&mut self, net: &Network) {
        let world = self;
        let buses: Vec<BusBundle> = net.bus.iter().map(|x| x.into()).collect();
        let ts: Vec<TransformerBundle> = net.trafo.clone().to_bundle_vec();
        let lines: Vec<LineBundle> = net.line.clone().to_bundle_vec();
        let gens: Vec<GeneratorBundle> = net.r#gen.clone().to_bundle_vec();
        let loads: Vec<LoadBundle> = net.load.clone().to_bundle_vec();
        let ext_grid: Vec<ExtGridBundle> = net.ext_grid.clone().to_bundle_vec();
        let shunts: Vec<ShuntBundle> = net.shunt.clone().to_bundle_vec();
        let sgens: Vec<SGenBundle> = net.sgen.clone().to_bundle_vec();
        let switches: Vec<SwitchBundle> = net.switch.clone().to_bundle_vec();
        world.commands().spawn_batch(buses);
        world.flush();

        let mut spawner = DeferBundleSpawner::new();
        spawner.spawn_batch(world, ts);
        spawner.spawn_batch(world, lines);
        spawner.spawn_batch(world, gens);
        spawner.spawn_batch(world, loads);
        spawner.spawn_batch(world, ext_grid);
        spawner.spawn_batch(world, shunts);
        spawner.spawn_batch(world, sgens);
        spawner.spawn_batch(world, switches);
        world.insert_resource(PFCommonData {
            wbase: net.f_hz * 2.0 * std::f64::consts::PI,
            f_hz: net.f_hz,
            sbase: net.sn_mva,
        });
    }
}

pub fn pandapower_init_system(world: &mut World) {
    let net = world.remove_resource::<PPNetwork>();
    if let Some(net) = net {
        world.load_pandapower_net(&net.0);
    }
}
pub fn init_powergrid_from_net(net: &Network, world: &mut World) {
    world.load_pandapower_net(net);
}

#[allow(unused_imports)]
mod tests {
    use bevy_ecs::system::RunSystemOnce;
    use nalgebra::ComplexField;
    use network::{DataOps, PowerGrid};

    use crate::basic;

    use super::*;
    use std::env;

    #[test]
    /// Test function for loading and running the power flow system using a CSV zip file.
    fn test_load_csv_zip() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        // let net = load_csv_zip(&name).unwrap();
        // let mut pf_net = PowerGrid::default();
        // let world = pf_net.world_mut();
        // println!("{}", net.bus.len());
        // world.insert_resource(PPNetwork(net));
        // world.run_system_once(init_pf).unwrap();
        // let mut a = world.query::<(&Transformer, &Port2)>();

        // println!("{:?}", a.iter(world).collect::<Vec<_>>().len());
    }
}
