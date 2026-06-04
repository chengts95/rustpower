use crate::basic::ecs::network::DataOps;
use crate::basic::ecs::network::PowerGrid;
use crate::basic::ecs::*;
use crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer;

use crate::prelude::pandapower::*;
use bevy_ecs::prelude::*;
use elements::*;

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
        let mut buffer = HarvardCommandBuffer::new();

        // Buses
        for bus in &net.bus {
            let b: BusBundle = bus.into();
            let e = world.spawn_empty().id();
            buffer.insert_bundle(world, e, b);
        }

        // Transformers
        let ts: Vec<TransformerBundle> = net.trafo.clone().to_bundle_vec();
        for t in ts {
            let e = world.spawn_empty().id();
            buffer.insert_bundle(world, e, t);
        }

        // Lines
        let lines: Vec<LineBundle> = net.line.clone().to_bundle_vec();
        for l in lines {
            let e = world.spawn_empty().id();
            buffer.insert_bundle(world, e, l);
        }

        // Generators
        let gens: Vec<GeneratorBundle> = net.r#gen.clone().to_bundle_vec();
        for g in gens {
            let e = world.spawn_empty().id();
            buffer.insert_bundle(world, e, g);
        }

        // Loads
        let loads: Vec<LoadBundle> = net.load.clone().to_bundle_vec();
        for l in loads {
            let e = world.spawn_empty().id();
            buffer.insert_bundle(world, e, l);
        }

        // Ext Grid
        let ext_grid: Vec<ExtGridBundle> = net.ext_grid.clone().to_bundle_vec();
        for g in ext_grid {
            let e = world.spawn_empty().id();
            buffer.insert_bundle(world, e, g);
        }

        // Shunts
        let shunts: Vec<ShuntBundle> = net.shunt.clone().to_bundle_vec();
        for s in shunts {
            let e = world.spawn_empty().id();
            buffer.insert_bundle(world, e, s);
        }

        // SGens
        let sgens: Vec<SGenBundle> = net.sgen.clone().to_bundle_vec();
        for s in sgens {
            let e = world.spawn_empty().id();
            buffer.insert_bundle(world, e, s);
        }

        // Switches
        let switches: Vec<SwitchBundle> = net.switch.clone().to_bundle_vec();
        for s in switches {
            let e = world.spawn_empty().id();
            buffer.insert_bundle(world, e, s);
        }

        buffer.apply(world);

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
        let net = load_csv_zip(&name).unwrap();
        let mut pf_net = PowerGrid::default();
        let world = pf_net.world_mut();
        println!("{}", net.bus.len());
        world.insert_resource(PPNetwork(net));
    }
}
