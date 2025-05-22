use super::bus::*;
use super::line::*;

use super::PPNetwork;
use super::SwitchSnapShotReg;
use super::switch;
use crate::basic::ecs::defer_builder::*;
use crate::basic::ecs::elements::*;
use crate::basic::ecs::network::PowerGrid;
use crate::basic::ecs::plugin::PFInitStage;
use bevy_app::PreUpdate;
use bevy_app::Startup;
use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::world::World;

use crate::basic::ecs::network::DataOps;
use crate::io::pandapower::Network;
pub use bus::*;
pub use generator::*;
pub use line::*;
pub use load::*;
pub use sgen::*;
pub use shunt::*;
pub use switch::*;
pub use trans::*;
pub use units::*;
pub trait LoadPandapowerNet {
    fn load_pandapower_net(&mut self, net: &Network);
}

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
        let shunts: Vec<shunt::ShuntBundle> = net.shunt.clone().to_bundle_vec();
        let sgens: Vec<SGenBundle> = net.sgen.clone().to_bundle_vec();
        let switches: Vec<switch::SwitchBundle> = net.switch.clone().to_bundle_vec();
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

pub struct DefaultSnapShotReg;
impl SnaptShotRegGroup for DefaultSnapShotReg {
    fn register_snap_shot(registry: &mut SnapshotRegistry) {
        BusSnapShotReg::register_snap_shot(registry);
        TransSnapShotReg::register_snap_shot(registry);
        GenSnapShotReg::register_snap_shot(registry);
        LineSnapshotReg::register_snap_shot(registry);
        LoadSnapshotReg::register_snap_shot(registry);
        ShuntSnapShotReg::register_snap_shot(registry);
        SGenSnapShotReg::register_snap_shot(registry);
        SwitchSnapShotReg::register_snap_shot(registry);
    }
}

pub struct ElementSetupPlugin;
impl bevy_app::Plugin for ElementSetupPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(Startup, (
            bus::systems::init_node_lookup,
            (
            trans::systems::setup_transformer,
            line::systems::setup_line_systems,
            shunt::systems::setup_shunt_systems,
            )
        ).chain().in_set(PFInitStage));

        app.add_systems(PreUpdate, bus::systems::update_node_lookup);
    }
}
pub fn build_snapshot_registry() -> SnapshotRegistry {
    let mut registry = SnapshotRegistry::default();

    DefaultSnapShotReg::register_snap_shot(&mut registry);

    registry
}

#[cfg(test)]
mod test {
    use std::env;

    use bevy_archive::prelude::{
        SnapshotRegistry, load_world_manifest, read_manifest_from_file, save_world_manifest,
    };

    use crate::io::pandapower::load_csv_zip;

    use super::*;

    fn load_net() -> Network {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        let net = load_csv_zip(&name).unwrap();
        net
    }

    #[test]
    fn test_ele_process() {
        let net = load_net();
        let mut pf_net = PowerGrid::default();

        let world = pf_net.world_mut();
        world.load_pandapower_net(&net);
        let registry = build_snapshot_registry();
        let a = save_world_manifest(world, &registry).unwrap();
        a.to_file("test_system.toml", None).unwrap();
        let mut world = World::default();
        let b = read_manifest_from_file("test_system.toml", None).unwrap();
        load_world_manifest(&mut world, &b, &registry).unwrap();
        let a = save_world_manifest(&world, &registry).unwrap();
        a.to_file("test_system.toml", None).unwrap();
    }
}
