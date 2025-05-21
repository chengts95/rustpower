use super::bus::*;
use super::line::*;

use super::trans::*;
use crate::basic::ecs::defer_builder::*;
use bevy_ecs::world::World;
use bumpalo::Bump;

#[cfg(test)]
mod test {
    use std::env;

    use bevy_archive::prelude::{
        SnapshotRegistry, load_world_manifest, read_manifest_from_file, save_world_manifest,
    };

    use super::*;

    use crate::basic::ecs::elements::generator::{ExtGridBundle, GenSnapShotReg, GeneratorBundle};
    use crate::basic::ecs::elements::load::{self, LoadBundle, LoadSnapshotReg};
    use crate::basic::ecs::elements::sgen::{self, SGenBundle, SGenSnapShotReg};
    use crate::basic::ecs::elements::shunt::{self, ShuntSnapShotReg};
    use crate::basic::ecs::network::DataOps;
    use crate::{
        basic::ecs::network::PowerGrid,
        io::pandapower::{Network, load_csv_zip},
    };
    fn load_net() -> Network {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        let net = load_csv_zip(&name).unwrap();
        net
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

    #[test]
    fn test_ele_process() {
        let net = load_net();
        let buses: Vec<BusBundle> = net.bus.iter().map(|x| x.into()).collect();
        let ts: Vec<TransformerBundle> = net.trafo.to_bundle_vec();
        let lines: Vec<LineBundle> = net.line.to_bundle_vec();
        let gens: Vec<GeneratorBundle> = net.r#gen.to_bundle_vec();
        let loads: Vec<LoadBundle> = net.load.to_bundle_vec();
        let ext_grid: Vec<ExtGridBundle> = net.ext_grid.to_bundle_vec();
        let shunts: Vec<shunt::ShuntBundle> = net.shunt.to_bundle_vec();
        let sgens: Vec<SGenBundle> = net.sgen.to_bundle_vec();
        let mut pf_net = PowerGrid::default();
        let mut cmd = pf_net.world_mut().commands();

        cmd.spawn_batch(buses);
        pf_net.world_mut().flush();
        let mut bump = Bump::new();
        let mut d = DeferBundleSpawner::new();
        d.spawn_batch(pf_net.world_mut(), ts);
        d.spawn_batch(pf_net.world_mut(), lines);
        d.spawn_batch(pf_net.world_mut(), gens);
        d.spawn_batch(pf_net.world_mut(), loads);
        d.spawn_batch(pf_net.world_mut(), ext_grid);
        d.spawn_batch(pf_net.world_mut(), shunts);
        d.spawn_batch(pf_net.world_mut(), sgens);
      

        let mut registry = SnapshotRegistry::default();
        BusSnapShotReg::register_snap_shot(&mut registry);
        TransSnapShotReg::register_snap_shot(&mut registry);
        GenSnapShotReg::register_snap_shot(&mut registry);
        LineSnapshotReg::register_snap_shot(&mut registry);
        LoadSnapshotReg::register_snap_shot(&mut registry);
        ShuntSnapShotReg::register_snap_shot(&mut registry);
        SGenSnapShotReg::register_snap_shot(&mut registry);

        let world = pf_net.world();
        let a = save_world_manifest(world, &registry).unwrap();
        a.to_file("test_system.toml", None).unwrap();
        let mut world = World::default();
        let b = read_manifest_from_file("test_system.toml", None).unwrap();
        let _b = load_world_manifest(&mut world, &b, &registry).unwrap();
    }
}
