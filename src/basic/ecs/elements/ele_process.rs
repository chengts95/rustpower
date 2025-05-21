use super::bus::*;
use super::line::*;

use super::trans::*;

#[cfg(test)]
mod test {
    use std::env;

    use bevy_archive::prelude::{SnapshotRegistry, save_world_manifest};
    use bevy_ecs::world::EntityWorldMut;
    use bumpalo::Bump;

    use super::*;
    use crate::basic::ecs::defer_builder::{DeferBundle, DeferredBundleBuilder};
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
    #[test]
    fn test_ele_process() {
        let net = load_net();
        let buses: Vec<BusBundle> = net.bus.iter().map(|x| x.into()).collect();
        let ts: Vec<TransformerBundle> = net.trafo.unwrap().iter().map(|x| x.into()).collect();
        let mut pf_net = PowerGrid::default();
        let mut cmd = pf_net.world_mut().commands();
        let bump = Bump::new();
        cmd.spawn_batch(buses);
        pf_net.world_mut().flush();

        for i in ts {
            let mut e = pf_net.world_mut().spawn_empty();
            let mut builder = DeferredBundleBuilder::new(&mut e, &bump);
            i.insert_to(&mut builder);
            builder.commit();
        }

        let mut registry = SnapshotRegistry::default();
        BusSnapShotReg::register_snap_shot(&mut registry);
        TransSnapShotReg::register_snap_shot(&mut registry);
        let world = pf_net.world();
        let a = save_world_manifest(world, &registry).unwrap();
        a.to_file("test_system.toml", None).unwrap();
    }
}
