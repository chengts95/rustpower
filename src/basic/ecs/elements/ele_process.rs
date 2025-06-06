//! # Power System Element Plugin
//!
//! This module coordinates the setup of core electrical elements in the ECS world,
//! including buses, transformers, generators, shunts, lines, etc.
//!
//! It also provides snapshot registration logic for all system components,
//! enabling serialization and restoration of ECS state.
//!
//! Main exports:
//! - `ElementSetupPlugin`: registers ECS systems for initializing simulation elements
//! - `DefaultSnapShotReg`: handles snapshot component registration
//! - `build_snapshot_registry`: builds the global registry for ECS world snapshots

use super::switch;
use crate::basic::ecs::elements::*;
use crate::basic::ecs::plugin::BeforePFInitStage;
use crate::prelude::ecs::network::SolverStage::BeforeSolve;

use bevy_app::Startup;
use bevy_app::Update;
use bevy_archive::prelude::SnapshotRegistry;

// Re-export all element modules for unified access
pub use bus::*;
pub use generator::*;
pub use line::*;
pub use load::*;
pub use sgen::*;
pub use shunt::*;
pub use switch::*;
pub use trans::*;
pub use units::*;

/// A snapshot registration group that aggregates all power system element snapshot registries.
///
/// This struct provides a unified interface for registering all ECS component serializers
/// used in saving and loading power system states.
pub struct DefaultSnapShotReg;

impl SnaptShotRegGroup for DefaultSnapShotReg {
    /// Registers the snapshot serializers for all supported element types.
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

/// A Bevy plugin that sets up ECS systems for power grid elements.
///
/// - Initializes bus lookup tables.
/// - Sets up transformers, shunts, and lines during startup.
/// - Registers dynamic bus update logic during solver stage.
#[derive(Default)]
pub struct ElementSetupPlugin;

impl bevy_app::Plugin for ElementSetupPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_systems(
            Startup,
            (
                bus::systems::init_node_lookup.in_set(BeforePFInitStage),
                (
                    trans::systems::setup_transformer,
                    line::systems::setup_line_systems,
                    shunt::systems::setup_shunt_systems,
                ),
            )
                .chain()
                .in_set(BeforePFInitStage),
        );

        app.add_systems(Update, bus::systems::update_node_lookup.in_set(BeforeSolve));
    }
}

/// Builds and returns a snapshot registry that includes all power system element serializers.
///
/// This is the main entry point for preparing snapshot support in a simulation environment.
pub fn build_snapshot_registry() -> SnapshotRegistry {
    let mut registry = SnapshotRegistry::default();
    DefaultSnapShotReg::register_snap_shot(&mut registry);
    registry
}

#[cfg(test)]
mod test {
    use crate::{
        basic::ecs::network::{DataOps, PowerGrid},
        prelude::pandapower::Network,
    };
    use bevy_archive::prelude::{
        load_world_manifest, read_manifest_from_file, save_world_manifest,
    };
    use std::env;

    use crate::io::pandapower::{ecs_net_conv::LoadPandapowerNet, load_csv_zip};

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
