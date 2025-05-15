use bevy_app::prelude::*;
use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::prelude::*;

use crate::io::pandapower::ecs_net_conv::PandaPowerStartupPlugin;

use super::{network::*, switch::*, systems::init_states};

/// Represents the power flow initialization stage for Bevy's ECS system.
#[derive(Debug, SystemSet, Hash, Eq, PartialEq, Clone)]
pub struct PFInitStage;

/// Base plugin for initializing power flow calculations.
///
/// This plugin sets up the basic configuration required for power flow computation,
/// and registers necessary systems to run during the application startup and update phases.
pub struct BasePFPlugin;

/// Plugin for initializing switch handling using Type A logic.
///
/// SwitchPluginTypeA includes node aggregation logic that results in merging nodes in the network.
/// This plugin is used when merging nodes is required, typically involving matrix operations
/// to aggregate nodes for power flow calculation purposes.
pub struct SwitchPluginTypeA;

/// Plugin for initializing switch handling using Type B logic.
///
/// SwitchPluginTypeB does not involve node merging and instead focuses on directly processing
/// switch admittance without the need for complex aggregation operations. This is used when
/// performance optimization is required, as node merging and matrix operations are not necessary.
pub struct SwitchPluginTypeB;

impl Plugin for BasePFPlugin {
    /// Builds the base power flow plugin by setting up essential resources and systems.
    ///
    /// Adds startup systems such as state initialization and permutation application,
    /// and registers the main power flow run system for the update phase.
    fn build(&self, app: &mut bevy_app::App) {
        app.world_mut().insert_resource(PowerFlowConfig {
            max_it: None,
            tol: None,
        });
        app.add_systems(
            Startup,
            (
                init_states.run_if(not(resource_exists::<PowerFlowMat>)),
                apply_permutation,
            )
                .chain()
                .in_set(PFInitStage),
        );

        app.add_systems(Update, ecs_run_pf);
    }
}

impl Plugin for SwitchPluginTypeA {
    /// Builds the plugin for handling switches using Type A logic.
    ///
    /// Sets up the systems for processing switch states, aggregating nodes, and handling node merges.
    /// This plugin is suitable for scenarios where node merging and matrix operations are necessary.
    fn build(&self, app: &mut bevy_app::App) {
        app.world_mut().insert_resource(PowerFlowConfig {
            max_it: None,
            tol: None,
        });
        app.add_systems(
            Startup,
            (process_switch_state)
                .chain()
                .before(init_states)
                .in_set(PFInitStage),
        );
        app.add_systems(
            Startup,
            (node_aggregation_system.pipe(handle_node_merge))
                .chain()
                .after(init_states)
                .before(apply_permutation)
                .in_set(PFInitStage),
        );
    }
}

impl Plugin for SwitchPluginTypeB {
    /// Builds the plugin for handling switches using Type B logic.
    ///
    /// Sets up the systems for processing switch states without node aggregation. This approach is more
    /// performant for networks where node merging is not necessary, as it avoids complex matrix operations.
    fn build(&self, app: &mut bevy_app::App) {
        app.world_mut().insert_resource(PowerFlowConfig {
            max_it: None,
            tol: None,
        });
        app.add_systems(
            Startup,
            (process_switch_state_admit)
                .before(init_states)
                .in_set(PFInitStage),
        );
    }
}
#[cfg_attr(feature = "archive")]
pub struct ArchivePlugin;
#[cfg_attr(feature = "archive")]
impl Plugin for ArchivePlugin {
    fn build(&self, app: &mut App) {
        use crate::prelude::ecs::elements::*;
        let mut reg = SnapshotRegistry::default();

        reg.register::<Admittance>();
        reg.register::<Port2>();
        reg.register::<VBase>();

        app.insert_resource(reg);
    }
}

/// Creates a default Bevy application with the base power flow plugin.
///
/// This function returns an instance of a Bevy `App` with the `PandaPowerStartupPlugin`
/// and `BasePFPlugin` already added. Additional plugins such as `SwitchPluginTypeA` or
/// `SwitchPluginTypeB` can be added based on the use case.
pub fn default_app() -> App {
    let mut app = App::new();
    app.add_plugins((PandaPowerStartupPlugin, BasePFPlugin));
    #[cfg(feature = "archive")]
    app.add_plugins(ArchivePlugin);

    app
}

#[cfg(test)]
mod test {

    use std::{env, fs};

    use serde_json::{Map, Value};

    use crate::{
        basic::ecs::{elements::PPNetwork, post_processing::PostProcessing},
        io::pandapower::{load_csv_zip, load_pandapower_json_obj},
    };

    use super::*;

    #[test]
    /// Tests the initialization of the power flow application.
    ///
    /// This test checks whether the power flow application can be initialized correctly,
    /// and runs the update and post-processing steps to ensure proper execution.
    fn test_pf_init() {
        let mut app = default_app();
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        let net = load_csv_zip(&name).unwrap();

        app.world_mut().insert_resource(PPNetwork(net));

        app.update();
        app.post_process();
        app.print_res_bus();
    }

    /// Loads a JSON object from a string.
    ///
    /// This helper function takes a JSON string and parses it into a `Map<String, Value>`.
    fn load_json_from_str(file_content: &str) -> Result<Map<String, Value>, std::io::Error> {
        let parsed: Value = serde_json::from_str(&file_content)?;
        let obj: Map<String, Value> = parsed.as_object().unwrap().clone();
        Ok(obj)
    }

    /// Loads a JSON object from a file.
    ///
    /// This helper function reads a JSON file from the specified path and parses it into a `Map<String, Value>`.
    fn load_json(file_path: &str) -> Result<Map<String, Value>, std::io::Error> {
        let file_content = fs::read_to_string(file_path).expect("Error reading network file");
        let obj = load_json_from_str(&file_content);
        obj
    }

    #[test]
    /// Tests the power flow calculation using SwitchPluginTypeA.
    ///
    /// This test initializes the power flow application with `SwitchPluginTypeA`, which involves
    /// node merging and aggregation, then runs the power flow calculation and checks the results.
    fn test_ecs_pf_switch_a() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/test/", dir);
        let name = folder.to_owned() + "/new_input_PFLV_modified.json";
        let json = load_json(&name).unwrap();
        let json: Map<String, Value> = json
            .get("pp_network")
            .and_then(|v| v.as_object())
            .unwrap()
            .clone();
        let net = load_pandapower_json_obj(&json);
        let mut pf_net = default_app();
        pf_net.add_plugins(SwitchPluginTypeA);
        pf_net.world_mut().insert_resource(PPNetwork(net));
        pf_net.update();

        pf_net.post_process();
        pf_net.print_res_bus();
    }

    #[test]
    /// Tests the power flow calculation using SwitchPluginTypeB.
    ///
    /// This test initializes the power flow application with `SwitchPluginTypeB`, which skips
    /// node merging and directly processes switch admittance, then runs the power flow calculation
    /// and checks the results.
    fn test_ecs_pf_switch_b() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/test/", dir);
        let name = folder.to_owned() + "/new_input_PFLV_modified.json";
        let json = load_json(&name).unwrap();
        let json: Map<String, Value> = json
            .get("pp_network")
            .and_then(|v| v.as_object())
            .unwrap()
            .clone();
        let net = load_pandapower_json_obj(&json);
        let mut pf_net = default_app();
        pf_net.add_plugins(SwitchPluginTypeB);
        pf_net.world_mut().insert_resource(PPNetwork(net));
        pf_net.update();

        pf_net.post_process();
        pf_net.print_res_bus();
    }
}
