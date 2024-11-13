use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::io::pandapower::ecs_net_conv::PandaPowerStartupPlugin;

use super::{network::*, systems::init_states};
#[derive(Debug, SystemSet, Hash, Eq, PartialEq, Clone)]
pub struct PFInitStage;
pub struct BasePFPlugin;

pub struct SwitchPluginTypeA;
pub struct SwitchPluginTypeB;

impl Plugin for BasePFPlugin {
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
pub fn default_app() -> App {
    let mut app = App::new();
    app.add_plugins((PandaPowerStartupPlugin, BasePFPlugin));
    app
}

#[cfg(test)]
mod test {

    use std::env;

    use crate::{basic::new_ecs::{elements::PPNetwork, post_processing::PostProcessing}, io::pandapower::load_csv_zip};

    use super::*;

    #[test]
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
}
