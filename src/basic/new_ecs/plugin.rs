use bevy_app::{Plugin, Startup};
use bevy_ecs::schedule::{IntoSystemConfigs, SystemSet};

use crate::io::pandapower::ecs_net_conv::init_pf;

pub struct BasePFPlugin;

impl Plugin for BasePFPlugin {
    fn build(&self, app: &mut bevy_app::App) {
     
    }
}
