use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::{bundle::Bundle, component::Component};
use serde::{Deserialize, Serialize};

use super::{bus::SnaptShotRegGroup, generator::{TargetBus, TargetPMW, TargetQMVar}};

#[derive(Component, Debug,Serialize,Deserialize, Clone)]
pub struct LoadCfg {
    pub scaling: f64,
    pub load_type: Option<String>,
}
impl Default for LoadCfg {
    fn default() -> Self {
        Self {
            scaling: 1.0,
            load_type: None,
        }
    }
}

#[derive(Component, Debug, Serialize,Deserialize,Clone)]
pub struct LoadModelType {
    pub const_i_percent: f64,
    pub const_z_percent: f64,
}


#[derive(Bundle, Debug, Clone)]
pub struct LoadBundle {
    pub target_bus: TargetBus,
    pub target_p: TargetPMW,
    pub target_q: TargetQMVar,
    pub cfg: LoadCfg,
    pub model: LoadModelType
}

pub struct LoadSnapshotReg;

impl SnaptShotRegGroup for LoadSnapshotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register::<LoadCfg>();
        reg.register::<LoadModelType>();
    }
}