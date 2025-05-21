use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::{component::Component, name::Name};
use rustpower_proc_marco::DeferBundle;
use serde::{Deserialize, Serialize};

use crate::{basic::ecs::defer_builder::*, io::pandapower::Load};

use super::{bus::SnaptShotRegGroup, generator::*};

#[derive(Component, Debug, Serialize, Deserialize, Clone)]
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

#[derive(Component, Debug, Serialize, Deserialize, Clone)]
pub struct LoadModelType {
    pub const_i_percent: f64,
    pub const_z_percent: f64,
}

#[derive(DeferBundle, Debug, Clone)]
pub struct LoadBundle {
    pub target_bus: TargetBus,
    pub target_p: TargetPMW,
    pub target_q: TargetQMVar,
    pub cfg: LoadCfg,
    pub model: LoadModelType,
    pub uncontrollable: Option<Uncontrollable>,
    pub name: Option<Name>,
    pub sn_mva: Option<SnMva>,
}

impl From<&Load> for LoadBundle {
    fn from(load: &Load) -> Self {
        Self {
            target_bus: TargetBus(load.bus),
            target_p: TargetPMW(load.p_mw),
            target_q: TargetQMVar(load.q_mvar),
            cfg: LoadCfg {
                scaling: load.scaling,
                load_type: load.type_.clone(),
            },
            model: LoadModelType {
                const_i_percent: load.const_i_percent,
                const_z_percent: load.const_z_percent,
            },
            uncontrollable: (!load.controllable.unwrap_or(true)).then_some(Uncontrollable),
            name: load.name.clone().map(Name::new),
            sn_mva: load.sn_mva.map(SnMva),
        }
    }
}
pub struct LoadSnapshotReg;

impl SnaptShotRegGroup for LoadSnapshotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register::<LoadCfg>();
        reg.register::<LoadModelType>();
    }
}
