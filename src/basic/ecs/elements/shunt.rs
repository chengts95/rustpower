use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::{bundle::Bundle, component::Component};
use rustpower_proc_marco::DeferBundle;
use serde::{Deserialize, Serialize};
use crate::prelude::ecs::defer_builder::*;
use crate::io::pandapower::Shunt;

use super::{
    bus::{OutOfService, SnaptShotRegGroup},
    generator::{TargetBus, TargetPMW, TargetQMVar, Uncontrollable},
};

#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShuntDevice {
    pub p_mw: f64,
    pub q_mvar: f64,
    pub vn_kv: f64,
    pub step: i32,
    pub max_step: i32,
}
#[derive(DeferBundle,  Clone)]
pub struct ShuntBundle {
    pub target_bus: TargetBus,
    pub device: ShuntDevice,
    pub oos: Option<OutOfService>,
}



impl From<&Shunt> for ShuntBundle {
    fn from(src: &Shunt) -> Self {
        ShuntBundle {
            target_bus: TargetBus(src.bus),
            device: ShuntDevice {
                p_mw: src.p_mw,
                q_mvar: src.q_mvar,
                vn_kv: src.vn_kv,
                step: src.step,
                max_step: src.max_step,
            },
            oos: if src.in_service {None} else {Some(OutOfService)},
        }
    }
}


pub struct ShuntSnapShotReg;

impl SnaptShotRegGroup for ShuntSnapShotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register::<ShuntDevice>();
    }
}

pub mod systems{
    use super::*;
    
}
