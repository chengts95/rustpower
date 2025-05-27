use crate::io::pandapower::SGen;
use crate::prelude::ecs::defer_builder::*;
use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::{component::Component, name::Name};
use rustpower_proc_marco::DeferBundle;

use super::{
    bus::SnaptShotRegGroup,
    generator::{TargetBus, Uncontrollable},
};

#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SGenDevice {
    pub p_mw: f64,
    pub q_mvar: f64,
    pub scaling: f64,
    pub sn_mva: Option<f64>,
    pub gen_type: Option<String>,
    pub is_current_source: bool,
}

#[derive(DeferBundle, Debug, Clone)]
pub struct SGenBundle {
    pub target_bus: TargetBus,
    pub device: SGenDevice,
    pub uncontrollable: Option<Uncontrollable>,
    pub name: Option<Name>,
}

impl From<&SGen> for SGenBundle {
    fn from(sgen: &SGen) -> Self {
        let bundle = SGenBundle {
            target_bus: TargetBus(sgen.bus),
            device: SGenDevice {
                p_mw: sgen.p_mw,
                q_mvar: sgen.q_mvar,
                scaling: sgen.scaling,
                sn_mva: sgen.sn_mva,
                gen_type: sgen.type_.clone(),
                is_current_source: sgen.current_source,
            },
            uncontrollable: (!sgen.controllable.unwrap_or(true)).then_some(Uncontrollable),
            name: sgen.name.clone().map(Name::new),
        };

        bundle
    }
}

pub struct SGenSnapShotReg;

impl SnaptShotRegGroup for SGenSnapShotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register::<SGenDevice>();
    }
}
