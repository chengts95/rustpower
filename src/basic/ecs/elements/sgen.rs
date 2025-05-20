use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::{bundle::Bundle, component::Component};

use crate::io::pandapower::SGen;

use super::{bus::SnaptShotRegGroup, generator::TargetBus};

#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SGenDevice {
    pub p_mw: f64,
    pub q_mvar: f64,
    pub scaling: f64,
    pub sn_mva: Option<f64>,
    pub gen_type: Option<String>,
    pub is_current_source: bool,
}

#[derive(Bundle, Debug, Clone)]
pub struct SGenBundle {
    pub target_bus: TargetBus,
    pub device: SGenDevice,
}

#[derive(Default)]
pub struct SGenFlags {
    pub uncontrollable: bool,
}

impl From<&SGen> for (SGenBundle, SGenFlags) {
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
        };
        let flags = SGenFlags {
            uncontrollable: sgen.controllable == Some(false),
        };
        (bundle, flags)
    }
}

pub struct SGenSnapShotReg;

impl SnaptShotRegGroup for SGenSnapShotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register::<SGenDevice>();
    }
}
