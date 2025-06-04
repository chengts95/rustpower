use crate::io::pandapower::SGen;
use crate::prelude::ecs::defer_builder::*;
use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::{component::Component, name::Name};
use rustpower_proc_marco::DeferBundle;

use super::{
    TargetPMW, TargetQMVar,
    bus::SnaptShotRegGroup,
    generator::{TargetBus, Uncontrollable},
};

/// Static Generator (SGen) device parameters, describing a fixed-power injection.
///
/// In contrast to regular controllable generators (`Gen`),
/// SGen represents fixed or externally controlled power sources,
/// such as PV inverters, battery inverters, or static diesel units.
///
/// This component describes physical attributes only. Control logic,
/// if any, is encoded via `TargetPMW`/`TargetQMVar`, but dispatchability
/// is gated by the presence (or absence) of `Uncontrollable`.
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SGenDevice {
    /// Fixed active power injection (MW)
    pub p_mw: f64,
    /// Fixed reactive power injection (MVAr)
    pub q_mvar: f64,
    /// Per-unit scaling factor (usually 1.0)
    pub scaling: f64,
    /// Optional rated apparent power (S_base) in MVA
    pub sn_mva: Option<f64>,
    /// Optional generator type string (e.g., "pv", "wind")
    pub gen_type: Option<String>,
    /// Whether this source is modeled as a current injection instead of a voltage-controlled source
    pub is_current_source: bool,
}

/// ECS bundle for inserting a static generator (SGen) into the simulation.
///
/// Includes fixed parameters, power target components, and controllability marker.
///
/// 💡 Important design note:
/// While SGen always carries `TargetPMW` / `TargetQMVar` components,
/// the presence of `Uncontrollable` marks it as **not dispatchable**.
/// This means the targets will be used **as-is**, not updated by optimization or control systems.
#[derive(DeferBundle, Debug, Clone)]
pub struct SGenBundle {
    /// Target bus to which the SGen is connected
    pub target_bus: TargetBus,
    /// Static device definition
    pub device: SGenDevice,
    /// Target active power (used during power flow initialization)
    pub target_p: TargetPMW,
    /// Target reactive power (used during power flow initialization)
    pub target_q: TargetQMVar,
    /// Optional uncontrollable flag
    pub uncontrollable: Option<Uncontrollable>,
    /// Optional name (for debugging or display)
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
            target_p: TargetPMW(sgen.p_mw),
            target_q: TargetQMVar(sgen.q_mvar),
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
