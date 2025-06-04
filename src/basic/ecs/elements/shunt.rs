use crate::io::pandapower::Shunt;
use crate::prelude::ecs::defer_builder::*;
use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::component::Component;
use rustpower_proc_marco::DeferBundle;

use super::{
    bus::{OutOfService, SnaptShotRegGroup},
    generator::TargetBus,
};

/// Represents a reactive (and possibly active) shunt device in the network.
///
/// Shunt devices inject or absorb reactive power (Q) and may also
/// consume active power (P) for internal losses or compensation.
/// Unlike generators, they are not dispatchable and thus do **not**
/// use `TargetPMW` or `TargetQMVar`, but are modeled as **fixed admittance**.
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShuntDevice {
    /// Active power loss in MW
    pub p_mw: f64,
    /// Reactive power in MVAr (usually negative for capacitive devices)
    pub q_mvar: f64,
    /// Nominal voltage level of the shunt terminal in kV
    pub vn_kv: f64,
    /// Current tap step (for tap-changing devices)
    pub step: i32,
    /// Maximum allowed tap steps
    pub max_step: i32,
}

/// ECS bundle for inserting a shunt device into the simulation.
///
/// Includes target bus reference, fixed parameter device model,
/// and optional out-of-service flag.
#[derive(DeferBundle, Clone)]
pub struct ShuntBundle {
    /// Target bus entity ID (as i64) for the shunt connection
    pub target_bus: TargetBus,
    /// Static device data (Q/P/voltage level)
    pub device: ShuntDevice,
    /// Optional marker for being disconnected from the network
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
            oos: if src.in_service {
                None
            } else {
                Some(OutOfService)
            },
        }
    }
}

pub struct ShuntSnapShotReg;

impl SnaptShotRegGroup for ShuntSnapShotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register::<ShuntDevice>();
    }
}

pub mod systems {

    use crate::basic::ecs::{elements::*, network::GND};
    use bevy_ecs::prelude::Commands;
    use nalgebra::vector;
    /// Converts a `ShuntDevice` into an equivalent 2-port admittance branch.
    ///
    /// This treats the shunt as a constant S = P + jQ load,
    /// and transforms it into Y = S / V² form. The resulting branch
    /// connects `target_bus` to the ground (GND).
    ///
    /// Note: Since shunts are passive, the direction of power is *injected*
    /// from the network into the shunt, thus the `S` is negated here.
    ///
    /// 💡 This system allows shunt devices to be modeled as embedded admittance
    /// branches in the power flow solution.
    fn shunt_internal(item: &ShuntDevice, bus: &TargetBus) -> AdmittanceBranch {
        let s = Complex::new(-item.p_mw, -item.q_mvar) * Complex::new(item.step as f64, 0.0);
        let y = s / (item.vn_kv * item.vn_kv);
        AdmittanceBranch {
            y: Admittance(y),
            port: Port2(vector![bus.0 as i64, GND.into()]),
            v_base: VBase(item.vn_kv),
        }
    }
    /// System for spawning shunt admittance branches into the simulation.
    ///
    /// Filters out all shunt devices marked `OutOfService`,
    /// then for each remaining `ShuntDevice`, calculates its
    /// equivalent admittance and adds it as an `EShunt` entity.
    pub fn setup_shunt_systems(
        mut commands: Commands,
        q: Query<(&TargetBus, &ShuntDevice), Without<OutOfService>>,
    ) {
        q.iter().for_each(|(target_bus, device)| {
            commands.spawn((EShunt, shunt_internal(device, target_bus)));
        });
    }
}
