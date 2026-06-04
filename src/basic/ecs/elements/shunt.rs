use crate::io::pandapower::Shunt;
use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::prelude::*;

use super::{
    bus::{OutOfService, SnaptShotRegGroup},
    generator::TargetBus,
};

/// Represents a reactive (and possibly active) shunt device in the network.
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShuntDevice {
    pub p_mw: f64,
    pub q_mvar: f64,
    pub vn_kv: f64,
    pub step: i32,
    pub max_step: i32,
}

use rustpower_proc_marco::DeferBundle;

/// ECS bundle for inserting a shunt device into the simulation.
#[derive(Clone, DeferBundle)]
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

pub mod shunt_systems {
    use crate::basic::ecs::{elements::*, network::GND};
    use bevy_ecs::prelude::Commands;
    use nalgebra::{vector, Complex};
    use super::*;

    fn shunt_internal(item: &ShuntDevice, bus: &TargetBus) -> AdmittanceBranch {
        let s = Complex::new(-item.p_mw, -item.q_mvar) * Complex::new(item.step as f64, 0.0);
        let y = s / (item.vn_kv * item.vn_kv);
        AdmittanceBranch {
            y: Admittance(y),
            port: Port2(vector![bus.0 as i64, GND.into()]),
            v_base: VBase(item.vn_kv),
        }
    }

    pub fn setup_shunt_systems(
        mut commands: Commands,
        q: Query<(&TargetBus, &ShuntDevice), Without<OutOfService>>,
    ) {
        q.iter().for_each(|(target_bus, device)| {
            commands.spawn((EShunt, shunt_internal(device, target_bus)));
        });
    }
}
