use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::prelude::*;
use derive_more::From;
use rustpower_proc_marco::DeferBundle;

use crate::io::pandapower::Line;

use super::bus::{OutOfService, SnaptShotRegGroup};
use crate::prelude::ecs::defer_builder::*;
use bevy_ecs::name::Name;

#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FromBus(pub i64);

#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]

pub struct ToBus(pub i64);

#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LineParams {
    pub r_ohm_per_km: f64,
    pub x_ohm_per_km: f64,
    pub g_us_per_km: f64,
    pub c_nf_per_km: f64,
    pub length_km: f64,
    pub df: f64, // dielectric factor?
    pub parallel: i32,
}

#[derive(Clone, DeferBundle)]
pub struct LineBundle {
    pub from: FromBus,
    pub to: ToBus,
    pub params: LineParams,
    pub name: Option<Name>,
    pub std_spec: Option<StandardModelType>,
    pub out: Option<OutOfService>,
}
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StandardModelType(pub String); // std_type only
pub struct LineSnapShotReg;
impl From<&Line> for LineBundle {
    fn from(line: &Line) -> Self {
        Self {
            from: FromBus(line.from_bus),
            to: ToBus(line.to_bus),
            params: LineParams {
                r_ohm_per_km: line.r_ohm_per_km,
                x_ohm_per_km: line.x_ohm_per_km,
                g_us_per_km: line.g_us_per_km,
                c_nf_per_km: line.c_nf_per_km,
                df: line.df,
                length_km: line.length_km,
                parallel: line.parallel,
            },
            name: line.name.clone().map(Name::new),
            std_spec: line.std_type.clone().map(StandardModelType),
            out: (!line.in_service).then_some(OutOfService),
        }
    }
}

pub struct LineSnapshotReg;

impl SnaptShotRegGroup for LineSnapshotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register::<FromBus>();
        reg.register::<ToBus>();
        reg.register::<LineParams>();
        reg.register::<StandardModelType>();
    }
}
