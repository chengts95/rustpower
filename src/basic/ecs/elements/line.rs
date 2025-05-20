use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::prelude::*;
use derive_more::From;


use super::bus::{OutOfService, SnaptShotRegGroup};
use crate::io::pandapower::Line;
use bevy_ecs::name::Name;


#[derive(Component, Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct FromBus(pub i64);

#[derive(Component, Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize)]

pub struct ToBus(pub i64);

#[derive(Component, Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct LineParams {
    pub r_ohm_per_km: f64,
    pub x_ohm_per_km: f64,
    pub g_us_per_km: f64,
    pub c_nf_per_km: f64,
    pub length_km: f64,
    pub df: f64, // dielectric factor?
    pub parallel: i32,
}
#[derive(Bundle)]
pub struct LineBundle {
    pub from: FromBus,
    pub to: ToBus,
    pub params: LineParams,
}

#[derive(Bundle)]
pub struct LineMeta {
    pub name: Name,
    pub std_spec: StandardModelType,
    pub out: OutOfService,
}
#[derive(Component, Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct StandardModelType(pub String); // std_type only
pub struct LineSnapShotReg;
impl From<&Line> for LineBundle {
    fn from(line: &Line) -> Self {
        // let meta = LineMeta {
        //     name: Name::new(line.name.clone().unwrap_or_else(|| format!("line_{}_{}", line.from_bus, line.to_bus))),
        //     std_spec: line.std_type.clone().map(StandardLineSpec),
        //     out: (!line.in_service).then_some(OutOfService),
        // };
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