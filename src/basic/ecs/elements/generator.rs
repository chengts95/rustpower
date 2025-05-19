use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::prelude::*;
use derive_more::{ From};

use std::marker::PhantomData;
use bevy_ecs::name::Name;
use crate::{io::pandapower::Gen};

use super::{bus::SnaptShotRegGroup, units::*};

#[derive(Component, Debug, Clone)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct TargetBus(pub i64);

#[derive(Component, Debug, Clone)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct TargetP(pub f64);
/// PU电压目标
#[derive(Component, Debug, Clone)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct TargetVm(pub Pair<f64, PerUnit>);
impl Default for TargetVm {
    fn default() -> Self {
        Self(Pair(1.0, PhantomData::default()))
    }
}

/// 发电机的有功/无功出力限值
#[derive(Component, Debug, Clone)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct PQRange {
    pub p: Limit<f64>, // MW
    pub q: Limit<f64>, // MVAr
}
impl PQRange {
    pub fn new(pmin: f64, pmax: f64, qmin: f64, qmax: f64) -> Self {
        Self {
            p: Limit {
                min: pmin,
                max: pmax,
            },
            q: Limit {
                min: qmin,
                max: qmax,
            },
        }
    }
}
/// 是否为平衡节点
#[derive(Component, Debug, Clone, Copy)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
#[component(storage = "SparseSet")]
pub struct Slack;

/// 不可控发电机标记
#[derive(Component, Debug, Default)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
#[component(storage = "SparseSet")]
pub struct Uncontrollable;

/// 发电机元信息（不参与计算）
#[derive(Component, Debug, Clone)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct GeneratorCfg {
    pub scaling: f64,
    pub r#gen_type: Option<String>,
    pub slack_weight: f64,
}
impl Default for GeneratorCfg {
    fn default() -> Self {
        Self {
            scaling: 1.0,
            r#gen_type: None,
            slack_weight: 1.0,
        }
    }
}

#[derive(Bundle, Debug, Clone)]
pub struct GeneratorBundle {
    pub target_bus: TargetBus,
    pub target_vm: TargetVm,
    pub target_p: TargetP,
    pub pq_range: PQRange,
    pub cfg: GeneratorCfg,
}
#[derive(Default)]
pub struct GeneratorFlags {
    pub slack: bool,
    pub uncontrollable: bool,
}
impl From<&Gen> for (GeneratorBundle, GeneratorFlags) {
    fn from(generator: &Gen) -> Self {
        let bundle = GeneratorBundle {
            target_bus: TargetBus(generator.bus),
            target_p: TargetP(generator.p_mw),
            target_vm: TargetVm(Pair(generator.vm_pu, PhantomData)),
            pq_range: PQRange {
                p: Limit {
                    min: generator.min_p_mw,
                    max: generator.max_p_mw,
                },
                q: Limit {
                    min: generator.min_q_mvar,
                    max: generator.max_q_mvar,
                },
            },
            cfg: GeneratorCfg {
                scaling: generator.scaling,
                r#gen_type: generator.type_.clone(),
                slack_weight: generator.slack_weight,
            },
        };
        let flags = GeneratorFlags {
            slack: generator.slack,
            uncontrollable: generator.controllable == Some(false),
        };
        (bundle, flags)
    }
}
/// Represents an external grid in the network.
#[derive(Default, Debug, Clone)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
struct ExtGrid {
    pub bus: i64,
    pub in_service: bool,
    pub va_degree: f64,
    pub vm_pu: f64,
    pub max_p_mw: Option<f64>,
    pub min_p_mw: Option<f64>,
    pub max_q_mvar: Option<f64>,
    pub min_q_mvar: Option<f64>,
    pub slack_weight: f64,
    pub name: Option<String>,
}

pub struct GenSnapShotReg;

impl SnaptShotRegGroup for GenSnapShotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register::<TargetBus>();
        reg.register::<TargetVm>();
        reg.register::<TargetP>();
        reg.register::<Slack>();
        reg.register::<GeneratorCfg>();
        reg.register::<Uncontrollable>();
        reg.register::<PQRange>();
    }
}
