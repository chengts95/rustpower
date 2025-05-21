use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::prelude::*;
use derive_more::From;
use rustpower_proc_marco::DeferBundle;

use crate::{
    basic::ecs::defer_builder::*,
    io::pandapower::{ExtGrid, Gen},
};

use super::{bus::SnaptShotRegGroup, units::*};

#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnMva(pub f64);

#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TargetBus(pub i64);

#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TargetPMW(pub f64);

#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TargetQMVar(pub f64);
/// PU电压目标
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TargetVmPu(f64);
impl Default for TargetVmPu {
    fn default() -> Self {
        Self(1.0)
    }
}
/// PU电压目标
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TargetVaDeg(f64);
impl Default for TargetVaDeg {
    fn default() -> Self {
        Self(0.0)
    }
}
/// 发电机的有功/无功出力限值
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Component, Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[component(storage = "SparseSet")]
pub struct Slack;

/// 不可控标记
#[derive(Component, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[component(storage = "SparseSet")]
pub struct Uncontrollable;

/// 发电机元信息（不参与计算）
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
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

#[derive(DeferBundle, Debug, Clone)]
pub struct GeneratorBundle {
    pub target_bus: TargetBus,
    pub target_vm: TargetVmPu,
    pub target_p: TargetPMW,
    pub pq_range: PQRange,
    pub cfg: GeneratorCfg,
    pub slack: Option<Slack>,
    pub uncontrollable: Option<Uncontrollable>,
    pub sn_mva: Option<SnMva>,
    pub name: Option<Name>,
}

/// 可以重用 Generator 架构
#[derive(Bundle, DeferBundle)]
pub struct ExtGridBundle {
    pub target_bus: TargetBus,
    pub target_vm: TargetVmPu,
    pub target_va: TargetVaDeg,
    pub cfg: GeneratorCfg, // slack_weight, gen_type, scaling
    pub pq_range: PQRange, // min/max p/q
    pub slack: Slack,
}

impl From<&Gen> for GeneratorBundle {
    fn from(generator: &Gen) -> Self {
        GeneratorBundle {
            target_bus: TargetBus(generator.bus),
            target_p: TargetPMW(generator.p_mw),
            target_vm: TargetVmPu(generator.vm_pu),
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

            slack: generator.slack.then_some(Slack),
            uncontrollable: (!generator.controllable.unwrap_or(true)).then_some(Uncontrollable),

            sn_mva: generator.sn_mva.map(SnMva),
            name: generator.name.clone().map(Name::new),
        }
    }
}

impl From<&ExtGrid> for ExtGridBundle {
    fn from(ext_grid: &ExtGrid) -> Self {
        ExtGridBundle {
            target_bus: TargetBus(ext_grid.bus),
            target_vm: TargetVmPu(ext_grid.vm_pu),
            target_va: TargetVaDeg(ext_grid.va_degree),
            cfg: GeneratorCfg {
                scaling: 1.0,
                r#gen_type: None,
                slack_weight: ext_grid.slack_weight,
            },
            pq_range: PQRange {
                p: Limit {
                    min: ext_grid.min_p_mw.unwrap_or(0.0),
                    max: ext_grid.max_p_mw.unwrap_or(f64::MAX),
                },
                q: Limit {
                    min: ext_grid.min_q_mvar.unwrap_or(0.0),
                    max: ext_grid.max_q_mvar.unwrap_or(f64::MAX),
                },
            },
            slack: Slack,
        }
    }
}
pub struct GenSnapShotReg;

impl SnaptShotRegGroup for GenSnapShotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register_named::<TargetBus>("target_bus");
        reg.register_named::<TargetVmPu>("vm_pu");
        reg.register_named::<TargetQMVar>("q_mvar");
        reg.register_named::<TargetPMW>("p_mw");
        reg.register::<Slack>();
        reg.register_named::<GeneratorCfg>("gen_cfg");
        reg.register_named::<Uncontrollable>("uncontrol");
        reg.register::<PQRange>();
    }
}
