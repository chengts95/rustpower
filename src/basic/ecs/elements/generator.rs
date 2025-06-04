//! ECS definitions for generator and external grid control parameters.
//!
//! This module defines the control targets (`TargetXXX`) and metadata components
//! for generator and external grid entities. These components guide the simulation
//! logic (e.g., power flow solver) by specifying voltage setpoints, output limits,
//! and slack behavior.

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

/// Voltage magnitude target in per-unit (pu).
///
/// Default = 1.0 pu if unspecified.
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TargetVmPu(pub f64);
impl Default for TargetVmPu {
    fn default() -> Self {
        Self(1.0)
    }
}
/// Voltage phase angle target in degrees.
///
/// Default = 0.0 deg if unspecified.
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TargetVaDeg(pub f64);
impl Default for TargetVaDeg {
    fn default() -> Self {
        Self(0.0)
    }
}
/// Active/reactive power limits of a generator.
///
/// Used to constrain its dispatch range in simulation.
/// Internally contains:
/// - `p`: Active power range in MW
/// - `q`: Reactive power range in MVAr
///
/// Supports serialization via `PQRangeProxy` for better field access.
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(from = "PQRangeProxy", into = "PQRangeProxy")]
pub struct PQLim {
    pub p: Limit<f64>, // MW
    pub q: Limit<f64>, // MVAr
}
impl PQLim {
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PQRangeProxy {
    p_min: f64,
    p_max: f64,
    q_min: f64,
    q_max: f64,
}
impl From<PQRangeProxy> for PQLim {
    fn from(proxy: PQRangeProxy) -> Self {
        PQLim {
            p: Limit {
                min: proxy.p_min,
                max: proxy.p_max,
            },
            q: Limit {
                min: proxy.q_min,
                max: proxy.q_max,
            },
        }
    }
}

impl From<PQLim> for PQRangeProxy {
    fn from(r: PQLim) -> Self {
        PQRangeProxy {
            p_min: r.p.min,
            p_max: r.p.max,
            q_min: r.q.min,
            q_max: r.q.max,
        }
    }
}
/// Marker for a slack generator (voltage reference node).
///
/// Typically used in external grids or special dispatch models.
#[derive(Component, Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[component(storage = "SparseSet")]
pub struct Slack;

/// Marker for uncontrollable entity (for opf which is not implemented yet).

#[derive(Component, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[component(storage = "SparseSet")]
pub struct Uncontrollable;

/// Generator metadata that affects its control behavior but not calculation directly.
///
/// - `scaling`: Global scaling multiplier applied to its output
/// - `gen_type`: Optional string indicating generation type ("pv", "coal", etc.)
/// - `slack_weight`: Relative contribution in multi-slack dispatch scenarios

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
/// ECS bundle for generator initialization from Pandapower `Gen`.
///
/// This bundle supports optional fields (slack, name, scaling)
/// and can represent both PQ and PV generators.

#[derive(DeferBundle, Debug, Clone)]
pub struct GeneratorBundle {
    pub target_bus: TargetBus,
    pub target_vm: TargetVmPu,
    pub target_p: TargetPMW,
    pub pq_range: PQLim,
    pub cfg: GeneratorCfg,
    pub slack: Option<Slack>,
    pub uncontrollable: Option<Uncontrollable>,
    pub sn_mva: Option<SnMva>,
    pub name: Option<Name>,
}

/// ECS bundle for generator initialization from Pandapower `ExtGrid`.

#[derive(Bundle, DeferBundle)]
pub struct ExtGridBundle {
    pub target_bus: TargetBus,
    pub target_vm: TargetVmPu,
    pub target_va: TargetVaDeg,
    pub cfg: GeneratorCfg, // slack_weight, gen_type, scaling
    pub pq_range: PQLim,   // min/max p/q
    pub slack: Slack,
}

impl From<&Gen> for GeneratorBundle {
    fn from(generator: &Gen) -> Self {
        GeneratorBundle {
            target_bus: TargetBus(generator.bus),
            target_p: TargetPMW(generator.p_mw),
            target_vm: TargetVmPu(generator.vm_pu),
            pq_range: PQLim {
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
            pq_range: PQLim {
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

/// Registers snapshot-compatible generator components for serialization.
///
/// This includes target values (p, q, vm, va), mode flags (slack/uncontrol),
/// and configuration metadata (e.g., `gen_cfg`, `pq_range`).

pub struct GenSnapShotReg;

impl SnaptShotRegGroup for GenSnapShotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register_named::<TargetBus>("target_bus");
        reg.register_named::<TargetVmPu>("vm_pu");
        reg.register_named::<TargetVaDeg>("va_deg");
        reg.register_named::<TargetQMVar>("q_mvar");
        reg.register_named::<TargetPMW>("p_mw");
        reg.register::<Slack>();
        reg.register_named::<GeneratorCfg>("gen_cfg");
        reg.register_named::<Uncontrollable>("uncontrol");
        reg.register::<PQLim>();
    }
}
