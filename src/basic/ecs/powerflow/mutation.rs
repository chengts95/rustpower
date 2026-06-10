//! The standardized parameter-mutation pipeline, built on the message bus.
//!
//! Flow:
//!
//! ```text
//! gateway fn (set_load_p, ...)      — synchronous case edit (Target*) for
//!        │                            read-your-writes semantics, then posts
//!        ▼                            a diff instruction to the bus
//! ParamDiff message bus             — the single carrier of dirty information
//!        │
//!        ▼
//! consume_param_diffs (system)      — ordinary parallel system (MessageReader
//!        │                            + Queries, NOT exclusive): applies the
//!        ▼                            real SBusInjPu/VBusPu diffs and fires
//! SBusChangeEvent / VoltageChangeEvent  the coarse change events in ONE place
//!        │
//!        ▼
//! structure_update                  — syncs PowerFlowMat from components
//! ```
//!
//! Sign conventions live only in the gateway functions; `Changed<T>` ticks
//! carry zero correctness responsibility anywhere in this path.
//!
//! Case data vs operating state: the gateway functions are *case edits*
//! (Target* updated + diff posted). Native callers that want *state-only*
//! changes (time series, OPF inner loops) post [`ParamDiff`] messages directly
//! — same bus, same consumer, case untouched.

use bevy_ecs::prelude::*;
use nalgebra::{Complex, SimdComplexField};

use crate::basic::ecs::elements::*;
use crate::basic::ecs::elements::generator::{TargetPMW, TargetQMVar, TargetVmPu};

use super::structure_update::{SBusChangeEvent, VoltageChangeEvent};

/// State-propagation instruction. Produced by the case-edit gateway functions
/// below, or posted directly by native callers for state-only changes.
#[derive(Message, Debug, Clone, Copy)]
pub enum ParamDiff {
    /// Add (dp, dq) MW/MVar to a bus injection (injection sign convention).
    Injection { bus: i64, dp_mw: f64, dq_mvar: f64 },
    /// Set a bus voltage magnitude target (p.u.); the angle is kept.
    VoltageMag { bus: i64, vm_pu: f64 },
}

/// Set a load's active power consumption (MW, positive = consumption).
/// Returns false if the entity is not a load-like element.
pub fn set_load_p(world: &mut World, entity: Entity, p_mw: f64) -> bool {
    let Some(bus) = world.get::<TargetBus>(entity).map(|b| b.0) else { return false; };
    let target = -p_mw; // consumption is a negative injection
    let Some(mut p) = world.get_mut::<TargetPMW>(entity) else { return false; };
    let old = p.0;
    if old == target {
        return true; // no-op filtered at the gateway
    }
    p.0 = target;
    world.write_message(ParamDiff::Injection { bus, dp_mw: target - old, dq_mvar: 0.0 });
    true
}

/// Set a load's reactive power consumption (MVar, positive = consumption).
pub fn set_load_q(world: &mut World, entity: Entity, q_mvar: f64) -> bool {
    let Some(bus) = world.get::<TargetBus>(entity).map(|b| b.0) else { return false; };
    let target = -q_mvar;
    let Some(mut q) = world.get_mut::<TargetQMVar>(entity) else { return false; };
    let old = q.0;
    if old == target {
        return true;
    }
    q.0 = target;
    world.write_message(ParamDiff::Injection { bus, dp_mw: 0.0, dq_mvar: target - old });
    true
}

/// Set a generator's active power production (MW, positive = production).
pub fn set_gen_p(world: &mut World, entity: Entity, p_mw: f64) -> bool {
    let Some(bus) = world.get::<TargetBus>(entity).map(|b| b.0) else { return false; };
    let Some(mut p) = world.get_mut::<TargetPMW>(entity) else { return false; };
    let old = p.0;
    if old == p_mw {
        return true;
    }
    p.0 = p_mw;
    world.write_message(ParamDiff::Injection { bus, dp_mw: p_mw - old, dq_mvar: 0.0 });
    true
}

/// Set a generator's voltage magnitude setpoint (p.u.).
pub fn set_gen_vm(world: &mut World, entity: Entity, vm_pu: f64) -> bool {
    let Some(bus) = world.get::<TargetBus>(entity).map(|b| b.0) else { return false; };
    let Some(mut vm) = world.get_mut::<TargetVmPu>(entity) else { return false; };
    if vm.0 == vm_pu {
        return true;
    }
    vm.0 = vm_pu;
    world.write_message(ParamDiff::VoltageMag { bus, vm_pu });
    true
}

/// Dedicated consumer of the [`ParamDiff`] bus: an ordinary scheduled system
/// (no exclusive World access). Applies the real `SBusInjPu` / `VBusPu` diffs
/// and fires the coarse change events — the single place those events
/// originate for parameter changes.
pub fn consume_param_diffs(
    mut diffs: MessageReader<ParamDiff>,
    lookup: Option<Res<NodeLookup>>,
    common: Option<Res<PFCommonData>>,
    mut sbus: Query<&mut SBusInjPu>,
    mut vbus: Query<&mut VBusPu>,
    mut s_evt: MessageWriter<SBusChangeEvent>,
    mut v_evt: MessageWriter<VoltageChangeEvent>,
) {
    if diffs.is_empty() {
        return;
    }
    let (Some(lookup), Some(common)) = (lookup, common) else {
        diffs.clear();
        return; // world not initialized yet; the pending full rebuild wins
    };
    let sbase_frac = 1.0 / common.sbase;

    let mut s_changed = false;
    let mut v_changed = false;
    for diff in diffs.read() {
        match *diff {
            ParamDiff::Injection { bus, dp_mw, dq_mvar } => {
                let Some(e) = lookup.get_entity(bus) else { continue; };
                if let Ok(mut s) = sbus.get_mut(e) {
                    s.0 += Complex::new(dp_mw, dq_mvar) * sbase_frac;
                    s_changed = true;
                }
            }
            ParamDiff::VoltageMag { bus, vm_pu } => {
                let Some(e) = lookup.get_entity(bus) else { continue; };
                if let Ok(mut v) = vbus.get_mut(e) {
                    v.0 = v.0.simd_signum() * Complex::new(vm_pu, 0.0);
                    v_changed = true;
                }
            }
        }
    }
    if s_changed {
        s_evt.write(SBusChangeEvent);
    }
    if v_changed {
        v_evt.write(VoltageChangeEvent);
    }
}
