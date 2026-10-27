//! Shared monitoring policy for the N-1 examples.
use bevy_ecs::prelude::*;
use rustpower::{
    n1::*,
    prelude::ecs::elements::{
        bus::{BusID, OutOfService, VmLimit},
        line::LineID,
        trans::TransformerID,
        units::PerUnit,
    },
};

/// Monitor line loading and emit one status event per failed solve.
/// Transformers remain contingency candidates, with their loading rule disabled.
pub fn install_watches(world: &mut World) {
    let devices: Vec<_> = world
        .query_filtered::<(Entity, Has<LineID>), Or<(With<LineID>, With<TransformerID>)>>()
        .iter(world)
        .collect();
    for (index, (entity, line)) in devices.into_iter().enumerate() {
        let mut rules = Vec::new();
        if line {
            rules.push(WatchRule {
                rule_id: 1,
                condition: WatchCondition::BranchLoadingAbove(100.),
            });
        }
        if index == 0 {
            rules.push(WatchRule {
                rule_id: 2,
                condition: WatchCondition::SolveFailed,
            });
        }
        if rules.is_empty() {
            world.entity_mut(entity).remove::<N1Watch>();
        } else {
            world.entity_mut(entity).insert(N1Watch { rules });
        }
    }
    let buses: Vec<_> = world
        .query_filtered::<Entity, (With<BusID>, With<VmLimit<PerUnit>>, Without<OutOfService>)>()
        .iter(world)
        .collect();
    for entity in buses {
        world.entity_mut(entity).insert(N1Watch {
            rules: vec![WatchRule {
                rule_id: 3,
                condition: WatchCondition::BusVoltageOutsideLimits,
            }],
        });
    }
}

/// Require the unchanged base to satisfy the example's configured constraints.
/// Per-case DC mode performs this validation solve once, keeping worker starts unchanged.
pub fn check_baseline(prepared: &PreparedN1) -> Result<(), String> {
    prepared.check_baseline()
}
