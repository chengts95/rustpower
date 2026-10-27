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

/// Built-in cases or a pandapower CSV ZIP path; shared by screening and retry.
pub fn load_case(case: &str) -> Result<rustpower::io::pandapower::Network, String> {
    match case {
        "39" => serde_json::from_str(rustpower::testcases::case_ieee39::IEEE_39)
            .map_err(|e| e.to_string()),
        "9241" => rustpower::io::pandapower::load_csv_zip(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/cases/pegase9241/data.zip"
        ))
        .map_err(|e| e.to_string()),
        path => rustpower::io::pandapower::load_csv_zip(path).map_err(|e| e.to_string()),
    }
}

/// Device IDs survive World reconstruction; Entity values identify this source World.
pub fn event_record(event: &N1Event) -> serde_json::Value {
    use serde_json::json;
    let hits: Vec<_> = event
        .hits
        .iter()
        .map(|hit| {
            json!({
                "rule_id": hit.rule_id,
                "source_entity": format!("{:?}", hit.source_entity),
                "target": format!("{:?}", hit.target),
                "value": hit.value,
                "result_index": hit.result_index,
            })
        })
        .collect();
    json!({
        "case_id": event.case_id, "outage_index": event.outage_index,
        "outage_kind": format!("{:?}", event.outage.kind), "outage_id": event.outage.id,
        "outage_entity": format!("{:?}", event.outage_entity),
        "stage": format!("{:?}", event.stage), "outcome": format!("{:?}", event.outcome),
        "iterations": event.iterations, "residual": event.residual, "error": event.error,
        "hits": hits, "branches": event.hits.branch_results(),
    })
}

#[cfg(test)]
mod tests {
    use rustpower::{n1::*, prelude::*};
    #[test]
    fn baseline_gate_uses_line_constraints_and_keeps_transformer_outages() {
        let mut net: rustpower::io::pandapower::Network =
            serde_json::from_str(rustpower::testcases::case_ieee39::IEEE_39).unwrap();
        for line in net.line.as_mut().unwrap() {
            line.max_i_ka = Some(1e6);
        }
        let mut grid = PowerGrid::default();
        grid.world_mut().insert_resource(PPNetwork(net));
        grid.init_pf_net();
        // Change only postprocessing ratings after Ybus initialization: every transformer
        // now has an artificial loading violation which the example policy must omit.
        let mut ratings = grid
            .world_mut()
            .query::<&mut ecs::elements::trans::TransformerDevice>();
        for mut device in ratings.iter_mut(grid.world_mut()) {
            device.sn_mva = 1e-9;
        }
        let mut limits = grid
            .world_mut()
            .query::<&mut ecs::elements::bus::VmLimit<ecs::elements::units::PerUnit>>();
        for mut limit in limits.iter_mut(grid.world_mut()) {
            *limit = ecs::elements::bus::VmLimit::new(0.1, 2.0);
        }
        super::install_watches(grid.world_mut());
        for mode in [N1Initialization::BaseAc, N1Initialization::DcPerCase] {
            let p = prepare_n1_with_initialization(grid.world_mut(), mode).unwrap();
            super::check_baseline(&p).unwrap();
            let transformer = p
                .base()
                .catalog()
                .identities()
                .iter()
                .position(|id| id.kind == BranchKind::Transformer)
                .unwrap();
            assert!(p.active_branches().any(|i| i == transformer));
            if let Some(base) = p.base_solution() {
                assert!(
                    p.base()
                        .catalog()
                        .compute_one(transformer, &base.voltage, None)
                        .unwrap()
                        .loading_percent
                        > 100.
                );
            }
            let line = p
                .base()
                .catalog()
                .identities()
                .iter()
                .position(|id| id.kind == BranchKind::Line)
                .unwrap();
            let entity = p.base().catalog().source_entities()[line];
            grid.world_mut().entity_mut(entity).insert(N1Watch {
                rules: vec![WatchRule {
                    rule_id: 10,
                    condition: WatchCondition::BranchLoadingAbove(-1.),
                }],
            });
            let bad = prepare_n1_with_initialization(grid.world_mut(), mode).unwrap();
            let error = super::check_baseline(&bad).unwrap_err();
            assert!(
                error.contains("base violates 1 monitored constraints"),
                "{error}"
            );
            super::install_watches(grid.world_mut());
            let bus =
                grid.world_mut()
                    .query_filtered::<bevy_ecs::entity::Entity, bevy_ecs::query::With<
                        ecs::elements::bus::VmLimit<ecs::elements::units::PerUnit>,
                    >>()
                    .iter(grid.world())
                    .next()
                    .unwrap();
            *grid
                .world_mut()
                .get_mut::<ecs::elements::bus::VmLimit<ecs::elements::units::PerUnit>>(bus)
                .unwrap() = ecs::elements::bus::VmLimit::new(0.0, 0.1);
            let voltage_bad = prepare_n1_with_initialization(grid.world_mut(), mode).unwrap();
            assert!(
                super::check_baseline(&voltage_bad)
                    .unwrap_err()
                    .contains("base violates 1 monitored constraints")
            );
            *grid
                .world_mut()
                .get_mut::<ecs::elements::bus::VmLimit<ecs::elements::units::PerUnit>>(bus)
                .unwrap() = ecs::elements::bus::VmLimit::new(0.1, 2.0);
        }
    }
}
