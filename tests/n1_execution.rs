#![cfg(feature = "experimental_n1")]
use bevy_ecs::prelude::*;
use rustpower::{
    n1::{postprocess::BranchKind, *},
    prelude::{
        ecs::{
            elements::{bus::OutOfService, line::LineID, trans::TransformerID},
            post_processing::{LineResultData, TrafoResultData},
            powerflow::{structure_update::FullRebuildEvent, systems::PowerFlowConfig},
        },
        *,
    },
};
use std::collections::{BTreeMap, BTreeSet};
fn grid() -> PowerGrid {
    let mut net: rustpower::io::pandapower::Network =
        serde_json::from_str(rustpower::testcases::case_ieee39::IEEE_39).unwrap();
    net.trafo.as_mut().unwrap()[3].shift_degree = 7.;
    let mut grid = PowerGrid::default();
    grid.app_mut()
        .add_plugins(rustpower::prelude::ecs::powerflow::structure_update::StructureUpdatePlugin);
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    grid.world_mut().insert_resource(PowerFlowConfig {
        max_it: Some(50),
        tol: Some(1e-10),
    });
    grid.run_pf();
    assert!(grid.world().resource::<PowerFlowResult>().converged);
    grid
}
fn branch_entity(grid: &mut PowerGrid, kind: BranchKind, id: i64) -> Entity {
    match kind {
        BranchKind::Line => {
            grid.world_mut()
                .query::<(Entity, &LineID)>()
                .iter(grid.world())
                .find(|(_, key)| key.0 == id)
                .unwrap()
                .0
        }
        BranchKind::Transformer => {
            grid.world_mut()
                .query::<(Entity, &TransformerID)>()
                .iter(grid.world())
                .find(|(_, key)| key.0 == id)
                .unwrap()
                .0
        }
    }
}
fn index(p: &PreparedN1, kind: BranchKind, id: i64) -> usize {
    p.base()
        .catalog()
        .identities()
        .iter()
        .position(|key| key.kind == kind && key.id == id)
        .unwrap()
}
fn watch(grid: &mut PowerGrid, entity: Entity, condition: WatchCondition) {
    grid.world_mut().entity_mut(entity).insert(N1Watch {
        rules: vec![WatchRule {
            rule_id: 1,
            condition,
        }],
    });
}
#[test]
fn true_outages_match_full_rebuild_and_source_entity_postprocess() {
    let mut original = grid();
    let entities: Vec<_> = original
        .world_mut()
        .query_filtered::<Entity, Or<(With<LineID>, With<TransformerID>)>>()
        .iter(original.world())
        .collect();
    for entity in entities {
        watch(&mut original, entity, WatchCondition::Always);
    }
    let p = prepare_n1(original.world_mut()).unwrap();
    assert!(p.base_solution().unwrap().converged);
    assert_eq!(
        p.base().baseline().ac_start().unwrap(),
        p.base_solution().unwrap().voltage
    );
    let expected_base = &original.world().resource::<PowerFlowResult>().v;
    for (i, &physical) in p.base().baseline().from_perm().iter().enumerate() {
        assert!((p.base_solution().unwrap().voltage[i] - expected_base[physical]).norm() < 1e-9);
    }
    let faults = [
        index(&p, BranchKind::Line, 0),
        index(&p, BranchKind::Transformer, 3),
    ];
    let before = p.base().baseline().y_bus().values().to_vec();
    for workers in [1, 3] {
        let mut output = vec![];
        let summary = run_n1(
            &p,
            N1Config {
                reporting: N1Reporting::Detailed,
                workers,
                batch_size: 1,
                ..Default::default()
            },
            faults,
            |batch| output.extend(batch.events),
        )
        .unwrap();
        assert_eq!(
            (
                summary.submitted,
                summary.processed,
                summary.failed,
                summary.emitted_cases
            ),
            (2, 2, 0, 2)
        );
        assert_eq!(summary.workers_started, workers);
        assert!(summary.dc_time.is_zero());
        assert_eq!(summary.dc_failed, 0);
        assert_eq!(
            output.iter().map(|e| e.case_id).collect::<BTreeSet<_>>(),
            [0, 1].into_iter().collect()
        );
        for event in output {
            assert_eq!(event.stage, SolveStage::Converged);
            assert!(event.residual < 1e-10);
            assert_eq!(event.outage_index, faults[event.case_id]);
            assert_eq!(
                event.outage_entity,
                p.base().catalog().source_entities()[event.outage_index]
            );
            let mut reference = grid();
            let entity = branch_entity(&mut reference, event.outage.kind, event.outage.id);
            reference
                .world_mut()
                .entity_mut(entity)
                .insert(OutOfService);
            reference.world_mut().write_message(FullRebuildEvent);
            reference.run_pf();
            assert!(reference.world().resource::<PowerFlowResult>().converged);
            assert!(
                reference
                    .world()
                    .resource::<ecs::powerflow::structure_update::LastStructureAction>()
                    .full_rebuild
            );
            {
                let base = p.base().baseline();
                let mut engine = N1Worker::new_ac(base);
                let task = outage::OutageTask::for_ac_branch(
                    base,
                    p.base().catalog(),
                    event.outage_index,
                    event.case_id,
                )
                .unwrap();
                outage::WorkerPatchState::default()
                    .apply(&mut engine, base, &task)
                    .unwrap();
                let m = reference
                    .world()
                    .resource::<ecs::powerflow::systems::PowerFlowMat>();
                let mut y_error = 0.0_f64;
                for col in 0..base.y_bus().ncols() {
                    let reference_col = m.to_perm[base.from_perm()[col]];
                    let lo = m.y_bus.col_offsets()[reference_col];
                    let hi = m.y_bus.col_offsets()[reference_col + 1];
                    for slot in base.y_bus().col_offsets()[col]..base.y_bus().col_offsets()[col + 1]
                    {
                        let row = m.to_perm[base.from_perm()[base.y_bus().row_indices()[slot]]];
                        let value = m.y_bus.row_indices()[lo..hi]
                            .binary_search(&row)
                            .map_or(num_complex::Complex64::new(0., 0.), |k| {
                                m.y_bus.values()[lo + k]
                            });
                        y_error = y_error.max((value - engine.y_values[slot]).norm());
                    }
                }
                assert!(y_error < 1e-9, "Ybus patch error {y_error}");
                assert!((0..m.s_bus.len()).all(|i| {
                    (m.s_bus[m.to_perm[base.from_perm()[i]]] - base.s_bus()[i]).norm() < 1e-9
                }));
                let solved = engine.solve_ac_current(base, 50, 1e-10);
                let expected_v = &reference.world().resource::<PowerFlowResult>().v;
                let v_error = base
                    .from_perm()
                    .iter()
                    .enumerate()
                    .map(|(i, &original)| (solved.voltage[i] - expected_v[original]).norm())
                    .fold(0., f64::max);
                assert!(v_error < 1e-8, "reference voltage differs {v_error}");
            }
            reference.post_process();
            let mut expected = BTreeMap::<BranchIdentity, [f64; 7]>::new();
            for (id, r) in reference
                .world_mut()
                .query::<(&LineID, &LineResultData)>()
                .iter(reference.world())
            {
                expected.insert(
                    BranchIdentity {
                        kind: BranchKind::Line,
                        id: id.0,
                    },
                    [
                        r.p_from_mw,
                        r.q_from_mvar,
                        r.p_to_mw,
                        r.q_to_mvar,
                        r.i_from_ka,
                        r.i_to_ka,
                        r.loading_percent,
                    ],
                );
            }
            for (id, r) in reference
                .world_mut()
                .query::<(&TransformerID, &TrafoResultData)>()
                .iter(reference.world())
            {
                expected.insert(
                    BranchIdentity {
                        kind: BranchKind::Transformer,
                        id: id.0,
                    },
                    [
                        r.p_hv_mw,
                        r.q_hv_mvar,
                        r.p_lv_mw,
                        r.q_lv_mvar,
                        r.i_hv_ka,
                        r.i_lv_ka,
                        r.loading_percent,
                    ],
                );
            }
            assert_eq!(event.hits.len(), p.base().catalog().len());
            for (hit_index, hit) in event.hits.iter().enumerate() {
                let (branch_index, identity) = hit.target.branch().unwrap();
                assert_eq!(
                    *hit.source_entity,
                    p.base().catalog().source_entities()[branch_index]
                );
                let result = event.hits.branch_for_hit(hit_index).unwrap();
                let actual = [
                    result.p_from_mw,
                    result.q_from_mvar,
                    result.p_to_mw,
                    result.q_to_mvar,
                    result.i_from_ka,
                    result.i_to_ka,
                    result.loading_percent,
                ];
                for (actual, expected) in actual.into_iter().zip(expected[&identity]) {
                    assert!(
                        (actual - expected).abs() < 1e-6,
                        "{:?} target {:?}: {actual} != {expected}",
                        event.outage,
                        hit.target
                    );
                }
                if branch_index == event.outage_index {
                    assert_eq!(result.i_ka, 0.);
                    assert_eq!(result.p_from_mw, 0.);
                }
            }
        }
    }
    assert_eq!(before, p.base().baseline().y_bus().values());
    assert!(p.base().catalog().in_service().iter().all(|&active| active));
}
#[test]
fn full_case_stream_can_be_silent_and_failure_does_not_stop_next_case() {
    let mut original = grid();
    let p = prepare_n1(original.world_mut()).unwrap();
    let m = p.active_branches().count();
    let summary = run_n1(
        &p,
        N1Config {
            reporting: N1Reporting::Detailed,
            workers: 3,
            batch_size: 8,
            ..Default::default()
        },
        p.active_branches(),
        |_| panic!("no watches must emit no events"),
    )
    .unwrap();
    assert_eq!(
        (summary.submitted, summary.processed, summary.emitted_cases),
        (m, m, 0)
    );
    assert!(summary.dc_time.is_zero());
    assert_eq!(summary.dc_failed, 0);
    assert!(summary.failed > 0); // numerical failures are attempted, no reachability screening
    let target = branch_entity(&mut original, BranchKind::Line, 0);
    watch(&mut original, target, WatchCondition::Always);
    let p = prepare_n1(original.world_mut()).unwrap();
    let faults = [
        index(&p, BranchKind::Transformer, 0),
        index(&p, BranchKind::Line, 0),
    ];
    let mut output = vec![];
    let summary = run_n1(
        &p,
        N1Config {
            reporting: N1Reporting::Detailed,
            workers: 1,
            batch_size: 2,
            ..Default::default()
        },
        faults,
        |batch| output.extend(batch.events),
    )
    .unwrap();
    assert_eq!(
        (summary.processed, summary.failed, summary.emitted_cases),
        (2, 1, 2)
    );
    assert_ne!(output[0].stage, SolveStage::Converged);
    assert!(output[0].hits.branch_for_hit(0).is_none());
    assert_eq!(output[1].stage, SolveStage::Converged);
}
#[test]
fn threshold_rules_emit_only_hits_and_config_is_bounded() {
    let mut original = grid();
    let target = branch_entity(&mut original, BranchKind::Line, 0);
    watch(
        &mut original,
        target,
        WatchCondition::BranchCurrentAbove(0.),
    );
    let p = prepare_n1(original.world_mut()).unwrap();
    let faults = [
        index(&p, BranchKind::Line, 0),
        index(&p, BranchKind::Line, 1),
    ];
    let mut output = vec![];
    let summary = run_n1(
        &p,
        N1Config {
            reporting: N1Reporting::Detailed,
            workers: 2,
            batch_size: 2,
            ..Default::default()
        },
        faults,
        |batch| output.extend(batch.events),
    )
    .unwrap();
    assert_eq!(
        (
            summary.processed,
            summary.failed,
            summary.emitted_cases,
            summary.emitted_hits
        ),
        (2, 0, 1, 1)
    );
    assert_eq!(output[0].case_id, 1);
    assert_eq!(output[0].hits.records().source_entity()[0], target);
    assert!(output[0].hits.records().value()[0].unwrap() > 0.);
    for config in [
        N1Config {
            reporting: N1Reporting::Detailed,
            workers: 0,
            ..Default::default()
        },
        N1Config {
            reporting: N1Reporting::Detailed,
            batch_size: 0,
            ..Default::default()
        },
        N1Config {
            reporting: N1Reporting::Detailed,
            batch_size: 33,
            ..Default::default()
        },
        N1Config {
            reporting: N1Reporting::Detailed,
            max_batch_bytes: 1,
            ..Default::default()
        },
    ] {
        assert!(run_n1(&p, config, [], |_| panic!("invalid config emitted")).is_err());
    }
    let empty = run_n1(&p, N1Config::default(), [], |_| panic!("empty emitted")).unwrap();
    assert_eq!(empty.processed, 0);
    assert!(run_n1(&p, N1Config::default(), [usize::MAX], |_| {}).is_err());
}

#[test]
fn initialization_option_preserves_per_scenario_dcpf() {
    let mut original = grid();
    let target = branch_entity(&mut original, BranchKind::Line, 0);
    watch(&mut original, target, WatchCondition::Always);
    let mut expected: Option<BTreeMap<usize, rustpower::prelude::branch::BranchFlowResult>> = None;
    for initialization in [N1Initialization::BaseAc, N1Initialization::DcPerCase] {
        let prepared =
            prepare_n1_with_initialization(original.world_mut(), initialization).unwrap();
        assert_eq!(prepared.initialization(), initialization);
        assert_eq!(
            prepared.base_solution().is_some(),
            initialization == N1Initialization::BaseAc
        );
        assert_eq!(
            prepared.base().baseline().ac_start().is_some(),
            initialization == N1Initialization::BaseAc
        );
        let faults = [
            index(&prepared, BranchKind::Line, 0),
            index(&prepared, BranchKind::Line, 1),
            index(&prepared, BranchKind::Transformer, 3),
        ];
        let mut results = BTreeMap::new();
        let summary = run_n1(
            &prepared,
            N1Config {
                reporting: N1Reporting::Detailed,
                workers: 2,
                batch_size: 2,
                ..Default::default()
            },
            faults,
            |batch| {
                for event in batch.events {
                    assert_eq!(event.stage, SolveStage::Converged);
                    results.insert(event.case_id, event.hits.branch_for_hit(0).unwrap().clone());
                }
            },
        )
        .unwrap();
        assert_eq!((summary.processed, summary.failed), (3, 0));
        assert_eq!(
            summary.dc_time.is_zero(),
            initialization == N1Initialization::BaseAc
        );
        if let Some(expected) = &expected {
            for (case, actual) in &results {
                let reference = &expected[case];
                for (a, b) in [
                    (actual.p_from_mw, reference.p_from_mw),
                    (actual.q_from_mvar, reference.q_from_mvar),
                    (actual.i_ka, reference.i_ka),
                    (actual.loading_percent, reference.loading_percent),
                ] {
                    assert!((a - b).abs() < 1e-7);
                }
            }
        } else {
            expected = Some(results);
        }
    }
}
