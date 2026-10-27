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

mod reporting {
    use bevy_ecs::prelude::*;
    use rustpower::{
        n1::*,
        prelude::{ecs::elements::line::LineID, *},
    };
    use std::collections::BTreeMap;
    fn grid() -> PowerGrid {
        let net = serde_json::from_str(rustpower::testcases::case_ieee39::IEEE_39).unwrap();
        let mut g = PowerGrid::default();
        g.world_mut().insert_resource(PPNetwork(net));
        g.init_pf_net();
        g
    }
    fn line(g: &mut PowerGrid, id: i64) -> Entity {
        g.world_mut()
            .query::<(Entity, &LineID)>()
            .iter(g.world())
            .find(|(_, key)| key.0 == id)
            .unwrap()
            .0
    }
    fn index(p: &PreparedN1, kind: BranchKind, id: i64) -> usize {
        p.base()
            .catalog()
            .identities()
            .iter()
            .position(|key| key.kind == kind && key.id == id)
            .unwrap()
    }
    #[test]
    fn status_and_details_agree_and_selected_cases_can_be_rerun_with_details() {
        let mut g = grid();
        let target = line(&mut g, 0);
        g.world_mut().entity_mut(target).insert(N1Watch {
            rules: vec![
                WatchRule {
                    rule_id: 1,
                    condition: WatchCondition::BranchCurrentAbove(0.),
                },
                WatchRule {
                    rule_id: 2,
                    condition: WatchCondition::SolveFailed,
                },
            ],
        });
        let p = prepare_n1(g.world_mut()).unwrap();
        let faults = [
            index(&p, BranchKind::Line, 0),
            index(&p, BranchKind::Line, 1),
            index(&p, BranchKind::Transformer, 0),
        ];
        let mut reference = None;
        let mut rejected = None;
        for reporting in [N1Reporting::Detailed, N1Reporting::StatusOnly] {
            let mut events = vec![];
            let s = run_n1(
                &p,
                N1Config {
                    workers: 2,
                    batch_size: 2,
                    reporting,
                    ..Default::default()
                },
                faults,
                |b| events.extend(b.events),
            )
            .unwrap();
            assert_eq!(
                (
                    s.processed,
                    s.failed,
                    s.violated_cases,
                    s.passed(),
                    s.rejected()
                ),
                (3, 1, 1, 1, 2)
            );
            let outcomes: BTreeMap<_, _> =
                events.iter().map(|e| (e.outage_index, e.outcome)).collect();
            assert_eq!(outcomes[&faults[1]], N1Outcome::ConstraintViolation);
            assert_eq!(outcomes[&faults[2]], N1Outcome::NumericalFailure);
            if let Some(expected) = &reference {
                assert_eq!(&outcomes, expected);
            } else {
                reference = Some(outcomes);
            }
            if reporting == N1Reporting::StatusOnly {
                assert_eq!((s.emitted_hits, s.violation_hits), (0, 0));
                for event in events {
                    assert!(event.hits.is_empty());
                    assert_eq!(event.hits.capacity(), 0);
                    assert!(event.hits.branch_results().is_empty());
                }
                rejected = Some(faults[1]);
            }
        }
        let mut details = vec![];
        run_n1(
            &p,
            N1Config {
                reporting: N1Reporting::Detailed,
                ..Default::default()
            },
            [rejected.unwrap()],
            |b| details.extend(b.events),
        )
        .unwrap();
        assert_eq!(details.len(), 1);
        assert_eq!(details[0].outage_index, faults[1]);
        assert_eq!(details[0].outcome, N1Outcome::ConstraintViolation);
        assert!(details[0].hits.branch_for_hit(0).unwrap().i_ka > 0.);
    }
    #[test]
    fn status_mode_ignores_diagnostic_always_and_reports_unwatched_solve_failure() {
        let mut g = grid();
        let target = line(&mut g, 0);
        g.world_mut().entity_mut(target).insert(N1Watch {
            rules: (0..1024)
                .map(|rule_id| WatchRule {
                    rule_id,
                    condition: WatchCondition::Always,
                })
                .collect(),
        });
        let p = prepare_n1(g.world_mut()).unwrap();
        let fault = index(&p, BranchKind::Line, 0);
        let s = run_n1(
            &p,
            N1Config {
                max_batch_bytes: 4096,
                batch_size: 1,
                ..Default::default()
            },
            [fault],
            |_| panic!("passing diagnostic generated status event"),
        )
        .unwrap();
        assert_eq!((s.passed(), s.rejected()), (1, 0));
        assert!(
            run_n1(
                &p,
                N1Config {
                    reporting: N1Reporting::Detailed,
                    max_batch_bytes: 4096,
                    batch_size: 1,
                    ..Default::default()
                },
                [fault],
                |_| panic!("oversized details emitted")
            )
            .is_err()
        );
        g.world_mut().entity_mut(target).remove::<N1Watch>();
        let p = prepare_n1(g.world_mut()).unwrap();
        let failed = index(&p, BranchKind::Transformer, 0);
        let mut events = vec![];
        let s = run_n1(
            &p,
            N1Config {
                ..Default::default()
            },
            [failed],
            |b| events.extend(b.events),
        )
        .unwrap();
        assert_eq!(s.rejected(), 1);
        assert_eq!(events[0].outcome, N1Outcome::NumericalFailure);
    }
    #[test]
    fn squared_branch_predicates_match_detailed_line_and_transformer_rules() {
        let mut g = grid();
        let p = prepare_n1(g.world_mut()).unwrap();
        for (kind, id) in [(BranchKind::Line, 0), (BranchKind::Transformer, 0)] {
            let idx = index(&p, kind, id);
            let entity = p.base().catalog().source_entities()[idx];
            let v = &p.base_solution().unwrap().voltage;
            let flow = p.base().catalog().compute_one(idx, v, None).unwrap();
            for condition in [
                WatchCondition::BranchCurrentAbove(flow.i_ka * 0.9),
                WatchCondition::BranchCurrentAbove(flow.i_ka * 1.1),
                WatchCondition::BranchLoadingAbove(flow.loading_percent * 0.9),
                WatchCondition::BranchLoadingAbove(flow.loading_percent * 1.1),
                WatchCondition::BranchCurrentAbove(-1.),
            ] {
                g.world_mut().entity_mut(entity).insert(N1Watch {
                    rules: vec![WatchRule {
                        rule_id: 1,
                        condition,
                    }],
                });
                let plan = WatchPlan::from_world(g.world_mut(), p.base().catalog()).unwrap();
                for outage in [None, Some(idx)] {
                    let details = plan.evaluate(p.base().catalog(), v, outage, true).unwrap();
                    assert_eq!(
                        plan.is_violated(p.base().catalog(), v, outage).unwrap(),
                        details.violation_count() > 0,
                        "{kind:?} {condition:?} {outage:?}"
                    );
                }
            }
            g.world_mut().entity_mut(entity).remove::<N1Watch>();
        }
    }
}

mod symbolic_sharing {
    use rustpower::prelude::{
        dcpf::DcpfModel, ecs::powerflow::systems::PowerFlowMat, jacobian_cache::JacobianCache, *,
    };
    use rustpower_sol_klu::KLUSolver;

    struct NumericWorker(KLUSolver);
    impl Drop for NumericWorker {
        fn drop(&mut self) {
            // Borrowed Symbolic belongs to the parent. Legacy Drop remains responsible
            // for this worker's own Numeric/Common; the parent frees Symbolic once.
            self.0.symbolic = std::ptr::null_mut();
        }
    }
    fn analyze(p: &[usize], i: &[usize]) -> KLUSolver {
        let mut owner = KLUSolver::default();
        // KLU ABI uses int64_t*: prepare owned input buffers before publication.
        let mut p: Vec<i64> = p.iter().map(|&v| v as i64).collect();
        let mut i: Vec<i64> = i.iter().map(|&v| v as i64).collect();
        unsafe {
            assert_eq!(
                owner.solve_sym(p.as_mut_ptr(), i.as_mut_ptr(), p.len() as i64 - 1),
                0
            );
        }
        assert!(!owner.symbolic.is_null());
        owner
    }
    fn exercise(owner: &KLUSolver, p: &[i64], i: &[i64], worker: usize) {
        let n = p.len() - 1;
        let mut numeric = NumericWorker(KLUSolver::default());
        numeric.0.symbolic = owner.symbolic;
        let mut values = vec![0.; i.len()];
        let mut rhs = vec![0.; n];
        for job in 0..24 {
            rhs.fill(0.);
            // Use both actual PF sparsity patterns with nonsingular test values;
            // this isolates FFI sharing from PF convergence and numerical physics.
            for col in 0..n {
                for slot in p[col] as usize..p[col + 1] as usize {
                    let row = i[slot] as usize;
                    values[slot] = if row == col {
                        (n + 1) as f64 + worker as f64 + job as f64 * 0.1
                    } else {
                        -0.25
                    };
                    rhs[row] += values[slot]; // A * ones
                }
            }
            unsafe {
                // System klu.h documents these CSC/Symbolic inputs as unmodified.
                // Cast only at FFI; shared Rust slices are never made mutable refs.
                let status = if job == 0 {
                    numeric.0.factor(
                        p.as_ptr() as *mut i64,
                        i.as_ptr() as *mut i64,
                        values.as_mut_ptr(),
                    )
                } else {
                    numeric.0.refactor(
                        p.as_ptr() as *mut i64,
                        i.as_ptr() as *mut i64,
                        values.as_mut_ptr(),
                        n as i64,
                    )
                };
                assert_eq!(status, 0);
                assert!(!numeric.0.numeric.is_null());
                assert_eq!(numeric.0.symbolic, owner.symbolic);
                assert_eq!(numeric.0.solve(rhs.as_mut_ptr(), n as i64, 1), 0);
            }
            assert!(rhs.iter().all(|&v| (v - 1.).abs() < 1e-10));
        }
    }
    #[test]
    fn four_workers_borrow_the_same_dc_and_ac_symbolics_without_copying() {
        let net = serde_json::from_str::<rustpower::io::pandapower::Network>(
            rustpower::testcases::case_ieee39::IEEE_39,
        )
        .unwrap();
        let mut grid = PowerGrid::default();
        grid.world_mut().insert_resource(PPNetwork(net));
        grid.init_pf_net();
        let mat = grid.world().resource::<PowerFlowMat>();
        let ac = JacobianCache::build_from_permuted(
            mat.y_bus.col_offsets(),
            mat.y_bus.row_indices(),
            mat.npv,
            mat.npq,
        );
        let dc = DcpfModel::from_ybus(
            &mat.y_bus,
            &mat.v_bus_init,
            mat.npv + mat.npq,
            &mat.to_perm,
            &vec![0.; mat.y_bus.nrows()],
        );
        let ac_owner = analyze(&ac.j_col_ptrs, &ac.j_row_indices);
        let dc_owner = analyze(&dc.col_ptrs, &dc.row_indices);
        // One immutable pattern per matrix, shared by all workers. No per-worker clone.
        let ac_p: Vec<i64> = ac.j_col_ptrs.iter().map(|&x| x as i64).collect();
        let ac_i: Vec<i64> = ac.j_row_indices.iter().map(|&x| x as i64).collect();
        let dc_p: Vec<i64> = dc.col_ptrs.iter().map(|&x| x as i64).collect();
        let dc_i: Vec<i64> = dc.row_indices.iter().map(|&x| x as i64).collect();
        std::thread::scope(|scope| {
            for worker in 0..4 {
                let (a, d, ap, ai, dp, di) = (&ac_owner, &dc_owner, &ac_p, &ac_i, &dc_p, &dc_i);
                scope.spawn(move || {
                    exercise(d, dp, di, worker);
                    exercise(a, ap, ai, worker);
                });
            }
        });
        // Parent retains ownership until every worker and its Numeric have dropped.
    }
}
