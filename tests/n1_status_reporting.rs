#![cfg(feature = "experimental_n1")]
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
        let outcomes: BTreeMap<_, _> = events.iter().map(|e| (e.outage_index, e.outcome)).collect();
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
