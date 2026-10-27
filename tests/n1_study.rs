#![cfg(feature = "experimental_n1")]
use bevy_ecs::prelude::*;
use rustpower::{n1::*, prelude::*};

fn grid() -> PowerGrid {
    let net = serde_json::from_str(rustpower::testcases::case_ieee39::IEEE_39).unwrap();
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    grid
}
fn line_zero(grid: &mut PowerGrid) -> Entity {
    grid.world_mut()
        .query::<(Entity, &ecs::elements::line::LineID)>()
        .iter(grid.world())
        .find(|(_, id)| id.0 == 0)
        .unwrap()
        .0
}

#[test]
fn preparation_binds_specific_faults_and_execute_only_dispatches_them() {
    let mut grid = grid();
    let entity = line_zero(&mut grid);
    let prepared = prepare_n1_for(grid.world_mut(), N1FaultSet::out_of_service([entity])).unwrap();
    assert_eq!(prepared.faults().unwrap().entities(), &[entity]);
    let study = prepared.study().unwrap();
    let mut events = Vec::new();
    let report = study
        .execute(
            N1RunOptions {
                emit_all: true,
                ..Default::default()
            },
            |batch| events.extend(batch.events),
        )
        .unwrap();
    assert_eq!(report.processed(), 1);
    assert_eq!(report.passed(), 1);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].outage_entity, entity);
    assert!(study.all_branch_outages().len() > 1);
}

#[test]
fn replacing_faults_is_atomic_and_an_explicit_empty_set_stays_empty() {
    let mut grid = grid();
    let entity = line_zero(&mut grid);
    let mut prepared =
        prepare_n1_for(grid.world_mut(), N1FaultSet::out_of_service([entity])).unwrap();
    let symbols = prepared.base().baseline().symbolic_ids();
    let unknown = grid.world_mut().spawn_empty().id();
    assert!(
        matches!(prepared.load_faults(N1FaultSet::out_of_service([entity, unknown])), Err(N1StudyError::UnknownBranch(e)) if e == unknown)
    );
    assert_eq!(prepared.faults().unwrap().entities(), &[entity]);
    prepared.load_faults(N1FaultSet::default()).unwrap();
    let report = prepared
        .study()
        .unwrap()
        .execute(N1RunOptions::default(), |_| {
            panic!("empty selection emitted")
        })
        .unwrap();
    assert_eq!(report.processed(), 0);
    assert_eq!(prepared.base().baseline().symbolic_ids(), symbols);
}

#[test]
fn entity_faults_classify_islands_and_return_selected_details_without_watchers() {
    let mut grid = grid();
    let entity = line_zero(&mut grid);
    let prepared = prepare_n1(grid.world_mut()).unwrap();
    let study = prepared.study().unwrap();
    let island_index = prepared
        .active_branches()
        .find(|&i| {
            prepared
                .topology()
                .bridge_cut(i)
                .is_some_and(|c| c.slackless_bus_count() > 0)
        })
        .unwrap();
    let island_entity = prepared.base().catalog().source_entities()[island_index];
    let faults = N1FaultSet::out_of_service([entity, island_entity, entity]);
    assert_eq!(faults.entities(), &[entity, island_entity]);
    assert_eq!(
        study.all_branch_outages().len(),
        prepared.active_branches().count()
    );
    let values = prepared.base().baseline().y_bus().values().to_vec();
    let mut events = Vec::new();
    let report = study
        .details(
            &faults,
            N1RunOptions {
                config: N1Config {
                    workers: 2,
                    batch_size: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
            |batch| events.extend(batch.events),
        )
        .unwrap();
    assert_eq!(
        (
            report.processed(),
            report.passed(),
            report.islanded(),
            report.numerical_failed(),
            report.violated()
        ),
        (2, 1, 1, 0, 0)
    );
    assert_eq!(report.rejected(), 1);
    assert_eq!(report.islands[0].outage_entity, island_entity);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].outage_entity, entity);
    assert_eq!(events[0].outcome, N1Outcome::Passed);
    assert!(events[0].hits.is_empty());
    assert_eq!(prepared.base().baseline().y_bus().values(), values);
    assert!(
        grid.world()
            .get::<ecs::elements::bus::OutOfService>(entity)
            .is_none()
    );
    assert!(
        grid.world()
            .get::<ecs::elements::bus::OutOfService>(island_entity)
            .is_none()
    );
}

#[test]
fn dc_retry_uses_the_same_borrowed_analysis_and_returns_successful_status() {
    let mut grid = grid();
    let entity = line_zero(&mut grid);
    let prepared = prepare_n1(grid.world_mut()).unwrap();
    let study = prepared.study().unwrap();
    let symbols = prepared.base().baseline().symbolic_ids();
    let patterns = prepared.base().baseline().pattern_ids();
    let ac_start = prepared.base().baseline().ac_start().unwrap().to_vec();
    let mut failed = Vec::new();
    let report = study
        .run(
            &N1FaultSet::out_of_service([entity]),
            N1RunOptions {
                config: N1Config {
                    max_iter: 0,
                    ..Default::default()
                },
                ..Default::default()
            },
            |batch| failed.extend(batch.events),
        )
        .unwrap();
    assert_eq!(report.numerical_failed(), 1);
    assert_eq!(failed.len(), 1);
    let retry: N1FaultSet = failed.iter().map(|event| event.outage_entity).collect();
    let mut events = Vec::new();
    let report = study
        .run(
            &retry,
            N1RunOptions {
                initialization: Some(N1Initialization::DcPerCase),
                emit_all: true,
                ..Default::default()
            },
            |batch| events.extend(batch.events),
        )
        .unwrap();
    assert_eq!(
        (
            report.processed(),
            report.passed(),
            report.numerical_failed()
        ),
        (1, 1, 0)
    );
    assert!(!report.solve.dc_time.is_zero());
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].outage_entity, failed[0].outage_entity);
    assert_eq!(events[0].outage_index, failed[0].outage_index);
    assert_eq!(events[0].stage, SolveStage::Converged);
    assert!(events[0].hits.is_empty());
    assert_eq!(prepared.base().baseline().symbolic_ids(), symbols);
    assert_eq!(prepared.base().baseline().pattern_ids(), patterns);
    assert_eq!(prepared.base().baseline().ac_start().unwrap(), ac_start);
    assert_eq!(prepared.initialization(), N1Initialization::BaseAc);
}

#[test]
fn diagnostic_hits_do_not_reject_the_base_and_errors_reference_real_constraints() {
    let mut grid = grid();
    let entity = line_zero(&mut grid);
    grid.world_mut().entity_mut(entity).insert(N1Watch {
        rules: vec![WatchRule {
            rule_id: 7,
            condition: WatchCondition::Always,
        }],
    });
    let prepared = prepare_n1(grid.world_mut()).unwrap();
    prepared.study().unwrap();
    grid.world_mut()
        .get_mut::<N1Watch>(entity)
        .unwrap()
        .rules
        .push(WatchRule {
            rule_id: 8,
            condition: WatchCondition::BranchCurrentAbove(-1.),
        });
    let bad = prepare_n1(grid.world_mut()).unwrap();
    let Err(N1StudyError::Baseline(error)) = bad.study() else {
        panic!("base constraint was accepted")
    };
    assert!(
        error.contains("base violates 1 monitored constraints"),
        "{error}"
    );
    assert!(error.contains("rule 8"), "{error}");
    let hits = bad
        .watch()
        .evaluate(
            bad.base().catalog(),
            &bad.base_solution().unwrap().voltage,
            None,
            true,
        )
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(*hits.first_violation().unwrap().rule_id, 8);
}

#[test]
fn invalid_entities_are_resolved_before_dispatch_and_missing_base_ac_is_explicit() {
    let mut grid = grid();
    let entity = line_zero(&mut grid);
    let prepared =
        prepare_n1_with_initialization(grid.world_mut(), N1Initialization::DcPerCase).unwrap();
    let study = prepared.study().unwrap();
    let unknown = grid.world_mut().spawn_empty().id();
    let error = study
        .run(
            &N1FaultSet::out_of_service([entity, unknown]),
            N1RunOptions::default(),
            |_| panic!("invalid set was partly dispatched"),
        )
        .unwrap_err();
    assert!(matches!(error, N1StudyError::UnknownBranch(e) if e == unknown));
    let error = study
        .run(
            &N1FaultSet::out_of_service([entity]),
            N1RunOptions {
                initialization: Some(N1Initialization::BaseAc),
                ..Default::default()
            },
            |_| panic!("missing base AC dispatched"),
        )
        .unwrap_err();
    assert!(
        matches!(error, N1StudyError::Execution(message) if message.contains("prepared base AC"))
    );
}

#[test]
fn bus_diagnostic_voltage_is_excluded_from_the_first_constraint_error() {
    let mut grid = grid();
    let bus = grid
        .world_mut()
        .query_filtered::<Entity, With<ecs::elements::bus::BusID>>()
        .iter(grid.world())
        .next()
        .unwrap();
    grid.world_mut().entity_mut(bus).insert((
        ecs::elements::bus::VmLimit::<ecs::elements::units::PerUnit>::new(0., 0.1),
        N1Watch {
            rules: vec![
                WatchRule {
                    rule_id: 17,
                    condition: WatchCondition::Always,
                },
                WatchRule {
                    rule_id: 18,
                    condition: WatchCondition::BusVoltageOutsideLimits,
                },
            ],
        },
    ));
    let prepared = prepare_n1(grid.world_mut()).unwrap();
    let hits = prepared
        .watch()
        .evaluate(
            prepared.base().catalog(),
            &prepared.base_solution().unwrap().voltage,
            None,
            true,
        )
        .unwrap();
    assert!(hits.get(0).unwrap().value.is_some());
    assert_eq!(*hits.get(0).unwrap().rule_id, 17);
    assert_eq!(*hits.first_violation().unwrap().rule_id, 18);
    assert!(prepared.check_baseline().unwrap_err().contains("rule 18"));
}
