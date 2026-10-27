#![cfg(feature = "experimental_n1")]
use bevy_ecs::prelude::*;
use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use rustpower::{
    n1::*,
    prelude::ecs::{
        elements::{
            BusMatrixIndex, NodeAggRes, PFCommonData,
            bus::{BusID, VmLimit},
            units::PerUnit,
        },
        powerflow::systems::PowerFlowMat,
    },
};
use std::collections::BTreeMap;

fn fixture() -> (World, [Entity; 4], BranchCatalog) {
    let mut world = World::new();
    world.insert_resource(PFCommonData {
        sbase: 100.,
        f_hz: 50.,
        wbase: std::f64::consts::TAU * 50.,
    });
    world.insert_resource(NodeAggRes {
        original_to_merged: vec![0, 0, 1, 2],
        ..Default::default()
    });
    world.insert_resource(PowerFlowMat {
        y_bus: CscMatrix::identity(3),
        s_bus: DVector::zeros(3),
        v_bus_init: DVector::from_element(3, Complex64::new(1., 0.)),
        npq: 2,
        npv: 0,
        to_perm: vec![1, 0, 2],
        from_perm: vec![1, 0, 2],
    });
    let entities = std::array::from_fn(|i| {
        world
            .spawn((
                BusID((i as i64 + 1) * 10),
                BusMatrixIndex(i),
                VmLimit::<PerUnit>::new(
                    if i == 1 { 0.95 } else { 0.9 },
                    if i == 1 { 1.05 } else { 1.1 },
                ),
            ))
            .id()
    });
    let catalog = BranchCatalog::from_world(&mut world).unwrap();
    (world, entities, catalog)
}
fn voltage_watch() -> N1Watch {
    N1Watch {
        rules: vec![WatchRule {
            rule_id: 5,
            condition: WatchCondition::BusVoltageOutsideLimits,
        }],
    }
}
#[test]
fn voltage_limits_use_merged_solver_mapping_and_original_entity_identity() {
    let (mut world, entities, catalog) = fixture();
    for &entity in &entities[..3] {
        world.entity_mut(entity).insert(voltage_watch());
    }
    let plan = WatchPlan::from_world(&mut world, &catalog).unwrap();
    let v = [
        Complex64::new(1.1, 0.),
        Complex64::new(0.94, 0.),
        Complex64::new(1., 0.),
    ];
    let hits = plan.evaluate(&catalog, &v, None, true).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits.violation_count(), 1);
    assert!(plan.is_violated(&catalog, &v, None).unwrap());
    let hit = hits.get(0).unwrap();
    assert_eq!(hit.target.bus(), Some((1, 20)));
    assert_eq!(*hit.source_entity, entities[1]);
    assert_eq!(*hit.value, Some(0.94));
    assert!(hits.branch_for_hit(0).is_none());
    assert!(hits.branch_results().is_empty());
    let changed = [
        Complex64::new(1.12, 0.),
        Complex64::new(0.00003, 0.),
        Complex64::new(1., 0.),
    ];
    let hits = plan.evaluate(&catalog, &changed, None, true).unwrap();
    let values: BTreeMap<_, _> = hits
        .iter()
        .map(|h| (h.target.bus().unwrap().1, h.value.unwrap()))
        .collect();
    assert!(plan.is_violated(&catalog, &changed, None).unwrap());
    assert_eq!(
        values,
        [(10, 0.00003), (20, 0.00003), (30, 1.12)]
            .into_iter()
            .collect()
    );
    *world.get_mut::<VmLimit<PerUnit>>(entities[1]).unwrap() = VmLimit::new(0., 2.);
    assert_eq!(plan.evaluate(&catalog, &v, None, true).unwrap().len(), 1);
    let updated = WatchPlan::from_world(&mut world, &catalog).unwrap();
    assert!(
        updated
            .evaluate(&catalog, &v, None, true)
            .unwrap()
            .is_empty()
    );
}
#[test]
fn voltage_checks_respect_boundaries_and_numeric_failure_and_flag_nonfinite_values() {
    let (mut world, entities, catalog) = fixture();
    world.entity_mut(entities[2]).insert(voltage_watch());
    let plan = WatchPlan::from_world(&mut world, &catalog).unwrap();
    for vm in [0.9, 1.1] {
        let v = [
            Complex64::new(vm, 0.),
            Complex64::new(1., 0.),
            Complex64::new(1., 0.),
        ];
        assert!(plan.evaluate(&catalog, &v, None, true).unwrap().is_empty());
        assert!(!plan.is_violated(&catalog, &v, None).unwrap());
    }
    assert!(
        plan.evaluate(&catalog, &[], None, false)
            .unwrap()
            .is_empty()
    );
    assert!(plan.evaluate(&catalog, &[], None, true).is_err());
    let v = [
        Complex64::new(f64::NAN, 0.),
        Complex64::new(1., 0.),
        Complex64::new(1., 0.),
    ];
    assert_eq!(
        plan.evaluate(&catalog, &v, None, true)
            .unwrap()
            .violation_count(),
        1
    );
    world.entity_mut(entities[2]).insert(N1Watch {
        rules: vec![WatchRule {
            rule_id: 7,
            condition: WatchCondition::SolveFailed,
        }],
    });
    let failure = WatchPlan::from_world(&mut world, &catalog)
        .unwrap()
        .evaluate(&catalog, &[], None, false)
        .unwrap();
    assert_eq!(failure.len(), 1);
    assert_eq!(failure.violation_count(), 0);
    assert!(failure.get(0).unwrap().value.is_none());
}
#[test]
fn invalid_voltage_bounds_and_wrong_rule_targets_return_errors() {
    let (mut world, entities, catalog) = fixture();
    world.entity_mut(entities[0]).insert(voltage_watch());
    for (min, max) in [
        (1.1, 0.9),
        (f64::NAN, 1.1),
        (-0.1, 1.1),
        (0.9, f64::INFINITY),
    ] {
        *world.get_mut::<VmLimit<PerUnit>>(entities[0]).unwrap() = VmLimit::new(min, max);
        assert!(WatchPlan::from_world(&mut world, &catalog).is_err());
    }
    *world.get_mut::<VmLimit<PerUnit>>(entities[0]).unwrap() = VmLimit::new(0.9, 1.1);
    world.entity_mut(entities[0]).insert(N1Watch {
        rules: vec![WatchRule {
            rule_id: 1,
            condition: WatchCondition::BranchLoadingAbove(100.),
        }],
    });
    assert!(WatchPlan::from_world(&mut world, &catalog).is_err());
}

#[test]
fn worker_emits_voltage_violation_separately_from_numerical_failure() {
    use rustpower::prelude::{DataOps, PPNetwork, PowerFlow, PowerGrid};
    let net = serde_json::from_str(rustpower::testcases::case_ieee39::IEEE_39).unwrap();
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    let bus = grid
        .world_mut()
        .query::<(Entity, &BusID)>()
        .iter(grid.world())
        .find(|(_, id)| id.0 == 0)
        .unwrap()
        .0;
    *grid.world_mut().get_mut::<VmLimit<PerUnit>>(bus).unwrap() = VmLimit::new(0., 0.01);
    grid.world_mut().entity_mut(bus).insert(voltage_watch());
    let prepared = prepare_n1(grid.world_mut()).unwrap();
    let branch = prepared
        .base()
        .catalog()
        .identities()
        .iter()
        .position(|id| id.kind == BranchKind::Line && id.id == 0)
        .unwrap();
    let mut events = Vec::new();
    let summary = run_n1(
        &prepared,
        N1Config {
            reporting: N1Reporting::Detailed,
            workers: 2,
            batch_size: 1,
            ..Default::default()
        },
        [branch],
        |batch| events.extend(batch.events),
    )
    .unwrap();
    assert_eq!(
        (
            summary.processed,
            summary.failed,
            summary.violated_cases,
            summary.violation_hits
        ),
        (1, 0, 1, 1)
    );
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].stage, SolveStage::Converged);
    let hit = events[0].hits.get(0).unwrap();
    assert_eq!(*hit.source_entity, bus);
    assert_eq!(hit.target.bus().unwrap().1, 0);
    assert!(hit.value.unwrap() > 0.01);
    assert!(events[0].hits.branch_results().is_empty());
}
