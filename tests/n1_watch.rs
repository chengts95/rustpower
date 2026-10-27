#![cfg(feature = "experimental_n1")]
use bevy_ecs::prelude::*;
use nalgebra::{DVector, Matrix2};
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use rustpower::{
    n1::{
        postprocess::BranchCatalog,
        watch::{N1Watch, WatchCondition, WatchPlan, WatchRule},
    },
    prelude::ecs::{
        elements::{
            BusMatrixIndex, FromBusMatrixIndex, PFCommonData, ToBusMatrixIndex,
            bus::VNominal,
            line::{LineID, LineParams},
            trans::Port4MatPatch,
            units::Pair,
        },
        powerflow::systems::PowerFlowMat,
    },
};
fn fixture() -> (World, Entity, Entity) {
    let mut world = World::new();
    world.insert_resource(PFCommonData {
        sbase: 100.,
        f_hz: 50.,
        wbase: std::f64::consts::TAU * 50.,
    });
    let mut bus0 = None;
    for index in 0..2 {
        let entity = world
            .spawn((
                BusMatrixIndex(index),
                VNominal(Pair(110., Default::default())),
            ))
            .id();
        if index == 0 {
            bus0 = Some(entity);
        }
    }
    world.insert_resource(PowerFlowMat {
        y_bus: CscMatrix::try_from_csc_data(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![Complex64::new(0., 0.); 4],
        )
        .unwrap(),
        s_bus: DVector::zeros(2),
        v_bus_init: DVector::from_element(2, Complex64::new(1., 0.)),
        npq: 1,
        npv: 0,
        to_perm: vec![0, 1],
        from_perm: vec![0, 1],
    });
    let y = Complex64::new(1., -3.);
    let line = world
        .spawn((
            LineID(900),
            LineParams {
                r_ohm_per_km: 0.1,
                x_ohm_per_km: 0.2,
                g_us_per_km: 0.,
                c_nf_per_km: 0.,
                length_km: 1.,
                df: 1.,
                parallel: 1,
                max_i_ka: 1.,
            },
            FromBusMatrixIndex(0),
            ToBusMatrixIndex(1),
            Port4MatPatch(Matrix2::new(y, -y, -y, y)),
        ))
        .id();
    (world, line, bus0.unwrap())
}
fn watch(rules: &[(u64, WatchCondition)]) -> N1Watch {
    N1Watch {
        rules: rules
            .iter()
            .map(|&(rule_id, condition)| WatchRule { rule_id, condition })
            .collect(),
    }
}
#[test]
fn entity_rules_sparse_hits_outage_mask_and_frozen_plan() {
    let (mut world, line, _) = fixture();
    world.entity_mut(line).insert(watch(&[
        (1, WatchCondition::BranchLoadingAbove(1.)),
        (2, WatchCondition::BranchCurrentAbove(0.)),
        (3, WatchCondition::BranchLoadingAbove(1e20)),
        (4, WatchCondition::SolveFailed),
    ]));
    let catalog = BranchCatalog::from_world(&mut world).unwrap();
    assert_eq!(catalog.source_entities(), [line]);
    assert_eq!(catalog.phase_injections(), [0.]);
    let plan = WatchPlan::from_world(&mut world, &catalog).unwrap();
    assert_eq!(plan.rule_count(), 4);
    let v = [Complex64::new(1.02, 0.01), Complex64::new(0.97, -0.05)];
    let hits = plan.evaluate(&catalog, &v, None, true).unwrap();
    assert_eq!(hits.iter().map(|h| *h.rule_id).collect::<Vec<_>>(), [1, 2]);
    assert!(hits.iter().all(|h| {
        *h.source_entity == line
            && h.target
                .branch()
                .is_some_and(|(index, identity)| index == 0 && identity.id == 900)
            && h.result_index.is_some()
    }));
    assert!(
        plan.evaluate(&catalog, &v, Some(0), true)
            .unwrap()
            .is_empty()
    );
    assert!(catalog.in_service()[0]);
    let failure = plan.evaluate(&catalog, &[], None, false).unwrap();
    assert_eq!(failure.len(), 1);
    assert_eq!(failure.records().rule_id()[0], 4);
    assert!(failure.branch_for_hit(0).is_none());
    world.entity_mut(line).insert(N1Watch::default());
    assert_eq!(plan.evaluate(&catalog, &v, None, true).unwrap().len(), 2);
    let empty = WatchPlan::from_world(&mut world, &catalog).unwrap();
    assert!(empty.is_empty());
    assert!(
        empty
            .evaluate(&catalog, &[], None, true)
            .unwrap()
            .is_empty()
    );
}
#[test]
fn explicit_always_trigger_and_failed_numerics_are_separate() {
    let (mut world, line, _) = fixture();
    world.entity_mut(line).insert(watch(&[
        (5, WatchCondition::Always),
        (6, WatchCondition::BranchCurrentAbove(0.)),
    ]));
    let catalog = BranchCatalog::from_world(&mut world).unwrap();
    let plan = WatchPlan::from_world(&mut world, &catalog).unwrap();
    let hits = plan.evaluate(&catalog, &[], None, false).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits.records().rule_id()[0], 5);
    assert!(hits.branch_for_hit(0).is_none());
    let zero = plan
        .evaluate(&catalog, &[Complex64::new(1., 0.); 2], Some(0), true)
        .unwrap();
    assert_eq!(zero.len(), 1);
    assert_eq!(zero.branch_for_hit(0).unwrap().i_ka, 0.);
}
#[test]
fn unsupported_targets_bad_thresholds_and_mixed_catalogs_report_errors() {
    let (mut world, line, bus) = fixture();
    let catalog = BranchCatalog::from_world(&mut world).unwrap();
    world
        .entity_mut(bus)
        .insert(watch(&[(1, WatchCondition::Always)]));
    assert!(WatchPlan::from_world(&mut world, &catalog).is_err());
    world.entity_mut(bus).remove::<N1Watch>();
    world
        .entity_mut(line)
        .insert(watch(&[(1, WatchCondition::BranchCurrentAbove(f64::NAN))]));
    assert!(WatchPlan::from_world(&mut world, &catalog).is_err());
    world.entity_mut(line).insert(watch(&[
        (1, WatchCondition::Always),
        (1, WatchCondition::SolveFailed),
    ]));
    assert!(WatchPlan::from_world(&mut world, &catalog).is_err());
    world
        .entity_mut(line)
        .insert(watch(&[(1, WatchCondition::Always)]));
    let plan = WatchPlan::from_world(&mut world, &catalog).unwrap();
    let other = BranchCatalog::from_world(&mut world).unwrap();
    assert!(plan.evaluate(&other, &[], None, false).is_err());
    let (mut another, _, _) = fixture();
    assert!(WatchPlan::from_world(&mut another, &catalog).is_err());
}

#[test]
fn sparse_outputs_do_not_reserve_all_rules_and_dense_outputs_remain_complete() {
    let (mut world, line, _) = fixture();
    let rules = (0..2048)
        .map(|rule_id| WatchRule {
            rule_id,
            condition: if rule_id == 0 {
                WatchCondition::Always
            } else {
                WatchCondition::BranchCurrentAbove(f64::MAX)
            },
        })
        .collect();
    world.entity_mut(line).insert(N1Watch { rules });
    let catalog = BranchCatalog::from_world(&mut world).unwrap();
    let plan = WatchPlan::from_world(&mut world, &catalog).unwrap();
    let v = [Complex64::new(1.02, 0.01), Complex64::new(0.97, -0.05)];
    let sparse = plan.evaluate(&catalog, &v, None, true).unwrap();
    assert_eq!(sparse.len(), 1);
    assert!(
        sparse.capacity() <= 8,
        "sparse result reserved {} entries",
        sparse.capacity()
    );
    world.entity_mut(line).insert(N1Watch {
        rules: (0..2048)
            .map(|rule_id| WatchRule {
                rule_id,
                condition: WatchCondition::Always,
            })
            .collect(),
    });
    let plan = WatchPlan::from_world(&mut world, &catalog).unwrap();
    let dense = plan.evaluate(&catalog, &v, None, true).unwrap();
    assert_eq!(dense.len(), 2048);
    assert_eq!(dense.branch_results().len(), 1);
    assert!(
        dense
            .records()
            .result_index()
            .iter()
            .all(|&index| index == Some(0))
    );
    assert!(std::mem::size_of::<rustpower::n1::WatchHit>() < 176);
    assert_eq!(
        dense
            .iter()
            .map(|h| *h.rule_id)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        2048
    );
}

#[test]
fn hits_share_one_result_per_branch_and_keep_scenarios_independent() {
    let (mut world, first, _) = fixture();
    let params = world.get::<LineParams>(first).unwrap().clone();
    let patch = world.get::<Port4MatPatch>(first).unwrap().0;
    let second = world
        .spawn((
            LineID(901),
            params,
            FromBusMatrixIndex(0),
            ToBusMatrixIndex(1),
            Port4MatPatch(patch),
        ))
        .id();
    for entity in [first, second] {
        world.entity_mut(entity).insert(watch(&[
            (1, WatchCondition::Always),
            (2, WatchCondition::BranchCurrentAbove(-1.)),
        ]));
    }
    let catalog = BranchCatalog::from_world(&mut world).unwrap();
    let plan = WatchPlan::from_world(&mut world, &catalog).unwrap();
    let v = [Complex64::new(1.02, 0.01), Complex64::new(0.97, -0.05)];
    let normal = plan.evaluate(&catalog, &v, None, true).unwrap();
    assert_eq!(normal.len(), 4);
    assert_eq!(normal.branch_results().len(), 2);
    assert_eq!(
        normal.records().result_index(),
        &[Some(0), Some(0), Some(1), Some(1)]
    );
    assert!(std::ptr::eq(
        normal.branch_for_hit(0).unwrap(),
        normal.branch_for_hit(1).unwrap()
    ));
    assert!(!std::ptr::eq(
        normal.branch_for_hit(0).unwrap(),
        normal.branch_for_hit(2).unwrap()
    ));
    assert_eq!(
        normal.records().source_entity(),
        &[first, first, second, second]
    );
    assert!(normal.branch_for_hit(99).is_none());
    let outage = plan.evaluate(&catalog, &v, Some(0), true).unwrap();
    assert_eq!(outage.branch_for_hit(0).unwrap().i_ka, 0.);
    assert!(normal.branch_for_hit(0).unwrap().i_ka > 0.);
    let failed = plan.evaluate(&catalog, &[], None, false).unwrap();
    assert_eq!(failed.len(), 2);
    assert!(failed.branch_results().is_empty());
    assert!(failed.records().result_index().iter().all(Option::is_none));
}

mod voltage {
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
}
