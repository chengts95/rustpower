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
