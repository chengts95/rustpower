#![cfg(feature = "experimental_n1")]
use bevy_ecs::prelude::*;
use rustpower::{
    n1::{
        BaseBatch, BaseTask, BranchKind, N1Worker, PreparedBase, SolveStage, prepare_base,
        run_base_batches,
    },
    prelude::{
        ecs::{
            elements::{bus::OutOfService, line::LineID, trans::TransformerID},
            post_processing::{LineResultData, TrafoResultData},
            powerflow::structure_update::FullRebuildEvent,
        },
        *,
    },
};
use std::collections::{BTreeMap, BTreeSet};

fn grid(shift: bool) -> PowerGrid {
    let mut net: rustpower::io::pandapower::Network =
        serde_json::from_str(rustpower::testcases::case_ieee39::IEEE_39).unwrap();
    if shift {
        net.trafo.as_mut().unwrap()[0].shift_degree = 7.;
    }
    let mut grid = PowerGrid::default();
    grid.app_mut()
        .add_plugins(rustpower::prelude::ecs::powerflow::structure_update::StructureUpdatePlugin);
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    grid.run_pf();
    assert!(grid.world().resource::<PowerFlowResult>().converged);
    grid.post_process();
    grid
}
fn compare_voltage(
    prepared: &PreparedBase,
    actual: &[num_complex::Complex64],
    expected: &nalgebra::DVector<num_complex::Complex64>,
) {
    for (k, &original) in prepared.baseline().from_perm().iter().enumerate() {
        assert!((actual[k] - expected[original]).norm() < 1e-8);
    }
}
#[test]
fn real_base_apps_share_analysis_and_match_voltage_and_postprocess() {
    let mut reference = grid(true);
    let expected = reference.world().resource::<PowerFlowResult>().v.clone();
    let lines: BTreeMap<_, _> = reference
        .world_mut()
        .query::<(&LineID, &LineResultData)>()
        .iter(reference.world())
        .map(|(id, r)| (id.0, *r))
        .collect();
    let trafos: BTreeMap<_, _> = reference
        .world_mut()
        .query::<(&TransformerID, &TrafoResultData)>()
        .iter(reference.world())
        .map(|(id, r)| (id.0, *r))
        .collect();
    let prepared = prepare_base(reference.world_mut()).unwrap();
    let symbolics = prepared.baseline().symbolic_ids();
    let patterns = prepared.baseline().pattern_ids();
    let before_y = prepared.baseline().y_bus().values().to_vec();
    let mut output = vec![];
    let batches = (0..12).map(|i| {
        BaseBatch::new(
            (0..3)
                .map(|j| BaseTask {
                    case_id: i * 3 + j,
                    max_iter: 50,
                    tolerance: 1e-10,
                    emit: true,
                })
                .collect(),
        )
        .unwrap()
    });
    let summary = run_base_batches(&prepared, 3, batches, |r| output.push(r)).unwrap();
    assert_eq!(
        (summary.processed, summary.failed, summary.emitted),
        (36, 0, 36)
    );
    assert_eq!(
        output.iter().map(|r| r.case_id).collect::<BTreeSet<_>>(),
        (0..36).collect()
    );
    assert_eq!(summary.workers_started, 3);
    assert!(output.iter().all(|r| r.worker_id < 3));
    for result in &output {
        assert_eq!(result.symbolic_ids, symbolics);
        assert_eq!(result.pattern_ids, patterns);
        assert_eq!(result.solution.stage, SolveStage::Converged);
        assert!(
            result.postprocess_error.is_none(),
            "{:?}",
            result.postprocess_error
        );
        compare_voltage(&prepared, &result.solution.voltage, &expected);
        for (id, r) in prepared.catalog().identities().iter().zip(&result.branches) {
            let (p, q, pt, qt, i1, i2, loading) = match id.kind {
                BranchKind::Line => {
                    let x = lines[&id.id];
                    (
                        x.p_from_mw,
                        x.q_from_mvar,
                        x.p_to_mw,
                        x.q_to_mvar,
                        x.i_from_ka,
                        x.i_to_ka,
                        x.loading_percent,
                    )
                }
                BranchKind::Transformer => {
                    let x = trafos[&id.id];
                    (
                        x.p_hv_mw,
                        x.q_hv_mvar,
                        x.p_lv_mw,
                        x.q_lv_mvar,
                        x.i_hv_ka,
                        x.i_lv_ka,
                        x.loading_percent,
                    )
                }
            };
            for (actual, expected) in [
                (r.p_from_mw, p),
                (r.q_from_mvar, q),
                (r.p_to_mw, pt),
                (r.q_to_mvar, qt),
                (r.i_from_ka, i1),
                (r.i_to_ka, i2),
                (r.loading_percent, loading),
            ] {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "device {id:?}: {actual} != {expected}"
                );
            }
        }
    }
    assert_eq!(prepared.baseline().y_bus().values(), before_y);
    assert!(prepared.catalog().in_service().iter().all(|&active| active));
}
#[test]
fn bounded_groups_empty_silent_and_mixed_failures_complete() {
    let mut reference = grid(false);
    let prepared = prepare_base(reference.world_mut()).unwrap();
    assert!(BaseBatch::new(vec![]).is_err());
    assert!(
        BaseBatch::new(vec![
            BaseTask {
                case_id: 0,
                max_iter: 50,
                tolerance: 1e-10,
                emit: true
            };
            33
        ])
        .is_err()
    );
    let empty = run_base_batches(&prepared, 2, [], |_| panic!("empty batch emitted")).unwrap();
    assert_eq!(empty.processed, 0);
    let silent = run_base_batches(
        &prepared,
        2,
        [BaseBatch::new(vec![
            BaseTask {
                case_id: 0,
                max_iter: 50,
                tolerance: 1e-10,
                emit: false
            };
            4
        ])
        .unwrap()],
        |_| panic!("silent task emitted"),
    )
    .unwrap();
    assert_eq!((silent.processed, silent.failed, silent.emitted), (4, 0, 0));
    let mut output = vec![];
    let summary = run_base_batches(
        &prepared,
        1,
        [BaseBatch::new(vec![
            BaseTask {
                case_id: 0,
                max_iter: 50,
                tolerance: 1e-10,
                emit: true,
            },
            BaseTask {
                case_id: 1,
                max_iter: 0,
                tolerance: 1e-10,
                emit: true,
            },
            BaseTask {
                case_id: 2,
                max_iter: 50,
                tolerance: 1e-10,
                emit: true,
            },
        ])
        .unwrap()],
        |r| output.push(r),
    )
    .unwrap();
    assert_eq!(
        (summary.processed, summary.failed, summary.emitted),
        (3, 1, 3)
    );
    assert!(output[0].solution.converged);
    assert_eq!(output[1].solution.stage, SolveStage::NewtonFailed);
    assert!(output[1].branches.is_empty());
    assert!(output[2].solution.converged);
}
#[test]
fn actual_outage_uses_shared_structure_and_postprocess_mask_then_base_recovers() {
    let mut reference = grid(false);
    let prepared = prepare_base(reference.world_mut()).unwrap();
    let base = prepared.baseline();
    let mut worker = N1Worker::new(base);
    let original = worker.solve_base(base, 50, 1e-10);
    assert!(original.converged);
    let out = prepared
        .catalog()
        .identities()
        .iter()
        .position(|id| id.kind == BranchKind::Line && id.id == 0)
        .unwrap();
    let line = reference
        .world_mut()
        .query::<(Entity, &LineID)>()
        .iter(reference.world())
        .find(|(_, id)| id.0 == 0)
        .unwrap()
        .0;
    reference.world_mut().entity_mut(line).insert(OutOfService);
    reference.world_mut().write_message(FullRebuildEvent);
    reference.run_pf();
    assert!(
        reference
            .world()
            .resource::<rustpower::prelude::ecs::powerflow::structure_update::LastStructureAction>()
            .full_rebuild
    );
    assert!(reference.world().resource::<PowerFlowResult>().converged);
    let outage = reference
        .world()
        .resource::<ecs::powerflow::systems::PowerFlowMat>();
    for col in 0..base.y_bus().ncols() {
        let lo = outage.y_bus.col_offsets()[col];
        let hi = outage.y_bus.col_offsets()[col + 1];
        let rows = &outage.y_bus.row_indices()[lo..hi];
        for slot in base.y_bus().col_offsets()[col]..base.y_bus().col_offsets()[col + 1] {
            worker.y_values[slot] = rows
                .binary_search(&base.y_bus().row_indices()[slot])
                .map_or(num_complex::Complex64::new(0., 0.), |k| {
                    outage.y_bus.values()[lo + k]
                });
        }
    }
    let y = nalgebra_sparse::CscMatrix::try_from_csc_data(
        base.y_bus().nrows(),
        base.y_bus().ncols(),
        base.y_bus().col_offsets().to_vec(),
        base.y_bus().row_indices().to_vec(),
        worker.y_values.clone(),
    )
    .unwrap();
    let dc = dcpf::DcpfModel::from_ybus(
        &y,
        &nalgebra::DVector::from_column_slice(base.v_init()),
        base.npv() + base.npq(),
        base.to_perm(),
        &vec![0.; y.ncols()],
    );
    assert_eq!(dc.col_ptrs, base.dc_model().col_ptrs);
    worker.dc_values.copy_from_slice(&dc.values);
    worker.dc_shift.copy_from_slice(dc.p_shift.as_slice());
    let solution = worker.solve_current(base, 50, 1e-10);
    assert!(solution.converged, "{:?}", solution.error);
    compare_voltage(
        &prepared,
        &solution.voltage,
        &reference.world().resource::<PowerFlowResult>().v,
    );
    let mut branches = vec![];
    prepared
        .catalog()
        .compute(
            &solution.voltage,
            Some(out),
            prepared.catalog().sbase(),
            &mut branches,
        )
        .unwrap();
    assert_eq!(branches[out].p_from_mw, 0.);
    assert_eq!(branches[out].i_ka, 0.);
    assert_eq!(branches[out].loading_percent, 0.);
    assert!(prepared.catalog().in_service()[out]);
    assert_eq!(worker.symbolic_ids(), base.symbolic_ids());
    worker.dc_values.fill(0.);
    assert_eq!(
        worker.solve_current(base, 50, 1e-10).stage,
        SolveStage::DcFailed
    );
    assert!(worker.solve_base(base, 50, 1e-10).converged);
    worker.y_values.clear();
    assert!(!worker.solve_current(base, 50, 1e-10).converged);
    assert!(worker.solve_base(base, 50, 1e-10).converged);
}
