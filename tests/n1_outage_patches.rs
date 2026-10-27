#![cfg(feature = "experimental_n1")]
use bevy_ecs::prelude::*;
use nalgebra::{DVector, Matrix2};
use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64;
use rustpower::{
    n1::{
        outage::{OutageTask, WorkerPatchState},
        postprocess::BranchCatalog,
        solver::{N1Worker, SharedBaseline},
    },
    prelude::{
        dcpf::DcpfModel,
        ecs::{
            elements::{
                BusMatrixIndex, FromBusMatrixIndex, PFCommonData, ToBusMatrixIndex,
                bus::VNominal,
                line::{LineID, LineParams},
                switch::NodeAggRes,
                trans::{Port4MatPatch, TransformerDevice, TransformerID},
                units::Pair,
            },
            powerflow::systems::PowerFlowMat,
        },
    },
};

fn fixture(merged: bool) -> (SharedBaseline, BranchCatalog) {
    let n = if merged { 2 } else { 3 };
    let from_perm = if merged { vec![0, 1] } else { vec![1, 0, 2] };
    let to_perm = from_perm.clone();
    let mut world = World::new();
    world.insert_resource(PFCommonData {
        sbase: 100.,
        f_hz: 50.,
        wbase: std::f64::consts::TAU * 50.,
    });
    for k in 0..3 {
        world.spawn((BusMatrixIndex(k), VNominal(Pair(10., Default::default()))));
    }
    if merged {
        world.insert_resource(NodeAggRes {
            original_to_merged: vec![0, 0, 1],
            ..Default::default()
        });
    }
    world.insert_resource(PowerFlowMat {
        y_bus: CscMatrix::identity(n),
        s_bus: DVector::zeros(n),
        v_bus_init: DVector::from_element(n, Complex64::new(1., 0.)),
        npq: n - 1,
        npv: 0,
        to_perm: to_perm.clone(),
        from_perm: from_perm.clone(),
    });
    for (id, f, t, y) in [
        (100, 0, 1, Complex64::new(2., -8.)),
        (103, 0, 1, Complex64::new(0.5, -1.)),
        (105, 1, 2, Complex64::new(1., -3.)),
        (108, 0, 2, Complex64::new(1.5, -4.)),
    ] {
        let patch = Matrix2::new(
            y + Complex64::new(0.03, 0.07),
            -y,
            -y,
            y + Complex64::new(0.02, 0.05),
        );
        world.spawn((
            LineID(id),
            LineParams {
                r_ohm_per_km: 0.1,
                x_ohm_per_km: 0.2,
                g_us_per_km: 0.,
                c_nf_per_km: 0.,
                length_km: 1.,
                df: 1.,
                parallel: 1,
                max_i_ka: 2.,
            },
            FromBusMatrixIndex(f),
            ToBusMatrixIndex(t),
            Port4MatPatch(patch),
        ));
    }
    let y = Complex64::new(0.7, -2.5);
    let phase = 7_f64.to_radians();
    world.spawn((
        TransformerID(901),
        TransformerDevice {
            df: 1.,
            i0_percent: 0.,
            pfe_kw: 0.,
            vk_percent: 10.,
            vkr_percent: 0.5,
            shift_degree: 7.,
            sn_mva: 40.,
            vn_hv_kv: 10.,
            vn_lv_kv: 10.,
            max_loading_percent: None,
            parallel: 1,
            tap: None,
        },
        FromBusMatrixIndex(0),
        ToBusMatrixIndex(2),
        Port4MatPatch(Matrix2::new(
            y,
            -y * Complex64::from_polar(1., -phase),
            -y * Complex64::from_polar(1., phase),
            y,
        )),
    ));
    let catalog = BranchCatalog::from_world(&mut world).unwrap();
    let y = stamp(&catalog, n, None);
    let mut v = DVector::from_element(n, Complex64::new(1., 0.));
    v[n - 1] = Complex64::from_polar(1.02, 0.13);
    let mat = PowerFlowMat {
        y_bus: y,
        s_bus: DVector::zeros(n),
        v_bus_init: v,
        npq: n - 1,
        npv: 0,
        to_perm,
        from_perm,
    };
    let shift = phase_vector(&catalog, &mat.from_perm, None);
    let base = SharedBaseline::from_mat(&mat, &shift).unwrap();
    (base, catalog)
}
fn stamp(catalog: &BranchCatalog, n: usize, outage: Option<usize>) -> CscMatrix<Complex64> {
    let mut coo = CooMatrix::new(n, n);
    for k in 0..n {
        coo.push(k, k, Complex64::new(0.2, 0.01));
    }
    for k in 0..catalog.len() {
        let b = catalog.branch(k).unwrap();
        if outage == Some(k) || !b.in_service {
            continue;
        }
        for (r, c, v) in [
            (b.from_bus, b.from_bus, b.y_mat[(0, 0)]),
            (b.from_bus, b.to_bus, b.y_mat[(0, 1)]),
            (b.to_bus, b.from_bus, b.y_mat[(1, 0)]),
            (b.to_bus, b.to_bus, b.y_mat[(1, 1)]),
        ] {
            coo.push(r, c, v);
        }
    }
    CscMatrix::from(&coo)
}
fn phase_vector(catalog: &BranchCatalog, from_perm: &[usize], outage: Option<usize>) -> Vec<f64> {
    let mut phase = vec![0.; from_perm.len()];
    for k in 0..catalog.len() {
        let b = catalog.branch(k).unwrap();
        if outage == Some(k) || !b.in_service {
            continue;
        }
        phase[from_perm[b.from_bus]] += catalog.phase_injections()[k];
        phase[from_perm[b.to_bus]] -= catalog.phase_injections()[k];
    }
    phase
}
fn assert_near(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }
}
#[test]
fn patches_match_independent_stamp_and_dc_rebuild_for_parallel_phase_and_slack() {
    let (base, catalog) = fixture(false);
    let mut worker = N1Worker::new(&base);
    let mut state = WorkerPatchState::default();
    for branch in 0..catalog.len() {
        let task = OutageTask::for_branch(&base, &catalog, branch, branch).unwrap();
        assert!(
            task.y_patch().len() <= 4
                && task.dc_patch().len() <= 4
                && task.dc_shift_patch().len() <= 2
        );
        state.apply(&mut worker, &base, &task).unwrap();
        let expected = stamp(&catalog, base.y_bus().nrows(), Some(branch));
        for col in 0..base.y_bus().ncols() {
            let rows = &expected.row_indices()
                [expected.col_offsets()[col]..expected.col_offsets()[col + 1]];
            for slot in base.y_bus().col_offsets()[col]..base.y_bus().col_offsets()[col + 1] {
                let value = rows
                    .binary_search(&base.y_bus().row_indices()[slot])
                    .map_or(Complex64::new(0., 0.), |k| {
                        expected.values()[expected.col_offsets()[col] + k]
                    });
                assert!((worker.y_values[slot] - value).norm() < 1e-10);
            }
        }
        // Rebuild with the worker's fixed structure, preserving explicit zeros.
        let frozen = CscMatrix::try_from_csc_data(
            base.y_bus().nrows(),
            base.y_bus().ncols(),
            base.y_bus().col_offsets().to_vec(),
            base.y_bus().row_indices().to_vec(),
            worker.y_values.clone(),
        )
        .unwrap();
        let dc = DcpfModel::from_ybus(
            &frozen,
            &DVector::from_column_slice(base.v_init()),
            base.npq() + base.npv(),
            base.to_perm(),
            &phase_vector(&catalog, base.from_perm(), Some(branch)),
        );
        assert_eq!(dc.col_ptrs, base.dc_model().col_ptrs);
        assert_eq!(dc.row_indices, base.dc_model().row_indices);
        assert_near(&worker.dc_values, &dc.values);
        assert_near(&worker.dc_shift, dc.p_shift.as_slice());
    }
    state.clear(&mut worker, &base).unwrap();
    assert_eq!(worker.y_values, base.y_bus().values());
    assert_eq!(worker.dc_values, base.dc_model().values);
    assert_eq!(worker.dc_shift, base.dc_model().p_shift.as_slice());
}
#[test]
fn merged_endpoint_aliases_collapse_into_one_y_slot_without_dc_change() {
    let (base, catalog) = fixture(true);
    let task = OutageTask::for_branch(&base, &catalog, 0, 10).unwrap();
    assert_eq!(
        catalog.branch(0).unwrap().from_bus,
        catalog.branch(0).unwrap().to_bus
    );
    assert_eq!(task.y_patch().len(), 1);
    let mut worker = N1Worker::new(&base);
    let mut state = WorkerPatchState::default();
    state.apply(&mut worker, &base, &task).unwrap();
    assert_eq!(worker.dc_values, base.dc_model().values);
    assert_eq!(worker.dc_shift, base.dc_model().p_shift.as_slice());
    let expected = stamp(&catalog, base.y_bus().nrows(), Some(0));
    for (a, b) in worker.y_values.iter().zip(expected.values()) {
        assert!((*a - *b).norm() < 1e-10);
    }
}
#[test]
fn baseline_mismatch_and_invalid_worker_lengths_leave_state_unchanged() {
    let (base, catalog) = fixture(false);
    let (other, other_catalog) = fixture(false);
    let task = OutageTask::for_branch(&base, &catalog, 0, 0).unwrap();
    let alien = OutageTask::for_branch(&other, &other_catalog, 1, 1).unwrap();
    let mut worker = N1Worker::new(&base);
    let mut state = WorkerPatchState::default();
    state.apply(&mut worker, &base, &task).unwrap();
    let previous = worker.y_values.clone();
    assert!(state.apply(&mut worker, &base, &alien).is_err());
    assert_eq!(previous, worker.y_values);
    worker.dc_shift.clear();
    assert!(state.apply(&mut worker, &base, &task).is_err());
    assert_eq!(previous, worker.y_values);
    assert!(OutageTask::for_branch(&base, &catalog, usize::MAX, 0).is_err());
}

#[test]
fn ac_only_tasks_patch_y_and_restore_without_dc_storage() {
    let (base, catalog) = fixture(false);
    let mut worker = N1Worker::new_ac(&base);
    let mut state = WorkerPatchState::default();
    assert!(worker.dc_values.is_empty());
    assert!(worker.dc_shift.is_empty());
    for branch in 0..catalog.len() {
        let task = OutageTask::for_ac_branch(&base, &catalog, branch, branch).unwrap();
        assert!(task.dc_patch().is_empty());
        assert!(task.dc_shift_patch().is_empty());
        state.apply(&mut worker, &base, &task).unwrap();
        let expected = stamp(&catalog, base.y_bus().nrows(), Some(branch));
        for col in 0..base.y_bus().ncols() {
            let lo = expected.col_offsets()[col];
            let rows = &expected.row_indices()[lo..expected.col_offsets()[col + 1]];
            for slot in base.y_bus().col_offsets()[col]..base.y_bus().col_offsets()[col + 1] {
                let value = rows
                    .binary_search(&base.y_bus().row_indices()[slot])
                    .map_or(Complex64::new(0., 0.), |k| expected.values()[lo + k]);
                assert!((worker.y_values[slot] - value).norm() < 1e-10);
            }
        }
        assert!(worker.dc_values.is_empty());
        assert!(worker.dc_shift.is_empty());
    }
    state.clear(&mut worker, &base).unwrap();
    assert_eq!(worker.y_values, base.y_bus().values());
    assert!(worker.dc_values.is_empty());
    assert!(worker.dc_shift.is_empty());
}

#[test]
fn cross_mode_tasks_are_rejected_before_modifying_private_values() {
    let (base, catalog) = fixture(false);
    let ac_task = OutageTask::for_ac_branch(&base, &catalog, 0, 0).unwrap();
    let dc_task = OutageTask::for_branch(&base, &catalog, 1, 1).unwrap();
    let mut ac_worker = N1Worker::new_ac(&base);
    let mut dc_worker = N1Worker::new(&base);
    let mut ac_state = WorkerPatchState::default();
    let mut dc_state = WorkerPatchState::default();
    let original_ac = ac_worker.y_values.clone();
    assert!(ac_state.apply(&mut ac_worker, &base, &dc_task).is_err());
    assert_eq!(ac_worker.y_values, original_ac);
    let original_dc = dc_worker.y_values.clone();
    assert!(dc_state.apply(&mut dc_worker, &base, &ac_task).is_err());
    assert_eq!(dc_worker.y_values, original_dc);
    ac_state.apply(&mut ac_worker, &base, &ac_task).unwrap();
    let previous = ac_worker.y_values.clone();
    assert!(ac_state.apply(&mut ac_worker, &base, &dc_task).is_err());
    assert_eq!(ac_worker.y_values, previous);
    ac_state.clear(&mut ac_worker, &base).unwrap();
    assert_eq!(ac_worker.y_values, original_ac);
    ac_worker.dc_values.push(1.);
    assert!(ac_state.apply(&mut ac_worker, &base, &ac_task).is_err());
    assert_eq!(ac_worker.y_values, original_ac);
}

#[test]
fn current_dc_rebuild_has_rank_one_single_branch_update() {
    for merged in [false, true] {
        let (base, catalog) = fixture(merged);
        let n = base.npq() + base.npv();
        let at = |y: &CscMatrix<Complex64>, row: usize, col: usize| {
            let lo = y.col_offsets()[col];
            let hi = y.col_offsets()[col + 1];
            y.row_indices()[lo..hi]
                .binary_search(&row)
                .map_or(Complex64::new(0., 0.), |k| y.values()[lo + k])
        };
        for branch in 0..catalog.len() {
            let f = catalog.from_buses()[branch];
            let t = catalog.to_buses()[branch];
            let changed = stamp(&catalog, base.v_init().len(), Some(branch));
            let shift = phase_vector(&catalog, base.from_perm(), Some(branch));
            let dc = DcpfModel::from_ybus(
                &changed,
                &DVector::from_column_slice(base.v_init()),
                n,
                base.to_perm(),
                &shift,
            );
            let mut u = vec![0.; n];
            let mut v = vec![0.; n];
            let mut alpha = 0.;
            if f != t {
                let df = at(&changed, t, f).norm() - at(base.y_bus(), t, f).norm();
                let dt = at(&changed, f, t).norm() - at(base.y_bus(), f, t).norm();
                let phase = catalog.phase_injections()[branch];
                alpha = match (f < n, t < n) {
                    (true, true) => phase,
                    (true, false) => phase + df * base.v_init()[t].arg(),
                    (false, true) => phase - dt * base.v_init()[f].arg(),
                    (false, false) => 0.,
                };
                if f < n {
                    u[f] += 1.;
                    v[f] += df;
                }
                if t < n {
                    u[t] -= 1.;
                    v[t] -= dt;
                }
            }
            for row in 0..n {
                let rhs_change = base.dc_model().p_shift[row] - dc.p_shift[row];
                assert!((rhs_change - alpha * u[row]).abs() < 1e-10);
            }
            let entry = |d: &DcpfModel, row: usize, col: usize| {
                let lo = d.col_ptrs[col];
                let hi = d.col_ptrs[col + 1];
                d.row_indices[lo..hi]
                    .binary_search(&row)
                    .map_or(0., |k| d.values[lo + k])
            };
            for col in 0..n {
                for row in 0..n {
                    let change = entry(&dc, row, col) - entry(base.dc_model(), row, col);
                    assert!(
                        (change - u[row] * v[col]).abs() < 1e-10,
                        "branch {branch}, ({row},{col}), delta {change}, outer {}",
                        u[row] * v[col]
                    );
                }
            }
        }
    }
}
