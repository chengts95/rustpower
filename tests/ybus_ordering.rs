#![cfg(feature = "klu_ybus_amd")]
use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64 as C;
use rustpower::prelude::ecs::powerflow::systems::PowerFlowMat;
use rustpower::{io::pandapower::Network, prelude::*, testcases::case_ieee39::IEEE_39};
fn fixture() -> PowerFlowMat {
    let net: Network = serde_json::from_str(IEEE_39).unwrap();
    let mut g = PowerGrid::default();
    g.world_mut().insert_resource(PPNetwork(net));
    g.init_pf_net();
    g.world().resource::<PowerFlowMat>().clone()
}
use rustpower::amd::{AmdOrderingPlugin, AmdOrderingState};
use rustpower::native::amd_solver::AmdNewtonSolver as NewtonSolver;

fn solver(m: &PowerFlowMat, enabled: bool) -> NewtonSolver {
    let mut s = NewtonSolver::new();
    s.mat = Some(m.clone());
    s.set_ybus_ordering(enabled);
    s
}
#[test]
fn explicit_native_and_ecs_addons_use_ordering() {
    let m = fixture();
    let mut a = solver(&m, true);
    let mut b = solver(&m, false);
    assert!(a.solve(100, 1e-10).unwrap().0);
    assert!(b.solve(100, 1e-10).unwrap().0);
    assert!(a.ybus_ordering_stats().optimized);
    assert_eq!(a.iterations().unwrap(), b.iterations().unwrap());
    for (x, y) in a.voltage().unwrap().iter().zip(b.voltage().unwrap()) {
        assert!((*x - y).norm() < 1e-9);
    }
    let generation = a.ybus_ordering_stats().preparations;
    assert!(a.solve(100, 1e-10).unwrap().0);
    assert_eq!(a.ybus_ordering_stats().preparations, generation);
    let net: Network = serde_json::from_str(IEEE_39).unwrap();
    let mut g = PowerGrid::default();
    g.world_mut().insert_resource(PPNetwork(net));
    g.init_pf_net();
    g.app_mut().add_plugins(AmdOrderingPlugin);
    g.run_pf();
    assert!(g.world().resource::<PowerFlowResult>().converged);
    assert!(
        g.world()
            .resource::<AmdOrderingState>()
            .ordering
            .ybus_ordering_stats()
            .optimized
    );
}
fn synthetic(change: bool) -> PowerFlowMat {
    let mut coo = CooMatrix::new(4, 4);
    let edge = if change {
        vec![(0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    } else {
        vec![(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    };
    let y = C::new(1., -8.);
    for (a, b) in edge {
        coo.push(a, a, y);
        coo.push(b, b, y);
        coo.push(a, b, -y);
        coo.push(b, a, -y);
    }
    let y = CscMatrix::from(&coo);
    let target = DVector::from_vec(vec![
        C::from_polar(0.98, -0.04),
        C::from_polar(0.99, -0.02),
        C::from_polar(1.03, 0.01),
        C::new(1.04, 0.),
    ]);
    let current = &y * &target;
    let s = target.component_mul(&current.conjugate());
    let mut init = target.clone();
    init[0] = C::new(1., 0.);
    init[1] = C::new(1., 0.);
    init[2] = C::new(1.03, 0.);
    PowerFlowMat {
        y_bus: y,
        s_bus: s,
        v_bus_init: init,
        npq: 2,
        npv: 1,
        to_perm: vec![0, 1, 2, 3],
        from_perm: vec![0, 1, 2, 3],
    }
}
#[test]
fn same_size_pattern_and_partition_changes_rebuild_both_caches() {
    let m = synthetic(false);
    let mut s = solver(&m, true);
    assert!(s.solve(100, 1e-10).unwrap().0);
    let first = s.ybus_ordering_stats().preparations;
    let next = synthetic(true);
    s.mat = Some(next.clone());
    assert!(s.solve(100, 1e-10).unwrap().0);
    assert_eq!(s.ybus_ordering_stats().preparations, first + 1);
    let mut fresh = solver(&next, false);
    assert!(fresh.solve(100, 1e-10).unwrap().0);
    for (a, b) in s.voltage().unwrap().iter().zip(fresh.voltage().unwrap()) {
        assert!((*a - b).norm() < 1e-9);
    }
    s.mat.as_mut().unwrap().npq = 1;
    s.mat.as_mut().unwrap().npv = 2;
    assert!(s.solve(100, 1e-10).unwrap().0);
    assert_eq!(s.ybus_ordering_stats().preparations, first + 2);
    assert_eq!(s.cache.j_pattern.as_ref().unwrap().j_col_ptrs.len(), 5);
}
#[test]
fn numerical_changes_reuse_symbolic_and_unsupported_asymmetry_uses_original() {
    let m = synthetic(false);
    let mut s = solver(&m, true);
    assert!(s.solve(100, 1e-10).unwrap().0);
    let generation = s.ybus_ordering_stats().preparations;
    s.mat.as_mut().unwrap().s_bus[0] += C::new(0.01, 0.);
    assert!(s.solve(100, 1e-10).unwrap().0);
    assert_eq!(s.ybus_ordering_stats().preparations, generation);
    let mut coo = CooMatrix::from(&m.y_bus);
    coo.push(0, 1, C::new(0., 0.1)); // numerical asymmetry alone is allowed
    s.mat.as_mut().unwrap().y_bus = CscMatrix::from(&coo);
    assert!(s.solve(100, 1e-10).unwrap().0);
    assert!(s.ybus_ordering_stats().optimized);
}

#[test]
fn asymmetric_pattern_falls_back_to_the_original_numerical_path() {
    let mut m = synthetic(false);
    let target = DVector::from_vec(vec![
        C::from_polar(0.98, -0.04),
        C::from_polar(0.99, -0.02),
        C::from_polar(1.03, 0.01),
        C::new(1.04, 0.),
    ]);
    let mut coo = CooMatrix::new(4, 4);
    for (i, j, v) in m.y_bus.triplet_iter() {
        if (i, j) != (0, 1) {
            coo.push(i, j, *v);
        }
    }
    m.y_bus = CscMatrix::from(&coo);
    let current = &m.y_bus * &target;
    m.s_bus = target.component_mul(&current.conjugate());
    let mut candidate = solver(&m, true);
    let mut baseline = solver(&m, false);
    assert!(candidate.solve(100, 1e-10).unwrap().0);
    assert!(baseline.solve(100, 1e-10).unwrap().0);
    assert!(!candidate.ybus_ordering_stats().optimized);
    for (a, b) in candidate
        .voltage()
        .unwrap()
        .iter()
        .zip(baseline.voltage().unwrap())
    {
        assert!((*a - b).norm() < 1e-9);
    }
}

#[test]
fn typed_rust_pattern_shares_j_layout_and_matches_c_with_reused_workspace() {
    use rustpower::prelude::{jacobian_cache::JacobianCache, solver::KLUSolver};
    use rustpower_amd::{
        Control,
        powerflow::{OrderingWorkspace, YbusPattern},
    };
    let mut workspace = OrderingWorkspace::new();
    // Reuse across different actual network sizes, then return to the first size.
    for net in [
        serde_json::from_str::<Network>(IEEE_39).unwrap(),
        rustpower::io::pandapower::load_csv_zip("cases/IEEE118/data.zip").unwrap(),
        rustpower::io::pandapower::load_csv_zip("cases/pegase9241/data.zip").unwrap(),
        serde_json::from_str::<Network>(IEEE_39).unwrap(),
    ] {
        let mut grid = PowerGrid::default();
        grid.world_mut().insert_resource(PPNetwork(net));
        grid.init_pf_net();
        let m = grid.world().resource::<PowerFlowMat>();
        let pattern =
            YbusPattern::analyze(m.y_bus.col_offsets(), m.y_bus.row_indices(), m.npv, m.npq)
                .unwrap();
        let j = pattern.reduced_jacobian();
        let old = JacobianCache::build_from_permuted(
            m.y_bus.col_offsets(),
            m.y_bus.row_indices(),
            m.npv,
            m.npq,
        );
        assert_eq!(j.column_offsets(), old.j_col_ptrs);
        assert_eq!(j.row_indices(), old.j_row_indices);
        assert_eq!(pattern.scalar_nnz(), j.row_indices().len());
        for (k, col) in pattern.columns().iter().enumerate() {
            assert_eq!(col.active_count, old.active_ends[k]);
            assert_eq!(col.pq_count, old.pq_ends[k]);
            assert_eq!(col.diagonal_position, old.diag_ptrs[k]);
        }
        let mut c = KLUSolver::default();
        let mut context =
            rustpower_amd::klu::KluOrdering::new(rustpower_amd::klu::YbusOrderingBackend::C);
        assert!(context.prepare_ybus(
            &mut c.0,
            m.y_bus.col_offsets(),
            m.y_bus.row_indices(),
            m.npv,
            m.npq
        ));
        assert!(context.ybus_ordering_stats().optimized);
        let expected =
            unsafe { std::slice::from_raw_parts((*c.0.symbolic).Q, pattern.scalar_dimension()) };
        let checked = workspace.order(&pattern, &Control::default()).unwrap();
        let unchecked =
            unsafe { workspace.order_unchecked(&pattern, &Control::default()) }.unwrap();
        assert_eq!(checked.permutation(), unchecked.permutation());
        assert_eq!(
            checked.permutation(),
            expected.iter().map(|&x| x as usize).collect::<Vec<_>>()
        );
        for (k, &variable) in checked.permutation().iter().enumerate() {
            assert_eq!(checked.inverse_permutation()[variable], k);
        }
        assert_eq!(checked.info().lnz, unchecked.info().lnz);
        assert_eq!(
            checked.info().nz_a_plus_at,
            pattern.scalar_nnz() - pattern.scalar_dimension()
        );
    }
}

#[cfg(feature = "klu_ybus_amd_rust")]
#[test]
fn rust_and_c_native_backends_match_with_btf_and_runtime_switching() {
    use rustpower_amd::klu::YbusOrderingBackend;
    for net in [
        serde_json::from_str::<Network>(IEEE_39).unwrap(),
        rustpower::io::pandapower::load_csv_zip("cases/pegase9241/data.zip").unwrap(),
    ] {
        let mut grid = PowerGrid::default();
        grid.world_mut().insert_resource(PPNetwork(net));
        grid.init_pf_net();
        let m = grid.world().resource::<PowerFlowMat>().clone();
        for btf in [0, 1] {
            let mut c = solver(&m, true);
            c.set_ybus_ordering_backend(YbusOrderingBackend::C);
            let mut rust = solver(&m, true);
            assert_eq!(rust.ybus_ordering_backend(), YbusOrderingBackend::Rust);
            unsafe {
                (*c.linear_solver.0.common).btf = btf;
                (*rust.linear_solver.0.common).btf = btf;
            }
            assert!(c.solve(100, 1e-10).unwrap().0);
            assert!(rust.solve(100, 1e-10).unwrap().0);
            assert!(rust.ybus_ordering_stats().optimized);
            assert_eq!(c.iterations().unwrap(), rust.iterations().unwrap());
            for (a, b) in c.voltage().unwrap().iter().zip(rust.voltage().unwrap()) {
                assert!((*a - b).norm() < 1e-9);
            }
            unsafe {
                let a = &*c.linear_solver.0.symbolic;
                let b = &*rust.linear_solver.0.symbolic;
                let n = a.n as usize;
                assert_eq!(
                    std::slice::from_raw_parts(a.Q, n),
                    std::slice::from_raw_parts(b.Q, n)
                );
                assert_eq!(
                    std::slice::from_raw_parts(a.P, n),
                    std::slice::from_raw_parts(b.P, n)
                );
                assert_eq!(
                    (
                        a.n,
                        a.nz,
                        a.nblocks,
                        a.maxblock,
                        a.do_btf,
                        a.structural_rank
                    ),
                    (
                        b.n,
                        b.nz,
                        b.nblocks,
                        b.maxblock,
                        b.do_btf,
                        b.structural_rank
                    )
                );
                assert_eq!((a.lnz, a.unz, a.est_flops), (b.lnz, b.unz, b.est_flops));
                assert_eq!(
                    (*c.linear_solver.0.numeric).lnz,
                    (*rust.linear_solver.0.numeric).lnz
                );
                assert_eq!(
                    (*c.linear_solver.0.numeric).unz,
                    (*rust.linear_solver.0.numeric).unz
                );
            }
            let generation = rust.ybus_ordering_stats().preparations;
            assert!(rust.solve(100, 1e-10).unwrap().0);
            assert_eq!(generation, rust.ybus_ordering_stats().preparations);
            rust.set_ybus_ordering_backend(YbusOrderingBackend::C);
            assert!(rust.linear_solver.0.symbolic.is_null());
            assert!(rust.linear_solver.0.numeric.is_null());
            assert!(rust.solve(100, 1e-10).unwrap().0);
            assert_eq!(generation + 1, rust.ybus_ordering_stats().preparations);
            rust.set_ybus_ordering_backend(YbusOrderingBackend::Rust);
            assert!(rust.solve(100, 1e-10).unwrap().0);
            assert_eq!(generation + 2, rust.ybus_ordering_stats().preparations);
        }
    }
}

#[test]
fn legacy_klu_struct_and_default_entry_remain_unchanged() {
    let original = rustpower_sol_klu::KLUSolver::default();
    let rebuilt = rustpower_sol_klu::KLUSolver {
        common: original.common,
        symbolic: original.symbolic,
        numeric: original.numeric,
    };
    std::mem::forget(original);
    drop(rebuilt);
    let m = fixture();
    let mut ordinary = rustpower::native::solver::NewtonSolver::new();
    ordinary.mat = Some(m.clone());
    assert!(ordinary.solve(100, 1e-10).unwrap().0);
    let mut candidate = solver(&m, true);
    assert!(candidate.solve(100, 1e-10).unwrap().0);
    for (a, b) in ordinary
        .voltage()
        .unwrap()
        .iter()
        .zip(candidate.voltage().unwrap())
    {
        assert!((*a - b).norm() < 1e-9);
    }
}
