#![cfg(feature = "klu")]
use rustpower_amd::klu::{KluOrdering, YbusOrderingBackend};
use rustpower_sol_klu::KLUSolver;
fn pattern() -> (Vec<usize>, Vec<usize>) {
    (
        (0..=4).map(|i| i * 4).collect(),
        (0..4).flat_map(|_| 0..4).collect(),
    )
}
#[test]
fn legacy_reset_rebuilds_even_when_structure_key_matches() {
    let (p, i) = pattern();
    let mut solver = KLUSolver::default();
    let mut state = KluOrdering::default();
    assert!(state.prepare_ybus(&mut solver, &p, &i, 1, 2));
    assert!(state.ybus_ordering_stats().optimized);
    assert!(!state.prepare_ybus(&mut solver, &p, &i, 1, 2));
    solver.reset();
    assert!(state.prepare_ybus(&mut solver, &p, &i, 1, 2));
    assert_eq!(state.ybus_ordering_stats().preparations, 2);
    unsafe {
        assert_eq!((*solver.symbolic).n, 5);
    }
}
#[test]
fn context_binds_to_the_solver_owner() {
    let (p, i) = pattern();
    let mut state = KluOrdering::default();
    let mut first = KLUSolver::default();
    assert!(state.prepare_ybus(&mut first, &p, &i, 1, 2));
    let mut second = KLUSolver::default();
    let mut diagonal_p = [0i64, 1, 2, 3, 4, 5];
    let mut diagonal_i = [0i64, 1, 2, 3, 4];
    unsafe {
        assert_eq!(
            second.solve_sym(diagonal_p.as_mut_ptr(), diagonal_i.as_mut_ptr(), 5),
            0
        );
        assert_eq!((*second.symbolic).nblocks, 5);
    }
    assert!(state.prepare_ybus(&mut second, &p, &i, 1, 2));
    unsafe {
        assert_eq!((*second.symbolic).nblocks, 1);
    }
    assert_eq!(state.ybus_ordering_stats().preparations, 2);
}
#[test]
fn policy_and_topology_invalidate_only_the_supplied_owner() {
    let (p, i) = pattern();
    let mut solver = KLUSolver::default();
    let mut state = KluOrdering::default();
    state.prepare_ybus(&mut solver, &p, &i, 1, 2);
    state.set_ybus_ordering(&mut solver, false);
    assert!(solver.symbolic.is_null());
    assert!(!state.prepare_ybus(&mut solver, &p, &i, 1, 2));
    state.set_ybus_ordering(&mut solver, true);
    state.prepare_ybus(&mut solver, &p, &i, 1, 2);
    let p2 = [0, 3, 6, 10, 14];
    let i2 = [0, 2, 3, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
    assert!(state.prepare_ybus(&mut solver, &p2, &i2, 1, 2));
    assert!(state.ybus_ordering_stats().optimized);
    assert!(state.prepare_ybus(&mut solver, &p2, &i2, 2, 1));
    unsafe {
        assert_eq!((*solver.symbolic).n, 4);
    }
}
#[cfg(feature = "c_amd")]
#[test]
fn c_rust_switches_and_invalid_structure_keep_scalar_fallback() {
    let (p, i) = pattern();
    let mut solver = KLUSolver::default();
    let mut state = KluOrdering::new(YbusOrderingBackend::C);
    assert!(state.prepare_ybus(&mut solver, &p, &i, 1, 2));
    let expected = unsafe { std::slice::from_raw_parts((*solver.symbolic).Q, 5).to_vec() };
    state.set_ybus_ordering_backend(&mut solver, YbusOrderingBackend::Rust);
    assert!(solver.symbolic.is_null());
    assert!(state.prepare_ybus(&mut solver, &p, &i, 1, 2));
    unsafe {
        assert_eq!(
            expected,
            std::slice::from_raw_parts((*solver.symbolic).Q, 5)
        );
    }
    for backend in [YbusOrderingBackend::C, YbusOrderingBackend::Rust] {
        state.set_ybus_ordering_backend(&mut solver, backend);
        assert!(state.prepare_ybus(
            &mut solver,
            &[0, 2, 4, 6, 10],
            &[0, 3, 1, 3, 2, 3, 0, 1, 2, 3],
            1,
            2
        ));
        assert!(!state.ybus_ordering_stats().optimized && solver.symbolic.is_null());
        unsafe {
            (*solver.common).ordering = 1;
        }
        state.prepare_ybus(&mut solver, &p, &i, 1, 2);
        assert!(solver.symbolic.is_null());
        unsafe {
            (*solver.common).ordering = 0;
        }
    }
}
