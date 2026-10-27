//! Experimental comparison only. Production solver dispatch is not changed.
use nalgebra::DVector;
use rustpower::prelude::ecs::powerflow::systems::PowerFlowMat;
use rustpower::{
    io::pandapower::{Network, load_csv_zip},
    prelude::*,
    prelude::{jacobian_cache::JacobianCache, newtonpf::NewtonCache},
    testcases::case_ieee39::IEEE_39,
};
use rustpower_amd::{Control, order, order_unchecked, order_ybus, order_ybus_unchecked};
use serde_json::json;
use std::ffi::c_void;
use std::{fs::File, io::Write, time::Instant};
#[link(name = "amd")]
unsafe extern "C" {
    fn amd_l_order(
        n: i64,
        p: *const i64,
        i: *const i64,
        q: *mut i64,
        ctl: *const f64,
        info: *mut f64,
    ) -> i32;
}
#[link(name = "klu")]
unsafe extern "C" {
    fn klu_l_analyze_given(
        n: i64,
        p: *const i64,
        i: *const i64,
        pr: *const i64,
        pc: *const i64,
        c: *mut c_void,
    ) -> *mut c_void;
}
unsafe extern "C" {
    fn rustpower_ybus_analyze(
        p: *const usize,
        i: *const usize,
        active: i64,
        npq: i64,
        c: *mut c_void,
    ) -> *mut c_void;
    fn sched_getcpu() -> i32;
}
fn install_j_cache(s: &mut NewtonSolver, j: JacobianCache, m: &PowerFlowMat) {
    let n = m.npv + 2 * m.npq;
    s.cache = NewtonCache {
        npv: m.npv,
        npq: m.npq,
        j_values: vec![0.; j.j_row_indices.len()],
        ibus: DVector::zeros(m.y_bus.ncols()),
        F: DVector::zeros(n),
        s_calc: DVector::zeros(m.y_bus.ncols()),
        j_pattern: Some(j),
        ..Default::default()
    };
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let reps: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(30);
    let mut out =
        File::create(args.get(2).map(String::as_str).unwrap_or("rust-amd.jsonl")).unwrap();
    let cpu = std::env::var("BENCH_CPU")
        .ok()
        .map(|s| s.parse::<i32>().unwrap());
    let paths = [
        "c_scalar",
        "rust_checked_scalar",
        "rust_unchecked_scalar",
        "c_seeded",
        "rust_checked_seeded",
        "rust_unchecked_seeded",
    ];
    for (case, net) in [
        ("39", serde_json::from_str::<Network>(IEEE_39).unwrap()),
        ("118", load_csv_zip("cases/IEEE118/data.zip").unwrap()),
        ("9241", load_csv_zip("cases/pegase9241/data.zip").unwrap()),
    ] {
        let mut grid = PowerGrid::default();
        grid.world_mut().insert_resource(PPNetwork(net));
        grid.init_pf_net();
        let m = grid.world().resource::<PowerFlowMat>().clone();
        let n = m.npv + 2 * m.npq;
        let jref = JacobianCache::build_from_permuted(
            m.y_bus.col_offsets(),
            m.y_bus.row_indices(),
            m.npv,
            m.npq,
        );
        let mut cq = vec![0i64; n];
        let mut cinfo = [0.; 20];
        assert_eq!(
            unsafe {
                amd_l_order(
                    n as i64,
                    jref.j_col_ptrs.as_ptr().cast(),
                    jref.j_row_indices.as_ptr().cast(),
                    cq.as_mut_ptr(),
                    std::ptr::null(),
                    cinfo.as_mut_ptr(),
                )
            },
            0
        );
        let mut target = NewtonSolver::new();
        target.mat = Some(m.clone());

        unsafe {
            (*target.linear_solver.0.common).btf = 0;
        }
        assert!(target.solve(100, 1e-10).unwrap().0);
        let voltage = target.voltage().unwrap();
        let iterations = target.iterations().unwrap();
        let mut cs = rustpower::prelude::solver::KLUSolver::default();
        unsafe {
            (*cs.0.common).btf = 0;
            cs.0.symbolic = rustpower_ybus_analyze(
                m.y_bus.col_offsets().as_ptr(),
                m.y_bus.row_indices().as_ptr(),
                (m.npv + m.npq) as i64,
                m.npq as i64,
                cs.0.common.cast(),
            )
            .cast();
            assert!(!cs.0.symbolic.is_null());
        }
        let seeded_q = unsafe { std::slice::from_raw_parts((*cs.0.symbolic).Q, n) }.to_vec();
        for pair in 0..(reps + 5) {
            for step in 0..paths.len() {
                let idx = if pair % 2 == 0 {
                    (step + pair) % 6
                } else {
                    (pair + 6 - step % 6) % 6
                };
                let path = paths[idx];
                let before = unsafe { sched_getcpu() };
                if let Some(cpu) = cpu {
                    assert_eq!(cpu, before);
                }
                let mut s = NewtonSolver::new();
                s.mat = Some(m.clone());

                unsafe {
                    (*s.linear_solver.0.common).btf = 0;
                }
                let cold = Instant::now();
                let jt = Instant::now();
                let j = JacobianCache::build_from_permuted(
                    m.y_bus.col_offsets(),
                    m.y_bus.row_indices(),
                    m.npv,
                    m.npq,
                );
                let j_us = jt.elapsed().as_secs_f64() * 1e6;
                let prep = Instant::now();
                let mut order_us = None;
                unsafe {
                    if idx == 3 {
                        assert!(rustpower_amd::ybus_is_eligible(
                            m.y_bus.col_offsets(),
                            m.y_bus.row_indices(),
                            m.npv,
                            m.npq
                        ));
                        s.linear_solver.0.symbolic = rustpower_ybus_analyze(
                            m.y_bus.col_offsets().as_ptr(),
                            m.y_bus.row_indices().as_ptr(),
                            (m.npv + m.npq) as i64,
                            m.npq as i64,
                            s.linear_solver.0.common.cast(),
                        )
                        .cast();
                    } else {
                        let t = Instant::now();
                        let (q, lnz) = if idx == 0 {
                            let mut q = vec![0i64; n];
                            let mut info = [0.; 20];
                            assert_eq!(
                                amd_l_order(
                                    n as i64,
                                    j.j_col_ptrs.as_ptr().cast(),
                                    j.j_row_indices.as_ptr().cast(),
                                    q.as_mut_ptr(),
                                    std::ptr::null(),
                                    info.as_mut_ptr()
                                ),
                                0
                            );
                            (q, info[9] as i64 + n as i64)
                        } else {
                            let ctl = Control::default();
                            let (q, _, info) = match idx {
                                1 => order(n, &j.j_col_ptrs, &j.j_row_indices, &ctl),
                                2 => order_unchecked(n, &j.j_col_ptrs, &j.j_row_indices, &ctl),
                                4 => order_ybus(
                                    m.y_bus.col_offsets(),
                                    m.y_bus.row_indices(),
                                    m.npv,
                                    m.npq,
                                    &ctl,
                                ),
                                5 => order_ybus_unchecked(
                                    m.y_bus.col_offsets(),
                                    m.y_bus.row_indices(),
                                    m.npv,
                                    m.npq,
                                    &ctl,
                                ),
                                _ => unreachable!(),
                            }
                            .unwrap();
                            (
                                q.into_iter().map(|x| x as i64).collect::<Vec<_>>(),
                                (info.lnz + n) as i64,
                            )
                        };
                        order_us = Some(t.elapsed().as_secs_f64() * 1e6);
                        assert_eq!(
                            &q,
                            if idx < 3 { &cq } else { &seeded_q },
                            "case={case} path={path}"
                        );
                        s.linear_solver.0.symbolic = klu_l_analyze_given(
                            n as i64,
                            j.j_col_ptrs.as_ptr().cast(),
                            j.j_row_indices.as_ptr().cast(),
                            q.as_ptr(),
                            q.as_ptr(),
                            s.linear_solver.0.common.cast(),
                        )
                        .cast();
                        assert!(!s.linear_solver.0.symbolic.is_null());
                        let sym = &mut *s.linear_solver.0.symbolic;
                        sym.lnz = lnz as f64;
                        sym.unz = lnz as f64;
                        *sym.Lnz = lnz as f64;
                        sym.symmetry = 1.;
                    }
                    assert!(!s.linear_solver.0.symbolic.is_null());
                }
                let prep_us = prep.elapsed().as_secs_f64() * 1e6;
                let q = unsafe { std::slice::from_raw_parts((*s.linear_solver.0.symbolic).Q, n) };
                assert_eq!(q, if idx < 3 { &cq } else { &seeded_q });
                install_j_cache(&mut s, j, &m);
                assert!(s.solve(100, 1e-10).unwrap().0);
                let cold_us = cold.elapsed().as_secs_f64() * 1e6;
                assert_eq!(s.iterations().unwrap(), iterations);
                let mut error = s
                    .voltage()
                    .unwrap()
                    .iter()
                    .zip(&voltage)
                    .map(|(a, b)| (*a - b).norm())
                    .fold(0f64, f64::max);
                assert!(error < 1e-9);
                let (lnz, unz) = unsafe {
                    (
                        (*s.linear_solver.0.numeric).lnz,
                        (*s.linear_solver.0.numeric).unz,
                    )
                };
                s.mat.as_mut().unwrap().v_bus_init = m.v_bus_init.clone();
                let t = Instant::now();
                assert!(s.solve(100, 1e-10).unwrap().0);
                let reuse_us = t.elapsed().as_secs_f64() * 1e6;
                assert_eq!(s.iterations().unwrap(), iterations);
                error = error.max(
                    s.voltage()
                        .unwrap()
                        .iter()
                        .zip(&voltage)
                        .map(|(a, b)| (*a - b).norm())
                        .fold(0f64, f64::max),
                );
                assert!(error < 1e-9);
                let after = unsafe { sched_getcpu() };
                if let Some(cpu) = cpu {
                    assert_eq!(cpu, after);
                }
                if pair >= 5 {
                    writeln!(out,"{}",json!({"case":case,"path":path,"repeat":pair-5,"cpu_before":before,"cpu_after":after,"btf":0,"j_symbolic_us":j_us,"ordering_us":order_us,"prepare_symbolic_us":prep_us,"cold_us":cold_us,"reuse_us":reuse_us,"lnz":lnz,"unz":unz,"iterations":iterations,"voltage_error":error,"permutation_matches_c":true})).unwrap();
                }
            }
        }
        eprintln!("case {case}: all6 paths complete");
    }
}
