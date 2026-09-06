use num_complex::Complex64;
use rustpower::io::pandapower::{Network, load_csv_zip};
use rustpower::new_pf::builder::*;
use rustpower::opf::builder::opf_data_from_network;
use std::env;

fn load_ieee39() -> Network {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = format!("{}/cases/IEEE39/data.zip", dir);
    load_csv_zip(&path).unwrap()
}

pub fn compare_new_pf_performance() {
    use crate::new_pf::solver;

    let net = load_ieee39();

    // 1. Setup Old Path
    let base_data = opf_data_from_network(&net);
    let ybus_old = base_data.ybus.clone();

    let mut sbus_old_vec = base_data.s_load.map(|e| -e);
    for g in 0..base_data.ng {
        let b = base_data.gen_bus[g];
        sbus_old_vec[b] += Complex64::new(base_data.pg_init[g], 0.0);
    }

    let v_init_x = base_data.warm_x0();
    let v0_old = base_data.v_from_x(&v_init_x);

    let mut bus_type = vec![2u8; base_data.nb];
    bus_type[base_data.ref_bus] = 3;
    for &b in &base_data.gen_bus {
        if b != base_data.ref_bus {
            bus_type[b] = 1;
        }
    }
    let npq = (0..base_data.nb).filter(|&b| bus_type[b] == 2).count();
    let npv = (0..base_data.nb).filter(|&b| bus_type[b] == 1).count();

    let mut solver = crate::basic::solver::RSparseSolver::default();

    // Run Old for accuracy baseline
    let (v_final_old, _) = crate::basic::newtonpf::newton_pf(
        &ybus_old,
        &sbus_old_vec,
        &v0_old,
        npv,
        npq,
        Some(1e-8),
        Some(10),
        &mut solver,
        None,
    )
    .expect("Old PF failed");

    // Warm up and then bench
    let start_old = std::time::Instant::now();
    for _ in 0..10 {
        let _ = crate::basic::newtonpf::newton_pf(
            &ybus_old,
            &sbus_old_vec,
            &v0_old,
            npv,
            npq,
            Some(1e-8),
            Some(10),
            &mut solver,
            None,
        );
    }
    let duration_old = start_old.elapsed() / 10;

    // 2. Setup New Path
    let (ybus_new, _, _) = build_ybus_binary(&net);

    let (v_final_new, _) = solver::run_newton_pf(
        &ybus_new,
        &sbus_old_vec,
        &v0_old,
        npv,
        npq,
        &mut solver,
        10,
        1e-8,
    )
    .expect("New PF failed");

    let start_new = std::time::Instant::now();
    for _ in 0..10 {
        let _ = solver::run_newton_pf(
            &ybus_new,
            &sbus_old_vec,
            &v0_old,
            npv,
            npq,
            &mut solver,
            10,
            1e-8,
        );
    }
    let duration_new = start_new.elapsed() / 10;

    // 3. Compare Results
    let mut max_err = 0.0f64;
    for i in 0..v_final_old.len() {
        let err = (v_final_old[i] - v_final_new[i]).norm();
        max_err = max_err.max(err);
    }

    println!("--- Performance Comparison (Release-like iteration) ---");
    println!("Old PF Path Avg: {:?}", duration_old);
    println!("New PF Path Avg: {:?}", duration_new);
    println!(
        "Speedup: {:.2}x",
        duration_old.as_secs_f64() / duration_new.as_secs_f64()
    );
    println!("Result Consistency (Max V Diff): {:.2e}", max_err);

    assert!(max_err < 1e-10, "New PF results diverged from baseline!");
}
