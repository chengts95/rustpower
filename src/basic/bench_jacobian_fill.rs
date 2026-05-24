//! Performance comparison, under the `klu` feature, of three Jacobian-assembly
//! paths that share the *identical* KLU solve:
//!
//! * **V0** — `newton_pf_v0`: `dSbus_dV_old` + uncached `build_jacobian`, i.e. the
//!   literal MATPOWER port — diagonal-matrix SpGEMM and a fresh CSC v/h-stack
//!   (structure regenerated) every iteration. The un-optimised baseline.
//! * **V1** — `newton_pf_old`: `dSbus_dV` + `build_jacobian_cached`, i.e. the
//!   cache-friendly O(NNZ) numeric pass plus block slicing with cached buffers
//!   ("semi-optimised" path — aggressively tuned, but P still applied per
//!   iteration).
//! * **V2** — `newton_pf`: `JacobianPattern2` + `fill_jacobian_v2`, i.e. the
//!   one-off symbolic pattern plus the branch-free numeric fill under `[PQ | PV | slack]`.
//!
//! Because all three paths hand the same sparse system to the same KLU solver,
//! the measured gap isolates the cost of Jacobian *assembly*: V0→V1 is the gain
//! from conventional tuning, V1→V2 the gain from the symbolic approach.
//!
//! Run with:
//! ```text
//! cargo test --release --features klu bench_jacobian_fill -- --nocapture
//! ```

use std::time::{Duration, Instant};

use nalgebra::*;
use crate::basic::dsbus_dv::{dSbus_dV, dSbus_dV_old};
use crate::basic::ecs::elements::PPNetwork;
use crate::basic::ecs::network::{DataOps, PowerFlow, PowerGrid};
use crate::basic::ecs::powerflow::systems::PowerFlowMat;
use crate::basic::new_dsdvbus2::{JacobianPattern2, fill_jacobian_v2};
use crate::basic::newtonpf::{
    JacobianCache, build_jacobian, build_jacobian_cached, newton_pf, newton_pf_old, newton_pf_v0,
};
use crate::basic::solver::KLUSolver;
use crate::io::pandapower::Network;

/// Build the ECS network, initialise it, and pull out the (reordered) power-flow
/// matrices that `newton_pf` / `newton_pf_old` consume directly.
fn extract_mat(net: Network) -> PowerFlowMat {
    let mut pf = PowerGrid::default();
    pf.world_mut().insert_resource(PPNetwork(net));
    pf.init_pf_net();
    pf.world()
        .get_resource::<PowerFlowMat>()
        .expect("init_pf_net did not produce a PowerFlowMat resource")
        .clone()
}


/// Silent variant of `timeit`: same logic, no stdout output.
fn timeit_quiet(repeats: usize, mut f: impl FnMut()) -> Duration {
    f();
    let mut total = Duration::ZERO;
    for _ in 0..repeats {
        let t = Instant::now();
        f();
        total += t.elapsed();
    }
    total / repeats as u32
}

/// Time `repeats` calls of `f` after one warm-up call; report average and min.
fn timeit(label: &str, repeats: usize, mut f: impl FnMut()) -> Duration {
    f(); // warm-up
    let mut total = Duration::ZERO;
    let mut min = Duration::MAX;
    for _ in 0..repeats {
        let t = Instant::now();
        f();
        let d = t.elapsed();
        total += d;
        min = min.min(d);
    }
    let avg = total / repeats as u32;
    println!("    {:<28} avg {:?}   min {:?}", label, avg, min);
    avg
}

/// Run the V0/V1/V2 comparison on a single system.
/// All three use the `[PQ | PV | slack]` ordering produced by ECS.
#[allow(dead_code)]
fn compare(name: &str, mat: &PowerFlowMat, repeats: usize) {
    println!(
        "--- {}  (n = {}, npv = {}, npq = {}) ---",
        name,
        mat.y_bus.ncols(),
        mat.npv,
        mat.npq
    );

    // Correctness.
    let (v_v0, it_v0) = newton_pf_v0(
        &mat.y_bus, &mat.s_bus, &mat.v_bus_init, mat.npv, mat.npq,
        None, None, &mut KLUSolver::default(),
    ).expect("V0 (newton_pf_v0) did not converge");
    let (v_v1, it_v1) = newton_pf_old(
        &mat.y_bus, &mat.s_bus, &mat.v_bus_init, mat.npv, mat.npq,
        None, None, &mut KLUSolver::default(),
    ).expect("V1 (newton_pf_old) did not converge");
    let (v_v2, it_v2) = newton_pf(
        &mat.y_bus, &mat.s_bus, &mat.v_bus_init, mat.npv, mat.npq,
        None, None, &mut KLUSolver::default(),
    ).expect("V2 (newton_pf) did not converge");

    let d02 = (&v_v0 - &v_v2).norm();
    let d12 = (&v_v1 - &v_v2).norm();
    println!(
        "    iters: V0={} V1={} V2={};   d(V0,V2)={:.2e}  d(V1,V2)={:.2e}",
        it_v0, it_v1, it_v2, d02, d12
    );
    assert!(d02 < 1e-6, "{}: V0 and V2 disagree ({:.2e})", name, d02);
    assert!(d12 < 1e-6, "{}: V1 and V2 disagree ({:.2e})", name, d12);

    // Timing.
    let mut s0 = KLUSolver::default();
    let t_v0 = timeit("V0  build_jacobian (raw)", repeats, || {
        let _ = newton_pf_v0(
            &mat.y_bus, &mat.s_bus, &mat.v_bus_init, mat.npv, mat.npq,
            None, None, &mut s0,
        );
    });
    let mut s1 = KLUSolver::default();
    let t_v1 = timeit("V1  build_jacobian_cached", repeats, || {
        let _ = newton_pf_old(
            &mat.y_bus, &mat.s_bus, &mat.v_bus_init, mat.npv, mat.npq,
            None, None, &mut s1,
        );
    });
    let mut s2 = KLUSolver::default();
    let t_v2 = timeit("V2  fill_jacobian_v2 [PQ-1st]", repeats, || {
        let _ = newton_pf(
            &mat.y_bus, &mat.s_bus, &mat.v_bus_init, mat.npv, mat.npq,
            None, None, &mut s2,
        );
    });
    println!(
        "    speedup:  V0/V2={:.2}x  V0/V1={:.2}x  V1/V2={:.2}x\n",
        t_v0.as_secs_f64() / t_v2.as_secs_f64(),
        t_v0.as_secs_f64() / t_v1.as_secs_f64(),
        t_v1.as_secs_f64() / t_v2.as_secs_f64(),
    );
}

/// Time the Jacobian *assembly* in isolation, with the shared linear solve
/// excluded. `compare` measures the whole Newton solve, in which the solver --
/// identical across V0/V1/V2 -- dilutes the assembly gap; here the assembly
/// step is called directly on a fixed voltage, so the reported ratios are the
/// undiluted cost of assembly itself.
fn compare_assembly(name: &str, mat: &PowerFlowMat, repeats: usize) {
    let n = mat.y_bus.ncols();
    let (npv, npq) = (mat.npv, mat.npq);
    let n_ext = n - npv - npq;
    let v = mat.v_bus_init.clone();
    let v_norm = v.map(|e| e.simd_signum());

    println!("--- {}  assembly only (n = {}) ---", name, n);

    // V0: dSbus_dV_old (diagonal-matrix SpGEMM) + uncached build_jacobian.
    let t_v0 = timeit("V0  dSbus_dV_old + build_jacobian", repeats, || {
        let (dsm, dsa) = dSbus_dV_old(&mat.y_bus, &v, &v_norm);
        let _ = build_jacobian(&dsm, &dsa, npq, n_ext);
    });

    // V1: single-pass dSbus_dV + build_jacobian_cached (block buffers cached).
    let mut cache: Option<JacobianCache> = None;
    let t_v1 = timeit("V1  dSbus_dV + build_jacobian_cached", repeats, || {
        let (dsm, dsa) = dSbus_dV(&mat.y_bus, &v, &v_norm);
        let _ = build_jacobian_cached(&dsm, &dsa, &mut cache, npq, n_ext);
    });

    // V2: symbolic pattern (PQ-first) + branch-free fill.
    let t_sym = timeit("V2  build_from_permuted (symbolic)", repeats, || {
        let _ = JacobianPattern2::build_from_permuted(
            mat.y_bus.col_offsets(),
            mat.y_bus.row_indices(),
            npv,
            npq,
        );
    });
    let j_pattern = JacobianPattern2::build_from_permuted(
        mat.y_bus.col_offsets(),
        mat.y_bus.row_indices(),
        npv,
        npq,
    );
    let mut j_values = vec![0.0; j_pattern.nnz_j];
    let t_v2 = timeit("V2  Ybus*v + fill_jacobian_v2 [PQ-1st]", repeats, || {
        let ibus = &mat.y_bus * &v;
        fill_jacobian_v2(
            &mat.y_bus,
            v.as_slice(),
            v_norm.as_slice(),
            ibus.as_slice(),
            &j_pattern,
            npv,
            npq,
            &mut j_values,
        );
    });

    println!(
        "    assembly speedup:  V0/V2={:.2}x  V0/V1={:.2}x  V1/V2={:.2}x",
        t_v0.as_secs_f64() / t_v2.as_secs_f64(),
        t_v0.as_secs_f64() / t_v1.as_secs_f64(),
        t_v1.as_secs_f64() / t_v2.as_secs_f64(),
    );
    println!(
        "    symbolic build:    {:.2} μs  ({:.1}x V2 per-iter assembly)\n",
        t_sym.as_secs_f64() * 1e6,
        t_sym.as_secs_f64() / t_v2.as_secs_f64(),
    );
}

/// Directly measure the split between Jacobian assembly, KLU numeric
/// re-factorisation, and KLU back-substitution for V2, all called in isolation
/// at a converged (fixed) voltage.  This replaces the back-computed estimate
/// used in the figure script with instrument-level numbers.
fn compare_klu_breakdown(name: &str, mat: &PowerFlowMat, repeats: usize) {
    let npv = mat.npv;
    let npq = mat.npq;
    let n_state = npv + 2 * npq;

    println!("--- {}  assembly vs KLU breakdown (V2, direct) ---", name);

    // One full solve: warms instruction caches and produces a converged voltage.
    let mut solver = KLUSolver::default();
    let (v_conv, n_iters) = newton_pf(
        &mat.y_bus, &mat.s_bus, &mat.v_bus_init, npv, npq,
        None, None, &mut solver,
    )
    .expect("warm-up solve failed");
    let v_norm = v_conv.map(|e| e.simd_signum());
    println!("    warm-up: {} NR iterations", n_iters);

    // Build the same JacobianPattern2 that newton_pf builds internally.
    let j_pattern = JacobianPattern2::build_from_permuted(
        mat.y_bus.col_offsets(),
        mat.y_bus.row_indices(),
        npv,
        npq,
    );
    let mut j_values = vec![0.0_f64; j_pattern.nnz_j];

    // Fill J at the converged voltage (fixed for all timing calls).
    let ibus0 = &mat.y_bus * &v_conv;
    fill_jacobian_v2(
        &mat.y_bus,
        v_conv.as_slice(),
        v_norm.as_slice(),
        ibus0.as_slice(),
        &j_pattern,
        npv,
        npq,
        &mut j_values,
    );

    // i64 structure arrays for direct KLU FFI calls.
    let mut ap: Vec<i64> = j_pattern.j_col_ptrs.iter().map(|&x| x as i64).collect();
    let mut ai: Vec<i64> = j_pattern.j_row_indices.iter().map(|&x| x as i64).collect();

    // Re-initialise KLU symbolic + numeric on OUR arrays so the internal
    // pointers are consistent (warm-up used its own internal j_pattern).
    unsafe {
        solver.0.solve_sym(ap.as_mut_ptr(), ai.as_mut_ptr(), n_state as i64);
        solver.0.factor(ap.as_mut_ptr(), ai.as_mut_ptr(), j_values.as_mut_ptr());
    }

    // ── 1. Assembly: SpMV (Ybus * v) + fill_jacobian_v2 ──
    let t_asm = timeit("Assembly  (SpMV + fill)", repeats, || {
        let ibus = &mat.y_bus * &v_conv;
        fill_jacobian_v2(
            &mat.y_bus,
            v_conv.as_slice(),
            v_norm.as_slice(),
            ibus.as_slice(),
            &j_pattern,
            npv,
            npq,
            &mut j_values,
        );
    });

    // ── 2. KLU numeric re-factorisation only ──
    let t_refactor = timeit("KLU numeric refactor   ", repeats, || unsafe {
        solver.0.refactor(
            ap.as_mut_ptr(),
            ai.as_mut_ptr(),
            j_values.as_mut_ptr(),
            n_state as i64,
        );
    });

    // ── 3. KLU back-substitution only ──
    // Zero RHS: measures throughput, not correctness of the solution.
    let mut rhs = vec![0.0_f64; n_state];
    let t_backsolve = timeit("KLU back-substitution  ", repeats, || unsafe {
        rhs.iter_mut().for_each(|x| *x = 0.0); // cheap reset
        solver.0.solve(rhs.as_mut_ptr(), n_state as i64, 1);
    });

    let total = t_asm + t_refactor + t_backsolve;
    let pct = |d: Duration| d.as_secs_f64() / total.as_secs_f64() * 100.0;
    println!(
        "    assembly {:.1}%   refactor {:.1}%   back-sub {:.1}%   \
         (measured sum = {:?})\n",
        pct(t_asm),
        pct(t_refactor),
        pct(t_backsolve),
        total,
    );
}

/// Time the two one-per-solve KLU costs that compare_klu_breakdown does NOT cover:
/// klu_l_analyze (symbolic) and klu_l_factor (first full numeric factorization).
/// These are the missing pieces needed to explain the full per-solve time in tab:ablation.
fn klu_one_time_costs(name: &str, mat: &PowerFlowMat, repeats: usize) {
    let npv = mat.npv;
    let npq = mat.npq;
    let n_state = npv + 2 * npq;

    println!("--- {}  KLU one-time costs (analyze + factor) ---", name);

    // Converged voltage → build J once.
    let mut solver = KLUSolver::default();
    let (v_conv, _) = newton_pf(
        &mat.y_bus, &mat.s_bus, &mat.v_bus_init, npv, npq,
        None, None, &mut solver,
    ).expect("warm-up solve failed");
    let v_norm = v_conv.map(|e| e.simd_signum());
    let j_pattern = JacobianPattern2::build_from_permuted(
        mat.y_bus.col_offsets(), mat.y_bus.row_indices(), npv, npq,
    );
    let mut j_values = vec![0.0_f64; j_pattern.nnz_j];
    let ibus = &mat.y_bus * &v_conv;
    fill_jacobian_v2(
        &mat.y_bus, v_conv.as_slice(), v_norm.as_slice(), ibus.as_slice(),
        &j_pattern, npv, npq, &mut j_values,
    );
    let mut ap: Vec<i64> = j_pattern.j_col_ptrs.iter().map(|&x| x as i64).collect();
    let mut ai: Vec<i64> = j_pattern.j_row_indices.iter().map(|&x| x as i64).collect();

    // klu_l_analyze: symbolic factorization (sparsity pattern only, values ignored).
    // solve_sym frees + re-runs analyze each call, so the loop is self-contained.
    let t_analyze = timeit("KLU klu_l_analyze (symbolic)   ", repeats, || unsafe {
        solver.0.solve_sym(ap.as_mut_ptr(), ai.as_mut_ptr(), n_state as i64);
    });

    // klu_l_factor: first full numeric factorization (uses values).
    // factor frees + re-runs each call.
    let t_factor = timeit("KLU klu_l_factor  (full numeric)", repeats, || unsafe {
        solver.0.factor(ap.as_mut_ptr(), ai.as_mut_ptr(), j_values.as_mut_ptr());
    });

    // klu_l_refactor for comparison (already in the figure, repeated here for ratio).
    let t_refactor = timeit("KLU klu_l_refactor (re-factor) ", repeats, || unsafe {
        solver.0.refactor(ap.as_mut_ptr(), ai.as_mut_ptr(), j_values.as_mut_ptr(), n_state as i64);
    });

    println!(
        "    factor/refactor = {:.1}x   analyze/refactor = {:.1}x\n",
        t_factor.as_secs_f64() / t_refactor.as_secs_f64(),
        t_analyze.as_secs_f64() / t_refactor.as_secs_f64(),
    );
}

/// Silently measure every component needed for the per-solve breakdown figure
/// (assembly V0/V1/V2, symbolic build, KLU analyze/factor/refactor/backsolve)
/// and append one CSV row per version to `path`.
///
/// CSV columns:
///   system, n_iter, version, asm_us, sym_us,
///   analyze_us, factor_us, refactor_us, backsolve_us
fn export_solve_breakdown_csv(path: &str, name: &str, mat: &PowerFlowMat, repeats: usize) {
    use std::io::Write as _;
    let npv = mat.npv;
    let npq = mat.npq;
    let n_ext = mat.y_bus.ncols() - npv - npq;
    let n_state = npv + 2 * npq;
    let v = mat.v_bus_init.clone();
    let v_norm = v.map(|e| e.simd_signum());

    // Warm-up: converged voltage + iteration count.
    let mut solver = KLUSolver::default();
    let (v_conv, n_iter) = newton_pf(
        &mat.y_bus, &mat.s_bus, &mat.v_bus_init, npv, npq,
        None, None, &mut solver,
    ).expect("warm-up");
    let v_norm_conv = v_conv.map(|e| e.simd_signum());

    // Assembly timings (per iteration, fixed voltage).
    let t_v0 = timeit_quiet(repeats, || {
        let (dsm, dsa) = dSbus_dV_old(&mat.y_bus, &v, &v_norm);
        let _ = build_jacobian(&dsm, &dsa, npq, n_ext);
    });
    let mut cache: Option<JacobianCache> = None;
    let t_v1 = timeit_quiet(repeats, || {
        let (dsm, dsa) = dSbus_dV(&mat.y_bus, &v, &v_norm);
        let _ = build_jacobian_cached(&dsm, &dsa, &mut cache, npq, n_ext);
    });
    let t_sym = timeit_quiet(repeats, || {
        let _ = JacobianPattern2::build_from_permuted(
            mat.y_bus.col_offsets(), mat.y_bus.row_indices(), npv, npq,
        );
    });
    let j_pattern = JacobianPattern2::build_from_permuted(
        mat.y_bus.col_offsets(), mat.y_bus.row_indices(), npv, npq,
    );
    let mut j_values = vec![0.0_f64; j_pattern.nnz_j];
    let ibus_conv = &mat.y_bus * &v_conv;
    fill_jacobian_v2(
        &mat.y_bus, v_conv.as_slice(), v_norm_conv.as_slice(),
        ibus_conv.as_slice(), &j_pattern, npv, npq, &mut j_values,
    );
    let t_v2 = timeit_quiet(repeats, || {
        let ibus = &mat.y_bus * &v_conv;
        fill_jacobian_v2(
            &mat.y_bus, v_conv.as_slice(), v_norm_conv.as_slice(),
            ibus.as_slice(), &j_pattern, npv, npq, &mut j_values,
        );
    });

    // KLU timings (using V2 J structure — same sparsity pattern for all versions).
    let mut ap: Vec<i64> = j_pattern.j_col_ptrs.iter().map(|&x| x as i64).collect();
    let mut ai: Vec<i64> = j_pattern.j_row_indices.iter().map(|&x| x as i64).collect();
    unsafe {
        solver.0.solve_sym(ap.as_mut_ptr(), ai.as_mut_ptr(), n_state as i64);
        solver.0.factor(ap.as_mut_ptr(), ai.as_mut_ptr(), j_values.as_mut_ptr());
    }
    let t_analyze = timeit_quiet(repeats, || unsafe {
        solver.0.solve_sym(ap.as_mut_ptr(), ai.as_mut_ptr(), n_state as i64);
    });
    let t_factor = timeit_quiet(repeats, || unsafe {
        solver.0.factor(ap.as_mut_ptr(), ai.as_mut_ptr(), j_values.as_mut_ptr());
    });
    let t_refactor = timeit_quiet(repeats, || unsafe {
        solver.0.refactor(ap.as_mut_ptr(), ai.as_mut_ptr(), j_values.as_mut_ptr(), n_state as i64);
    });
    let mut rhs = vec![0.0_f64; n_state];
    let t_backsolve = timeit_quiet(repeats, || unsafe {
        rhs.iter_mut().for_each(|x| *x = 0.0);
        solver.0.solve(rhs.as_mut_ptr(), n_state as i64, 1);
    });

    let us = |d: Duration| d.as_secs_f64() * 1e6;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .expect("CSV open failed");
    for (ver, asm, sym) in [
        ("V0", us(t_v0), 0.0_f64),
        ("V1", us(t_v1), 0.0_f64),
        ("V2", us(t_v2), us(t_sym)),
    ] {
        writeln!(
            f,
            "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            name, n_iter, ver,
            asm, sym,
            us(t_analyze), us(t_factor), us(t_refactor), us(t_backsolve),
        ).expect("CSV write");
    }
    println!("    [solve_breakdown.csv: {} rows written for {}]", 3, name);
}

#[test]
fn bench_jacobian_fill() {
    println!(
        "\nJacobian assembly benchmark -- V0 (raw MATPOWER port) vs \
         V1 (build_jacobian_cached) vs V2 (symbolic-cached fill), KLU solver\n"
    );

    // IEEE 39 -- transformers modelled as lines, topology identical to pandapower.
    let net: Network =
        serde_json::from_str(crate::testcases::case_ieee39::IEEE_39).unwrap();
    let mat = extract_mat(net);
    compare_assembly("IEEE 39", &mat, 300);
    compare_klu_breakdown("IEEE 39", &mat, 300);
    klu_one_time_costs("IEEE 39", &mat, 300);

    // Larger systems, loaded from the bundled case archives when present.
    if let Ok(dir) = std::env::var("CARGO_MANIFEST_DIR") {
        for (name, rel, repeats) in [
            ("IEEE 118", "cases/IEEE118/data.zip", 300usize),
            ("PEGASE 9241", "cases/pegase9241/data.zip", 30usize),
        ] {
            let path = format!("{}/{}", dir, rel);
            match crate::io::pandapower::load_csv_zip(&path) {
                Ok(net) => {
                    let mat = extract_mat(net);
                    compare_assembly(name, &mat, repeats);
                    compare_klu_breakdown(name, &mat, repeats);
                    klu_one_time_costs(name, &mat, repeats);
                }
                Err(_) => {
                    println!("--- {}  (skipped: {} not found) ---\n", name, rel)
                }
            }
        }

        // ── CSV export for per-solve breakdown figure ──────────────────────
        let csv = format!("{}/paper/solve_breakdown.csv", dir);
        {
            use std::io::Write as _;
            let mut f = std::fs::File::create(&csv).expect("cannot create solve_breakdown.csv");
            writeln!(f, "system,n_iter,version,asm_us,sym_us,analyze_us,factor_us,refactor_us,backsolve_us")
                .expect("CSV header");
        }
        let net39: Network =
            serde_json::from_str(crate::testcases::case_ieee39::IEEE_39).unwrap();
        let mat39 = extract_mat(net39);
        export_solve_breakdown_csv(&csv, "IEEE 39", &mat39, 300);

        for (name, rel, repeats) in [
            ("IEEE 118",    "cases/IEEE118/data.zip",    300usize),
            ("PEGASE 9241", "cases/pegase9241/data.zip",  30usize),
        ] {
            let path = format!("{}/{}", dir, rel);
            if let Ok(net) = crate::io::pandapower::load_csv_zip(&path) {
                let mat = extract_mat(net);
                export_solve_breakdown_csv(&csv, name, &mat, repeats);
            }
        }
        println!("\n    CSV saved to {}", csv);
    }
}
