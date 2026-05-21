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
//! * **V2** — `newton_pf`: `JacobianPattern` + `fill_jacobian_ultimate`, i.e. the
//!   one-off symbolic pattern plus the in-place, branch-free numeric fill.
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
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

use crate::basic::dsbus_dv::{dSbus_dV, dSbus_dV_old};
use crate::basic::ecs::elements::PPNetwork;
use crate::basic::ecs::network::{DataOps, PowerFlow, PowerGrid};
use crate::basic::ecs::powerflow::systems::PowerFlowMat;
use crate::basic::new_dsdvbus::{JacobianPattern, fill_jacobian_ultimate};
use crate::basic::new_dsdvbus2::{JacobianPattern2, fill_jacobian_v2};
use crate::basic::newtonpf::{
    JacobianCache, build_jacobian, build_jacobian_cached, newton_pf, newton_pf_old, newton_pf_v0,
    newton_pf_v2,
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

/// Re-permute a `PowerFlowMat` from the `[PV | PQ | slack]` order produced by
/// `init_pf_net` into the `[PQ | PV | slack]` order that `newton_pf_v2`
/// expects. Returns the new matrices together with `perm`, where
/// `perm[old_idx] = new_idx`. Done once outside the timing loop.
fn repermute_to_pq_first(mat: &PowerFlowMat) -> (PowerFlowMat, Vec<usize>) {
    let npv = mat.npv;
    let npq = mat.npq;
    let n = mat.y_bus.ncols();

    let perm: Vec<usize> = (0..n)
        .map(|i| {
            if i < npv {
                npq + i
            } else if i < npv + npq {
                i - npv
            } else {
                i
            }
        })
        .collect();

    // Re-permute Ybus by sorting (new_col, new_row, value) triples.
    let mut entries: Vec<(usize, usize, Complex64)> = Vec::with_capacity(mat.y_bus.nnz());
    let cols = mat.y_bus.col_offsets();
    let rows = mat.y_bus.row_indices();
    let vals = mat.y_bus.values();
    for j in 0..n {
        for k in cols[j]..cols[j + 1] {
            entries.push((perm[j], perm[rows[k]], vals[k]));
        }
    }
    entries.sort_by_key(|&(c, r, _)| (c, r));

    let mut col_offsets = vec![0usize; n + 1];
    let mut row_indices = Vec::with_capacity(entries.len());
    let mut values = Vec::with_capacity(entries.len());
    let mut current_col = 0usize;
    for (idx, (c, r, val)) in entries.iter().enumerate() {
        while current_col < *c {
            col_offsets[current_col + 1] = idx;
            current_col += 1;
        }
        row_indices.push(*r);
        values.push(*val);
    }
    while current_col < n {
        col_offsets[current_col + 1] = row_indices.len();
        current_col += 1;
    }
    let y_bus_new = CscMatrix::try_from_csc_data(n, n, col_offsets, row_indices, values)
        .expect("repermute_to_pq_first: invalid CSC");

    // Re-permute the dense bus vectors.
    let mut s_bus_new = mat.s_bus.clone();
    let mut v_bus_init_new = mat.v_bus_init.clone();
    for old in 0..n {
        s_bus_new[perm[old]] = mat.s_bus[old];
        v_bus_init_new[perm[old]] = mat.v_bus_init[old];
    }

    let mat_new = PowerFlowMat {
        reorder: mat.reorder.clone(),
        y_bus: y_bus_new,
        s_bus: s_bus_new,
        v_bus_init: v_bus_init_new,
        npv,
        npq,
        to_perm: mat.to_perm.clone(),
        from_perm: mat.from_perm.clone(),
    };
    (mat_new, perm)
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

/// Run the V0/V1/V2/V2' comparison on a single system.
///   V0-V2 use the original `[PV | PQ | slack]` ordering;
///   V2' uses the new `[PQ | PV | slack]` ordering and the branch-free fill.
fn compare(name: &str, mat: &PowerFlowMat, repeats: usize) {
    println!(
        "--- {}  (n = {}, npv = {}, npq = {}) ---",
        name,
        mat.y_bus.ncols(),
        mat.npv,
        mat.npq
    );

    // One-time re-permutation for V2' (outside the timing loop).
    let (mat_pq, perm) = repermute_to_pq_first(mat);

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
    let (v_v2p_pq, it_v2p) = newton_pf_v2(
        &mat_pq.y_bus, &mat_pq.s_bus, &mat_pq.v_bus_init, mat_pq.npv, mat_pq.npq,
        None, None, &mut KLUSolver::default(),
    ).expect("V2' (newton_pf_v2) did not converge");

    // Un-permute V2''s result back to PV-first to compare with v_v2.
    let n = mat.y_bus.ncols();
    let mut v_v2p = mat.v_bus_init.clone();
    for old in 0..n {
        v_v2p[old] = v_v2p_pq[perm[old]];
    }

    let d02 = (&v_v0 - &v_v2).norm();
    let d12 = (&v_v1 - &v_v2).norm();
    let d22p = (&v_v2 - &v_v2p).norm();
    println!(
        "    iters: V0={} V1={} V2={} V2'={};   \
         d(V0,V2)={:.2e}  d(V1,V2)={:.2e}  d(V2,V2')={:.2e}",
        it_v0, it_v1, it_v2, it_v2p, d02, d12, d22p
    );
    assert!(d02 < 1e-6, "{}: V0 and V2 disagree ({:.2e})", name, d02);
    assert!(d12 < 1e-6, "{}: V1 and V2 disagree ({:.2e})", name, d12);
    assert!(d22p < 1e-6, "{}: V2 and V2' disagree ({:.2e})", name, d22p);

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
    let t_v2 = timeit("V2  fill_jacobian_ultimate", repeats, || {
        let _ = newton_pf(
            &mat.y_bus, &mat.s_bus, &mat.v_bus_init, mat.npv, mat.npq,
            None, None, &mut s2,
        );
    });
    let mut s2p = KLUSolver::default();
    let t_v2p = timeit("V2' fill_jacobian_v2 [PQ-1st]", repeats, || {
        let _ = newton_pf_v2(
            &mat_pq.y_bus, &mat_pq.s_bus, &mat_pq.v_bus_init, mat_pq.npv, mat_pq.npq,
            None, None, &mut s2p,
        );
    });
    println!(
        "    speedup:  V0/V2={:.2}x  V0/V1={:.2}x  V1/V2={:.2}x  V2/V2'={:.2}x  V0/V2'={:.2}x\n",
        t_v0.as_secs_f64() / t_v2.as_secs_f64(),
        t_v0.as_secs_f64() / t_v1.as_secs_f64(),
        t_v1.as_secs_f64() / t_v2.as_secs_f64(),
        t_v2.as_secs_f64() / t_v2p.as_secs_f64(),
        t_v0.as_secs_f64() / t_v2p.as_secs_f64(),
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
        let _ = build_jacobian(&dsm, &dsa, npv, n_ext);
    });

    // V1: single-pass dSbus_dV + build_jacobian_cached (block buffers cached).
    let mut cache: Option<JacobianCache> = None;
    let t_v1 = timeit("V1  dSbus_dV + build_jacobian_cached", repeats, || {
        let (dsm, dsa) = dSbus_dV(&mat.y_bus, &v, &v_norm);
        let _ = build_jacobian_cached(&dsm, &dsa, &mut cache, npv, n_ext);
    });

    // V2: symbolic pattern once (PV-first), then the in-place fill with in-loop branches.
    let j_pattern = JacobianPattern::build_from_permuted(
        mat.y_bus.col_offsets(),
        mat.y_bus.row_indices(),
        npv,
        npq,
    );
    let mut j_values = vec![0.0; j_pattern.nnz_j];
    let t_v2 = timeit("V2  Ybus*v + fill_jacobian_ultimate", repeats, || {
        let ibus = &mat.y_bus * &v;
        fill_jacobian_ultimate(
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

    // V2': symbolic pattern (PQ-first) + branch-free fill.
    let (mat_pq, _perm) = repermute_to_pq_first(mat);
    let v_pq = mat_pq.v_bus_init.clone();
    let v_norm_pq = v_pq.map(|e| e.simd_signum());
    let j_pattern_v2 = JacobianPattern2::build_from_permuted(
        mat_pq.y_bus.col_offsets(),
        mat_pq.y_bus.row_indices(),
        npv,
        npq,
    );
    let mut j_values_v2 = vec![0.0; j_pattern_v2.nnz_j];
    let t_v2p = timeit("V2' Ybus*v + fill_jacobian_v2 [PQ-1st]", repeats, || {
        let ibus = &mat_pq.y_bus * &v_pq;
        fill_jacobian_v2(
            &mat_pq.y_bus,
            v_pq.as_slice(),
            v_norm_pq.as_slice(),
            ibus.as_slice(),
            &j_pattern_v2,
            npv,
            npq,
            &mut j_values_v2,
        );
    });

    println!(
        "    assembly speedup:  V0/V2={:.2}x  V0/V1={:.2}x  V1/V2={:.2}x  V2/V2'={:.2}x  V0/V2'={:.2}x\n",
        t_v0.as_secs_f64() / t_v2.as_secs_f64(),
        t_v0.as_secs_f64() / t_v1.as_secs_f64(),
        t_v1.as_secs_f64() / t_v2.as_secs_f64(),
        t_v2.as_secs_f64() / t_v2p.as_secs_f64(),
        t_v0.as_secs_f64() / t_v2p.as_secs_f64(),
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

    // Build the same JacobianPattern that newton_pf builds internally.
    let j_pattern = JacobianPattern::build_from_permuted(
        mat.y_bus.col_offsets(),
        mat.y_bus.row_indices(),
        npv,
        npq,
    );
    let mut j_values = vec![0.0_f64; j_pattern.nnz_j];

    // Fill J at the converged voltage (fixed for all timing calls).
    let ibus0 = &mat.y_bus * &v_conv;
    fill_jacobian_ultimate(
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

    // ── 1. Assembly: SpMV (Ybus * v) + fill_jacobian_ultimate ──
    let t_asm = timeit("Assembly  (SpMV + fill)", repeats, || {
        let ibus = &mat.y_bus * &v_conv;
        fill_jacobian_ultimate(
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
    compare("IEEE 39", &mat, 300);
    compare_assembly("IEEE 39", &mat, 300);
    compare_klu_breakdown("IEEE 39", &mat, 300);

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
                    compare(name, &mat, repeats);
                    compare_assembly(name, &mat, repeats);
                    compare_klu_breakdown(name, &mat, repeats);
                }
                Err(_) => {
                    println!("--- {}  (skipped: {} not found) ---\n", name, rel)
                }
            }
        }
    }
}
