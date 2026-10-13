//! Symbolic preparation benchmark, independent of the Newton iteration loop.
//! Run with SYMBOLIC_OUTPUT_DIR set and:
//! cargo test --offline --release --features klu_dyn --lib bench_symbolic_analysis -- --ignored --nocapture
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use nalgebra::DVector;
use num_complex::Complex64;
use serde::Serialize;

use super::ecs::elements::PPNetwork;
use super::ecs::network::{DataOps, PowerFlow, PowerGrid};
use super::ecs::powerflow::systems::PowerFlowMat;
use super::new_dsdvbus2::{JacobianPattern2, column_boundaries};
use super::new_dsdvbus3::fill_jacobian_v3;
use super::newtonpf::{NewtonCache, csc_matvec_and_scalc, newton_pf};
use super::solver::KLUSolver;
use super::test_symbolic_ranges::assert_patterns_equal;

const BATCHES: usize = 9;
const WARMUP: usize = 30;

#[derive(Serialize)]
struct Measurement {
    mean_us: f64,
    median_us: f64,
    min_us: f64,
    max_us: f64,
    batches_us: Vec<f64>,
}

fn distribution(samples: Vec<f64>) -> Measurement {
    let mean_us = samples.iter().sum::<f64>() / samples.len() as f64;
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    Measurement {
        mean_us,
        median_us: sorted[sorted.len() / 2],
        min_us: sorted[0],
        max_us: sorted[sorted.len() - 1],
        batches_us: samples,
    }
}

fn measure(repeats: usize, mut call: impl FnMut()) -> Measurement {
    for _ in 0..WARMUP {
        call();
    }
    distribution(
        (0..BATCHES)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..repeats {
                    call();
                }
                start.elapsed().as_secs_f64() * 1e6 / repeats as f64
            })
            .collect(),
    )
}

fn measure_pair(
    repeats: usize,
    mut baseline: impl FnMut(),
    mut optimized: impl FnMut(),
) -> (Measurement, Measurement) {
    for _ in 0..WARMUP {
        baseline();
        optimized();
    }
    let mut old = Vec::new();
    let mut new = Vec::new();
    let timed = |f: &mut dyn FnMut()| {
        let start = Instant::now();
        for _ in 0..repeats {
            f();
        }
        start.elapsed().as_secs_f64() * 1e6 / repeats as f64
    };
    for batch in 0..BATCHES {
        // Alternate the order within paired batches to reduce drift bias.
        let (a, b) = if batch % 2 == 0 {
            (timed(&mut baseline), timed(&mut optimized))
        } else {
            let b = timed(&mut optimized);
            (timed(&mut baseline), b)
        };
        old.push(a);
        new.push(b);
    }
    (distribution(old), distribution(new))
}

fn lookup_columns<const RESTRICTED: bool>(cp: &[usize], rows: &[usize], npq: usize, npv: usize) {
    for k in 0..npq + npv {
        black_box(column_boundaries::<RESTRICTED>(
            black_box(&rows[cp[k]..cp[k + 1]]),
            k,
            npq,
            npq + npv,
        ));
    }
}

#[derive(Serialize)]
struct Variant {
    name: &'static str,
    full_build_and_drop: Measurement,
    allocation: Measurement,
    angle_columns_with_lookups: Measurement,
    magnitude_columns: Measurement,
    lookup_only: Measurement,
}

fn measure_variant<const RESTRICTED: bool>(mat: &PowerFlowMat, repeats: usize) -> Variant {
    let cp = mat.y_bus.col_offsets();
    let rows = mat.y_bus.row_indices();
    let (npv, npq) = (mat.npv, mat.npq);
    let build = || {
        if RESTRICTED {
            JacobianPattern2::build_from_permuted(cp, rows, npv, npq)
        } else {
            JacobianPattern2::build_full_range(cp, rows, npv, npq)
        }
    };
    let full_build_and_drop = measure(repeats, || {
        black_box(build());
    });
    for _ in 0..WARMUP {
        black_box(JacobianPattern2::build_profiled::<RESTRICTED>(
            cp, rows, npv, npq,
        ));
    }
    let mut allocation = Vec::new();
    let mut angle = Vec::new();
    let mut magnitude = Vec::new();
    for _ in 0..BATCHES {
        let mut totals = [0.0; 3];
        for _ in 0..repeats {
            let (pattern, times) =
                JacobianPattern2::build_profiled::<RESTRICTED>(cp, rows, npv, npq);
            totals[0] += times.allocate.as_secs_f64() * 1e6;
            totals[1] += times.angle_columns.as_secs_f64() * 1e6;
            totals[2] += times.magnitude_columns.as_secs_f64() * 1e6;
            black_box(pattern);
        }
        allocation.push(totals[0] / repeats as f64);
        angle.push(totals[1] / repeats as f64);
        magnitude.push(totals[2] / repeats as f64);
    }
    // This independent microbenchmark isolates search work. It is already
    // included in angle_columns_with_lookups and must not be added to it.
    let lookup_only = measure(repeats, || lookup_columns::<RESTRICTED>(cp, rows, npq, npv));
    Variant {
        name: if RESTRICTED {
            "restricted_ranges_0.5.3"
        } else {
            "full_ranges_0.5.2"
        },
        full_build_and_drop,
        allocation: distribution(allocation),
        angle_columns_with_lookups: distribution(angle),
        magnitude_columns: distribution(magnitude),
        lookup_only,
    }
}

fn klu_symbolic(pattern: &JacobianPattern2, repeats: usize) -> Measurement {
    let mut ap: Vec<i64> = pattern.j_col_ptrs.iter().map(|&i| i as i64).collect();
    let mut ai: Vec<i64> = pattern.j_row_indices.iter().map(|&i| i as i64).collect();
    let n = (ap.len() - 1) as i64;
    let mut call = || {
        let mut solver = KLUSolver::default();
        let start = Instant::now();
        let status = unsafe { solver.0.solve_sym(ap.as_mut_ptr(), ai.as_mut_ptr(), n) };
        let time = start.elapsed().as_secs_f64() * 1e6;
        assert_eq!(status, 0);
        assert!(!solver.0.symbolic.is_null());
        // Context allocation and symbolic destruction are outside this timer.
        black_box(&solver);
        time
    };
    for _ in 0..WARMUP {
        black_box(call());
    }
    distribution(
        (0..BATCHES)
            .map(|_| (0..repeats).map(|_| call()).sum::<f64>() / repeats as f64)
            .collect(),
    )
}

fn cache(pattern: JacobianPattern2, mat: &PowerFlowMat) -> NewtonCache {
    NewtonCache {
        npv: mat.npv,
        npq: mat.npq,
        j_values: vec![0.0; pattern.nnz_j],
        // This benchmark compares the V3 boundary searches. Reuse their exact
        // structure and cuts in the current V4 solver for convergence checks.
        j_pattern: Some(super::jacobian_cache::JacobianCache {
            j_col_ptrs: pattern.j_col_ptrs,
            j_row_indices: pattern.j_row_indices,
            pq_ends: pattern.pq_ends,
            active_ends: pattern.active_ends,
            diag_ptrs: pattern.diag_ptrs,
        }),
        ibus: DVector::zeros(mat.y_bus.ncols()),
        F: DVector::zeros(mat.npv + 2 * mat.npq),
        s_calc: DVector::zeros(mat.y_bus.ncols()),
        ..Default::default()
    }
}

fn verify(mat: &PowerFlowMat, old: &JacobianPattern2, new: &JacobianPattern2) -> (usize, f64) {
    use nalgebra::SimdComplexField;
    assert_patterns_equal(old, new);
    let v = &mat.v_bus_init;
    let norm = v.map(|x| x.simd_signum());
    let mut ibus = DVector::<Complex64>::zeros(v.len());
    let mut scalc = DVector::<Complex64>::zeros(v.len());
    csc_matvec_and_scalc(
        mat.y_bus.col_offsets(),
        mat.y_bus.row_indices(),
        mat.y_bus.values(),
        v.as_slice(),
        ibus.as_mut_slice(),
        scalc.as_mut_slice(),
    );
    let mut a = vec![0.0; old.nnz_j];
    let mut b = vec![0.0; new.nnz_j];
    fill_jacobian_v3(
        &mat.y_bus,
        v.as_slice(),
        norm.as_slice(),
        scalc.as_slice(),
        old,
        mat.npv,
        mat.npq,
        &mut a,
    );
    fill_jacobian_v3(
        &mat.y_bus,
        v.as_slice(),
        norm.as_slice(),
        scalc.as_slice(),
        new,
        mat.npv,
        mat.npq,
        &mut b,
    );
    assert_eq!(a, b);
    let mut old_cache = cache(old.clone(), mat);
    let mut new_cache = cache(new.clone(), mat);
    let mut old_klu = KLUSolver::default();
    let mut new_klu = KLUSolver::default();
    let (va, ia) = newton_pf(
        &mat.y_bus,
        &mat.s_bus,
        v,
        mat.npv,
        mat.npq,
        Some(1e-6),
        Some(100),
        &mut old_klu,
        Some(&mut old_cache),
    )
    .expect("baseline solve");
    let (vb, ib) = newton_pf(
        &mat.y_bus,
        &mat.s_bus,
        v,
        mat.npv,
        mat.npq,
        Some(1e-6),
        Some(100),
        &mut new_klu,
        Some(&mut new_cache),
    )
    .expect("optimized solve");
    assert_eq!(ia, ib);
    let error = va
        .iter()
        .zip(vb.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max);
    assert!(error < 1e-13, "voltage error {error}");
    (ia, error)
}

#[derive(Serialize)]
struct Case {
    system: &'static str,
    buses: usize,
    npq: usize,
    npv: usize,
    ybus_nnz: usize,
    jacobian_nnz: usize,
    repeats_per_batch: usize,
    variants: [Variant; 2],
    klu_symbolic_analysis: Measurement,
    full_build_speedup: f64,
    lookup_speedup: f64,
    identical_structure_and_numeric_jacobian: bool,
    iterations: usize,
    voltage_error: f64,
}

#[test]
#[ignore = "explicit benchmark; writes results to SYMBOLIC_OUTPUT_DIR"]
fn bench_symbolic_analysis() {
    let output =
        PathBuf::from(std::env::var("SYMBOLIC_OUTPUT_DIR").expect("set SYMBOLIC_OUTPUT_DIR"));
    std::fs::create_dir_all(&output).unwrap();
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let inputs = std::env::var("SYMBOLIC_CASE_DIR").ok();
    let mut cases = Vec::new();
    for (name, id, relative) in [
        ("IEEE 39", "39", ""),
        ("IEEE 118", "118", "cases/IEEE118/data.zip"),
        ("PEGASE 1354", "1354", ""),
        ("PEGASE 9241", "9241", "cases/pegase9241/data.zip"),
    ] {
        let net = if let Some(ref dir) = inputs {
            let source = std::fs::read_to_string(format!("{dir}/{id}.json")).expect("case JSON");
            serde_json::from_str(&source).expect("RustPower network JSON")
        } else if id == "39" {
            serde_json::from_str(crate::testcases::case_ieee39::IEEE_39).unwrap()
        } else {
            assert!(
                !relative.is_empty(),
                "set SYMBOLIC_CASE_DIR to include PEGASE 1354"
            );
            crate::io::pandapower::load_csv_zip(repo.join(relative).to_str().unwrap()).unwrap()
        };
        let mut grid = PowerGrid::default();
        grid.world_mut().insert_resource(PPNetwork(net));
        grid.init_pf_net();
        let mat = grid
            .world()
            .get_resource::<PowerFlowMat>()
            .expect("power-flow matrices")
            .clone();
        drop(grid);
        let old = JacobianPattern2::build_full_range(
            mat.y_bus.col_offsets(),
            mat.y_bus.row_indices(),
            mat.npv,
            mat.npq,
        );
        let new = JacobianPattern2::build_from_permuted(
            mat.y_bus.col_offsets(),
            mat.y_bus.row_indices(),
            mat.npv,
            mat.npq,
        );
        let (iterations, voltage_error) = verify(&mat, &old, &new);
        let repeats = if id == "9241" { 500 } else { 2000 };
        let mut baseline = measure_variant::<false>(&mat, repeats);
        let mut optimized = measure_variant::<true>(&mat, repeats);
        let cp = mat.y_bus.col_offsets();
        let rows = mat.y_bus.row_indices();
        let (old_total, new_total) = measure_pair(
            repeats,
            || {
                black_box(JacobianPattern2::build_full_range(
                    cp, rows, mat.npv, mat.npq,
                ));
            },
            || {
                black_box(JacobianPattern2::build_from_permuted(
                    cp, rows, mat.npv, mat.npq,
                ));
            },
        );
        baseline.full_build_and_drop = old_total;
        optimized.full_build_and_drop = new_total;
        let (old_lookup, new_lookup) = measure_pair(
            repeats,
            || lookup_columns::<false>(cp, rows, mat.npq, mat.npv),
            || lookup_columns::<true>(cp, rows, mat.npq, mat.npv),
        );
        baseline.lookup_only = old_lookup;
        optimized.lookup_only = new_lookup;
        let klu_symbolic_analysis = klu_symbolic(&new, if id == "9241" { 50 } else { 300 });
        let full_build_speedup =
            baseline.full_build_and_drop.mean_us / optimized.full_build_and_drop.mean_us;
        let lookup_speedup = baseline.lookup_only.mean_us / optimized.lookup_only.mean_us;
        println!(
            "{name}: full symbolic {:.3} -> {:.3} us ({full_build_speedup:.3}x), lookup {:.3} -> {:.3} us ({lookup_speedup:.3}x); KLU analyze {:.3} us; iterations {iterations}, voltage error {voltage_error:.2e}",
            baseline.full_build_and_drop.mean_us,
            optimized.full_build_and_drop.mean_us,
            baseline.lookup_only.mean_us,
            optimized.lookup_only.mean_us,
            klu_symbolic_analysis.mean_us
        );
        cases.push(Case {
            system: name,
            buses: mat.y_bus.ncols(),
            npq: mat.npq,
            npv: mat.npv,
            ybus_nnz: mat.y_bus.nnz(),
            jacobian_nnz: new.nnz_j,
            repeats_per_batch: repeats,
            variants: [baseline, optimized],
            klu_symbolic_analysis,
            full_build_speedup,
            lookup_speedup,
            identical_structure_and_numeric_jacobian: true,
            iterations,
            voltage_error,
        });
        let report = serde_json::json!({
            "rustpower_version": env!("CARGO_PKG_VERSION"),
            "batches": BATCHES, "warmup_calls": WARMUP,
            "scope": "Jacobian symbolic preparation; KLU symbolic analysis is measured separately",
            "timing_notes": [
                "Full build includes destruction; stage timers exclude destruction and have clock overhead.",
                "Angle-column time includes row boundaries, diagonal search and J11/J21 structure construction.",
                "Magnitude-column time includes J12/J22 construction using cached row boundaries.",
                "Lookup-only is a separate microbenchmark already included in angle-column time; do not add it.",
                "KLU analyze excludes context initialization and symbolic destruction.",
                "Production builds contain no profiling clocks."
            ],
            "cases": cases,
        });
        std::fs::write(
            output.join("symbolic_analysis.json"),
            serde_json::to_string_pretty(&report).unwrap(),
        )
        .unwrap();
    }
}
