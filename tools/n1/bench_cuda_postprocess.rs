//! Validate complete GPU two-port results against the authoritative CPU catalog.
use rustpower::n1::{
    BranchFlowResult,
    cuda::{CudaBranchCatalog, CudaCaseStatus, CudaExecutor},
    prepare_base,
};
use rustpower::prelude::*;
use serde_json::json;
use std::time::Instant;

const FIELDS: [&str; 14] = [
    "p_from_mw",
    "q_from_mvar",
    "p_to_mw",
    "q_to_mvar",
    "pl_mw",
    "ql_mvar",
    "i_from_ka",
    "i_to_ka",
    "i_ka",
    "vm_from_pu",
    "vm_to_pu",
    "va_from_degree",
    "va_to_degree",
    "loading_percent",
];

fn values(value: &BranchFlowResult) -> [f64; 14] {
    [
        value.p_from_mw,
        value.q_from_mvar,
        value.p_to_mw,
        value.q_to_mvar,
        value.pl_mw,
        value.ql_mvar,
        value.i_from_ka,
        value.i_to_ka,
        value.i_ka,
        value.vm_from_pu,
        value.vm_to_pu,
        value.va_from_degree,
        value.va_to_degree,
        value.loading_percent,
    ]
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 || args.len() > 4 {
        return Err("usage: bench_cuda_postprocess CSV_ZIP [BATCH=64] [REPEATS=10]".into());
    }
    let batch: usize = args
        .get(2)
        .map_or(Ok(64), |s| s.parse())
        .map_err(|e| format!("batch: {e}"))?;
    let repeats: usize = args
        .get(3)
        .map_or(Ok(10), |s| s.parse())
        .map_err(|e| format!("repeats: {e}"))?;
    if batch == 0 || repeats == 0 {
        return Err("batch and repeats must be positive".into());
    }
    let net = rustpower::io::pandapower::load_csv_zip(&args[1]).map_err(|e| e.to_string())?;
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    let prepared = prepare_base(grid.world_mut())?;
    let catalog = prepared.catalog();
    let mut executor = CudaExecutor::new(prepared.baseline(), batch)?;
    let cuda_catalog = CudaBranchCatalog::new(catalog)?;
    let solved = executor.solve_base(30, 1e-8)?;
    if solved
        .cases
        .iter()
        .any(|case| case.status != CudaCaseStatus::Converged)
    {
        return Err("cannot validate postprocessing from a non-converged base case".into());
    }
    // Outages deliberately affect only postprocessing here: this is a formula test,
    // not yet an N-1 network solve. Every seventh scenario zeroes one selected branch.
    let outages: Vec<_> = (0..batch)
        .map(|scenario| (scenario % 7 == 3).then_some((scenario * 13) % catalog.len()))
        .collect();
    let gpu = executor.postprocess_branches(&cuda_catalog, &outages)?;
    let mut expected = Vec::new();
    let mut max_abs = [0.0_f64; 14];
    let mut max_relative = [0.0_f64; 14];
    let mut worst = (0.0, 0, 0, 0);
    for (scenario, case) in solved.cases.iter().enumerate() {
        catalog.compute(
            &case.voltage,
            outages[scenario],
            catalog.sbase(),
            &mut expected,
        )?;
        let actual = gpu.scenario(scenario).ok_or("missing GPU scenario")?;
        for branch in 0..catalog.len() {
            let must_be_zero = outages[scenario] == Some(branch) || !catalog.in_service()[branch];
            for field in 0..14 {
                let (a, b) = (
                    values(&actual[branch])[field],
                    values(&expected[branch])[field],
                );
                if must_be_zero && a != 0.0 {
                    return Err(format!(
                        "outage/service zero semantics failed at {scenario}/{branch}"
                    ));
                }
                let error = (a - b).abs();
                if !error.is_finite() || error > 1e-8 * b.abs().max(1.0) {
                    return Err(format!(
                        "{} mismatch at scenario {scenario}, branch {branch}: GPU {a}, CPU {b}",
                        FIELDS[field]
                    ));
                }
                max_abs[field] = max_abs[field].max(error);
                if b.abs() > 1e-12 {
                    max_relative[field] = max_relative[field].max(error / b.abs());
                }
                if error > worst.0 {
                    worst = (error, scenario, branch, field);
                }
            }
        }
    }

    let cpu_started = Instant::now();
    for _ in 0..repeats {
        for (scenario, case) in solved.cases.iter().enumerate() {
            catalog.compute(
                &case.voltage,
                outages[scenario],
                catalog.sbase(),
                &mut expected,
            )?;
        }
    }
    let cpu_seconds = cpu_started.elapsed().as_secs_f64();
    let mut gpu_seconds = 0.0;
    let mut kernel_seconds = 0.0;
    let mut download_seconds = 0.0;
    for _ in 0..repeats {
        let started = Instant::now();
        let result = executor.postprocess_branches(&cuda_catalog, &outages)?;
        gpu_seconds += started.elapsed().as_secs_f64();
        kernel_seconds += result.kernel_seconds;
        download_seconds += result.download_seconds;
    }
    let results = (batch * catalog.len() * repeats) as f64;
    println!("{}", serde_json::to_string_pretty(&json!({
        "buses": prepared.baseline().v_init().len(),
        "branches": catalog.len(),
        "batch": batch,
        "repeats": repeats,
        "field_order": FIELDS,
        "max_abs_error": max_abs,
        "max_relative_error_for_reference_above_1e-12": max_relative,
        "worst_absolute_error": {
            "value": worst.0, "scenario": worst.1, "branch": worst.2, "field": FIELDS[worst.3]
        },
        "cpu_single_thread_seconds": cpu_seconds,
        "gpu_complete_seconds": gpu_seconds,
        "gpu_kernel_seconds": kernel_seconds,
        "gpu_download_seconds": download_seconds,
        "cpu_results_per_second": results / cpu_seconds,
        "gpu_complete_results_per_second": results / gpu_seconds,
        "complete_gpu_over_cpu": cpu_seconds / gpu_seconds,
        "outage_stride": 7,
        "validation": "all scenarios, all branches, all 14 fields; outage and out-of-service outputs exactly zero"
    })).unwrap());
    Ok(())
}
