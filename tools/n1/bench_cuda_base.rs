//! Compare the native CUDA executor with ordinary Rust Newton on one unchanged base.
use nalgebra::DVector;
use rustpower::n1::{
    cuda::{CudaCaseStatus, CudaExecutor},
    prepare_base,
};
use rustpower::prelude::{
    newtonpf::{NewtonCache, newton_pf},
    solver::KLUSolver,
    *,
};
use serde_json::json;
use std::time::Instant;

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 || args.len() > 5 {
        return Err("usage: bench_cuda_base CSV_ZIP [BATCH=64] [STREAMS=1] [REPEATS=3]".into());
    }
    let batch: usize = args
        .get(2)
        .map_or(Ok(64), |s| s.parse())
        .map_err(|e| format!("batch: {e}"))?;
    let streams: usize = args
        .get(3)
        .map_or(Ok(1), |s| s.parse())
        .map_err(|e| format!("streams: {e}"))?;
    let repeats: usize = args
        .get(4)
        .map_or(Ok(3), |s| s.parse())
        .map_err(|e| format!("repeats: {e}"))?;
    if streams == 0 || repeats == 0 {
        return Err("streams and repeats must be positive".into());
    }
    let total_batch = batch
        .checked_mul(streams)
        .ok_or("total CUDA batch overflow")?;
    let net = rustpower::io::pandapower::load_csv_zip(&args[1]).map_err(|e| e.to_string())?;
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    let prepared = prepare_base(grid.world_mut())?;
    let base = prepared.baseline();
    let nbus = base.v_init().len();
    let tolerance = 1e-8;
    let max_iter = 30;

    let create_started = Instant::now();
    let mut executors = Vec::with_capacity(streams);
    for _ in 0..streams {
        executors.push(CudaExecutor::new(base, batch)?);
    }
    let create_seconds = create_started.elapsed().as_secs_f64();
    let solve_started = Instant::now();
    let stream_results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(streams);
        for (stream, mut executor) in executors.into_iter().enumerate() {
            handles.push(scope.spawn(move || {
                let mut seconds = Vec::with_capacity(repeats);
                let mut result = None;
                for _ in 0..repeats {
                    let started = Instant::now();
                    result = Some(executor.solve_base(max_iter, tolerance)?);
                    seconds.push(started.elapsed().as_secs_f64());
                }
                Ok::<_, String>((stream, seconds, result.expect("at least one repeat")))
            }));
        }
        let mut results = Vec::with_capacity(streams);
        for handle in handles {
            results.push(handle.join().map_err(|_| "CUDA worker panicked")??);
        }
        Ok::<_, String>(results)
    })?;
    let solve_seconds = solve_started.elapsed().as_secs_f64();

    let y = base.y_bus();
    let s = DVector::from_column_slice(base.s_bus());
    let v = DVector::from_column_slice(base.v_init());
    let mut solver = KLUSolver::default();
    let mut cache = NewtonCache::default();
    let reference = newton_pf(
        y,
        &s,
        &v,
        base.npv(),
        base.npq(),
        Some(tolerance),
        Some(max_iter),
        &mut solver,
        Some(&mut cache),
    )
    .map_err(|e| e.0)?;

    let mut max_voltage_error = 0.0_f64;
    let mut max_residual = 0.0_f64;
    let mut stream_seconds = Vec::with_capacity(streams);
    for (stream, seconds, result) in stream_results {
        stream_seconds.push(json!({"stream": stream, "repeat_seconds": seconds}));
        for (scenario, case) in result.cases.iter().enumerate() {
            if case.status != CudaCaseStatus::Converged {
                return Err(format!(
                    "GPU stream {stream} scenario {scenario} did not converge: {:?}",
                    case.status
                ));
            }
            if case.iterations != reference.1 {
                return Err(format!(
                    "iteration mismatch in stream {stream} scenario {scenario}: GPU {}, CPU {}",
                    case.iterations, reference.1
                ));
            }
            let voltage_error = case
                .voltage
                .iter()
                .zip(reference.0.iter())
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0, f64::max);
            max_voltage_error = max_voltage_error.max(voltage_error);
            max_residual = max_residual.max(case.residual);
        }
    }
    if max_voltage_error > 1e-8 {
        return Err(format!("CUDA/CPU voltage mismatch: {max_voltage_error}"));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "buses": nbus,
            "per_stream_batch": batch,
            "streams": streams,
            "total_batch": total_batch,
            "repeats": repeats,
            "iterations": reference.1,
            "tolerance": tolerance,
            "max_final_residual": max_residual,
            "max_voltage_abs_error": max_voltage_error,
            "executor_create_seconds_including_analysis": create_seconds,
            "concurrent_solve_seconds": solve_seconds,
            "complete_base_flows_per_second": (total_batch * repeats) as f64 / solve_seconds,
            "streams_detail": stream_seconds,
            "initial_voltage": "Rust SharedBaseline::v_init, identical for every scenario",
            "backend": "FP64 CUDA kernels + FP64 cuDSS, one direct executor per stream"
        }))
        .unwrap()
    );
    Ok(())
}
