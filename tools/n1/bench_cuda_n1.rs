//! Numerical and end-to-end comparison of real single-branch N-1 sweeps.
use bevy_ecs::prelude::{Entity, Has, Or, With, Without};
use rustpower::n1::{
    N1Config, N1FaultSet, N1RunOptions, N1Watch, N1Worker, WatchCondition, WatchRule,
    cuda::{CudaCaseStatus, CudaExecutor, CudaN1Config, run_n1_cuda, run_n1_study_cuda},
    outage::{OutageTask, WorkerPatchState},
    prepare_n1, run_n1,
};
use rustpower::prelude::{
    ecs::elements::{
        bus::{BusID, OutOfService, VmLimit},
        line::LineID,
        trans::TransformerID,
        units::PerUnit,
    },
    *,
};
use serde_json::json;
use std::time::Instant;

fn install_watches(world: &mut bevy_ecs::world::World) {
    let devices: Vec<_> = world
        .query_filtered::<(Entity, Has<LineID>), Or<(With<LineID>, With<TransformerID>)>>()
        .iter(world)
        .collect();
    for (index, (entity, line)) in devices.into_iter().enumerate() {
        let mut rules = Vec::new();
        if line {
            rules.push(WatchRule {
                rule_id: 1,
                condition: WatchCondition::BranchLoadingAbove(100.0),
            });
        }
        if index == 0 {
            rules.push(WatchRule {
                rule_id: 2,
                condition: WatchCondition::SolveFailed,
            });
        }
        if !rules.is_empty() {
            world.entity_mut(entity).insert(N1Watch { rules });
        }
    }
    let buses: Vec<_> = world
        .query_filtered::<Entity, (With<BusID>, With<VmLimit<PerUnit>>, Without<OutOfService>)>()
        .iter(world)
        .collect();
    for entity in buses {
        world.entity_mut(entity).insert(N1Watch {
            rules: vec![WatchRule {
                rule_id: 3,
                condition: WatchCondition::BusVoltageOutsideLimits,
            }],
        });
    }
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 || args.len() > 7 {
        return Err(
            "usage: bench_cuda_n1 CSV_ZIP [BATCH=64] [CPU_WORKERS=1] [LIMIT=256] [STREAMS=1] [REPEAT=1]"
                .into(),
        );
    }
    let batch: usize = args
        .get(2)
        .map_or(Ok(64), |s| s.parse())
        .map_err(|e| format!("batch: {e}"))?;
    let workers: usize = args
        .get(3)
        .map_or(Ok(1), |s| s.parse())
        .map_err(|e| format!("workers: {e}"))?;
    let limit: usize = args
        .get(4)
        .map_or(Ok(256), |s| s.parse())
        .map_err(|e| format!("limit: {e}"))?;
    let streams: usize = args
        .get(5)
        .map_or(Ok(1), |s| s.parse())
        .map_err(|e| format!("streams: {e}"))?;
    let repeat: usize = args
        .get(6)
        .map_or(Ok(1), |s| s.parse())
        .map_err(|e| format!("repeat: {e}"))?;
    if batch == 0 || workers == 0 || limit == 0 || streams == 0 || repeat == 0 {
        return Err("batch, workers, limit, streams and repeat must be positive".into());
    }
    let net = rustpower::io::pandapower::load_csv_zip(&args[1]).map_err(|e| e.to_string())?;
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    install_watches(grid.world_mut());
    let prepared = prepare_n1(grid.world_mut())?;
    let study = prepared.study().map_err(|e| e.to_string())?;
    let all_faults = study.all_branch_outages();
    let faults = N1FaultSet::out_of_service(all_faults.entities().iter().copied().take(limit));
    let (candidates, islands) = study.partition(&faults).map_err(|e| e.to_string())?;
    if candidates.is_empty() {
        return Err("no numerical N-1 candidates after island screening".into());
    }
    let base = prepared.base().baseline();
    let catalog = prepared.base().catalog();
    let validation_count = batch.min(candidates.len());
    let mut tasks = Vec::with_capacity(batch);
    for (case_id, &branch) in candidates.iter().take(validation_count).enumerate() {
        tasks.push(OutageTask::for_ac_branch(base, catalog, branch, case_id)?);
    }
    while tasks.len() < batch {
        tasks.push(tasks[0]);
    }
    let mut cuda = CudaExecutor::new_ac(base, batch)?;
    let gpu = cuda.solve_outages(&tasks, 30, 1e-10)?;
    let mut cpu = N1Worker::new_ac(base);
    let mut patches = WorkerPatchState::default();
    let mut max_voltage_error = 0.0_f64;
    let mut compared = 0_usize;
    for scenario in 0..validation_count {
        patches.apply(&mut cpu, base, &tasks[scenario])?;
        let reference = cpu.solve_ac_current(base, 30, 1e-10);
        let actual = &gpu.cases[scenario];
        let gpu_converged = actual.status == CudaCaseStatus::Converged;
        if reference.converged != gpu_converged {
            return Err(format!(
                "convergence mismatch in validation scenario {scenario}"
            ));
        }
        if reference.converged && reference.iterations != actual.iterations {
            return Err(format!(
                "iteration mismatch in converged scenario {scenario}: GPU {}, CPU {}",
                actual.iterations, reference.iterations
            ));
        }
        if reference.converged {
            let error = reference
                .voltage
                .iter()
                .zip(&actual.voltage)
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0, f64::max);
            if !error.is_finite() || error > 1e-8 {
                return Err(format!("voltage mismatch in scenario {scenario}: {error}"));
            }
            max_voltage_error = max_voltage_error.max(error);
            compared += 1;
        }
    }

    let work_len = candidates
        .len()
        .checked_mul(repeat)
        .ok_or("repeated scenario count overflow")?;
    let work: Vec<_> = candidates.iter().copied().cycle().take(work_len).collect();
    let cuda_config = CudaN1Config {
        batch_size: batch,
        streams,
        ..Default::default()
    };
    let gpu_started = Instant::now();
    let (gpu_summary, gpu_islanded) = if repeat == 1 {
        let report =
            run_n1_study_cuda(&study, &faults, cuda_config, |_| {}).map_err(|e| e.to_string())?;
        let islanded = report.islands.len();
        (report.solve, islanded)
    } else {
        (
            run_n1_cuda(&prepared, cuda_config, work.iter().copied(), |_| {})?,
            islands.len(),
        )
    };
    let gpu_seconds = gpu_started.elapsed().as_secs_f64();
    let cpu_config = N1Config {
        workers,
        batch_size: batch.min(32),
        ..Default::default()
    };
    let cpu_started = Instant::now();
    let (cpu_summary, cpu_islanded) = if repeat == 1 {
        let report = study
            .run(
                &faults,
                N1RunOptions {
                    config: cpu_config,
                    ..Default::default()
                },
                |_| {},
            )
            .map_err(|e| e.to_string())?;
        let islanded = report.islands.len();
        (report.solve, islanded)
    } else {
        (
            run_n1(&prepared, cpu_config, work.iter().copied(), |_| {})?,
            islands.len(),
        )
    };
    let cpu_seconds = cpu_started.elapsed().as_secs_f64();
    if (
        gpu_summary.processed,
        gpu_summary.failed,
        gpu_summary.violated_cases,
        gpu_islanded,
    ) != (
        cpu_summary.processed,
        cpu_summary.failed,
        cpu_summary.violated_cases,
        cpu_islanded,
    ) {
        return Err(format!(
            "full-sweep summary mismatch: GPU {:?}, CPU {:?}",
            (
                gpu_summary.processed,
                gpu_summary.failed,
                gpu_summary.violated_cases,
                gpu_islanded,
            ),
            (
                cpu_summary.processed,
                cpu_summary.failed,
                cpu_summary.violated_cases,
                cpu_islanded,
            )
        ));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "buses": base.v_init().len(),
            "branches": catalog.len(),
            "faults": faults.len(),
            "unique_numerical_candidates": candidates.len(),
            "numerical_scenarios": work.len(),
            "repeat": repeat,
            "synthetic_repeated_scenarios": repeat > 1,
            "islanded": islands.len(),
            "batch": batch,
            "gpu_streams": streams,
            "cpu_workers": workers,
            "validation": {
                "cases": validation_count,
                "converged_cases_compared": compared,
                "max_voltage_abs_error": max_voltage_error,
                "iterations": "exact match"
            },
            "gpu": {
                "seconds_including_executor_creation": gpu_seconds,
                "numerical_scenarios_per_second": work.len() as f64 / gpu_seconds,
                "processed": gpu_summary.processed,
                "failed": gpu_summary.failed,
                "violated": gpu_summary.violated_cases,
                "islanded": gpu_islanded,
                "workers_started": gpu_summary.workers_started,
                "mean_newton_iterations": gpu_summary.total_iterations as f64 / gpu_summary.processed as f64,
                "solve_seconds_sum_over_streams": gpu_summary.ac_time.as_secs_f64(),
                "watch_seconds_sum_over_streams": gpu_summary.watch_time.as_secs_f64()
            },
            "cpu": {
                "seconds": cpu_seconds,
                "numerical_scenarios_per_second": work.len() as f64 / cpu_seconds,
                "processed": cpu_summary.processed,
                "failed": cpu_summary.failed,
                "violated": cpu_summary.violated_cases,
                "islanded": cpu_islanded,
                "workers_started": cpu_summary.workers_started,
                "mean_newton_iterations": cpu_summary.total_iterations as f64 / cpu_summary.processed as f64,
                "solve_seconds_sum_over_workers": cpu_summary.ac_time.as_secs_f64(),
                "watch_seconds_sum_over_workers": cpu_summary.watch_time.as_secs_f64()
            },
            "gpu_over_cpu_end_to_end": cpu_seconds / gpu_seconds,
            "initialization": "BaseAc",
            "reporting": "StatusOnly with the same prepared WatchPlan and N1Study island screening"
        }))
        .unwrap()
    );
    Ok(())
}
