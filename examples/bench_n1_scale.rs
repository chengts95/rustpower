//! Basic ACPF scalability through the entity-oriented Rust study API.
use rustpower::{n1::*, prelude::*};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 6 {
        return Err(
            "usage: bench_n1_scale CASE_LABEL CSV_ZIP WORKERS REPEATS OUTPUT_JSON [BATCH=8]".into(),
        );
    }
    let workers = args[3].parse::<usize>().map_err(|e| e.to_string())?;
    let repeats = args[4].parse::<usize>().map_err(|e| e.to_string())?;
    let batch_size = args
        .get(6)
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(8);
    if repeats == 0 {
        return Err("repeats must be positive".into());
    }
    let output = PathBuf::from(&args[5]);
    let call_started = Instant::now();
    let model_started = Instant::now();
    let net = rustpower::io::pandapower::load_csv_zip(&args[2]).map_err(|e| e.to_string())?;
    let buses = net.bus.len();
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    let model_seconds = model_started.elapsed().as_secs_f64();
    let prepare_started = Instant::now();
    let prepared = prepare_n1(grid.world_mut())?;
    let prepare_seconds = prepare_started.elapsed().as_secs_f64();
    let api_started = Instant::now();
    // This sweep measures basic ACPF convergence with identical empty constraint
    // plans for every network. Original device limits are preserved in the World.
    let study = prepared.study().map_err(|e| e.to_string())?;
    let faults = study.all_branch_outages();
    let study_seconds = api_started.elapsed().as_secs_f64();
    let options = N1RunOptions {
        config: N1Config {
            workers,
            batch_size,
            max_iter: 30,
            tolerance: 1e-10,
            reporting: N1Reporting::StatusOnly,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut samples = Vec::with_capacity(repeats);
    let mut iteration_samples = Vec::with_capacity(repeats);
    let mut reference = None;
    let mut first_call_seconds = 0.;
    let mut terminal_counts = None;
    let mut last_events = BTreeMap::new();
    let mut last_islands = Vec::new();
    for repeat in 0..repeats {
        let mut events = BTreeMap::new();
        let started = Instant::now();
        let report = study
            .run(&faults, options, |batch| {
                for event in batch.events {
                    assert!(
                        events
                            .insert(
                                event.outage_index,
                                (format!("{:?}", event.outcome), format!("{:?}", event.stage))
                            )
                            .is_none()
                    );
                }
            })
            .map_err(|e| e.to_string())?;
        let seconds = started.elapsed().as_secs_f64();
        if repeat == 0 {
            first_call_seconds = call_started.elapsed().as_secs_f64();
        }
        assert_eq!(report.processed(), faults.len());
        assert_eq!(
            report.passed() + report.violated() + report.numerical_failed() + report.islanded(),
            faults.len()
        );
        let islands: Vec<_> = report
            .islands
            .iter()
            .map(|case| case.outage_index)
            .collect();
        let counts = (
            report.passed(),
            report.violated(),
            report.numerical_failed(),
            report.islanded(),
        );
        let signature = (events.clone(), islands.clone(), counts);
        if let Some(expected) = &reference {
            assert_eq!(&signature, expected, "repeat changed a scenario decision");
        } else {
            reference = Some(signature);
        }
        samples.push(seconds);
        iteration_samples.push(report.solve.attempted_iterations);
        terminal_counts = Some(counts);
        last_events = events;
        last_islands = islands;
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.
    } else {
        sorted[sorted.len() / 2]
    };
    let (passed, violated, failed, islanded) = terminal_counts.unwrap();
    let result = json!({
        "case": args[1], "input": args[2], "buses":buses, "workers":workers, "repeats":repeats,
        "actual_cpu_binding": std::fs::read_to_string("/proc/self/status").ok().and_then(|s|s.lines().find(|line|line.starts_with("Cpus_allowed_list:")).map(str::to_owned)),
        "batch_size":batch_size,"initialization":"BaseAc","reporting":"StatusOnly","watch_policy":"basic_acpf_convergence","constraint_rules":prepared.watch().rule_count(),
        "max_iter":30,"tolerance":1e-10,"candidates":faults.len(),"numerically_solved":faults.len()-islanded,
        "passed":passed,"violated":violated,"numerical_failed":failed,"islanded":islanded,
        "model_load_and_world_init_seconds":model_seconds,"shared_preparation_seconds":prepare_seconds,
        "study_validation_and_fault_set_seconds":study_seconds,"first_call_end_to_end_seconds":first_call_seconds,
        "run_wall_seconds_median":median,"run_wall_seconds_min":sorted[0],"run_wall_seconds_max":sorted[sorted.len()-1],
        "run_wall_seconds_samples":samples,"attempted_ac_iterations_samples":iteration_samples,
        "base_ac_iterations":prepared.base_solution().map(|s|s.iterations),
        "rejected_scenario_statuses":last_events,"island_outage_indices":last_islands,
        "timing_scope":"study.run entry through classification, scoped worker creation, PF, configured screening, collector and all joins; JSON write excluded"
    });
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap())
        .map_err(|e| e.to_string())?;
    println!(
        "case={} buses={buses} workers={workers} repeats={repeats} scenarios={} solved={} islands={islanded} failed={failed} median={median:.6}s",
        args[1],
        faults.len(),
        faults.len() - islanded
    );
    Ok(())
}
