//! Estimate serial numerical cost from ordinary cached ACPF on sampled outages.
use nalgebra::DVector;
use rustpower::prelude::{
    newtonpf::{NewtonCache, newton_pf},
    solver::KLUSolver,
};
use rustpower::{n1::*, prelude::*};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: bench_acpf_floor CSV_ZIP PIPELINE_RESULT_JSON OUTPUT_JSON".into());
    }
    let pipeline: Value =
        serde_json::from_str(&std::fs::read_to_string(&args[2]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let net = rustpower::io::pandapower::load_csv_zip(&args[1]).map_err(|e| e.to_string())?;
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    let prepared = prepare_n1(grid.world_mut())?;
    let base = prepared.base().baseline();
    let v = DVector::from_column_slice(base.ac_start().unwrap());
    let s = DVector::from_column_slice(base.s_bus());
    let candidates: Vec<_> = prepared
        .active_branches()
        .filter(|&i| {
            !prepared
                .topology()
                .bridge_cut(i)
                .is_some_and(|c| c.slackless_bus_count() > 0)
        })
        .collect();
    let mut samples = Vec::new();
    for position in [0, candidates.len() / 2, candidates.len() - 1] {
        let index = candidates[position];
        let task = rustpower::n1::outage::OutageTask::for_ac_branch(
            base,
            prepared.base().catalog(),
            index,
            0,
        )?;
        let mut y = base.y_bus().clone();
        for (&slot, &value) in task.y_patch().slots().iter().zip(task.y_patch().values()) {
            y.values_mut()[slot] = value;
        }
        let mut solver = KLUSolver::default();
        let mut cache = NewtonCache::default();
        let started = Instant::now();
        let first = newton_pf(
            &y,
            &s,
            &v,
            base.npv(),
            base.npq(),
            Some(1e-10),
            Some(30),
            &mut solver,
            Some(&mut cache),
        )
        .map_err(|e| e.0)?;
        let first_seconds = started.elapsed().as_secs_f64();
        let iterations = first.1;
        if iterations == 0 {
            return Err(format!(
                "sample {index} unexpectedly returned at zero iterations"
            ));
        }
        let symbol = solver.0.symbolic;
        let mut reference_worker = N1Worker::new_ac(base);
        rustpower::n1::outage::WorkerPatchState::default().apply(
            &mut reference_worker,
            base,
            &task,
        )?;
        let reference = reference_worker.solve_ac_current(base, 30, 1e-10);
        assert!(reference.converged);
        let voltage_difference = first
            .0
            .iter()
            .zip(&reference.voltage)
            .map(|(a, b)| (*a - *b).norm())
            .fold(0., f64::max);
        assert!(voltage_difference < 1e-7);
        let pattern = cache.j_pattern.as_ref().unwrap().j_col_ptrs.as_ptr();
        let mut times = Vec::new();
        for _ in 0..2001 {
            let start = Instant::now();
            let result = std::hint::black_box(newton_pf(
                &y,
                &s,
                &v,
                base.npv(),
                base.npq(),
                Some(1e-10),
                Some(30),
                &mut solver,
                Some(&mut cache),
            ))
            .map_err(|e| e.0)?;
            times.push(start.elapsed().as_secs_f64());
            assert_eq!(result.1, iterations);
            assert_eq!(solver.0.symbolic, symbol);
            assert_eq!(
                cache.j_pattern.as_ref().unwrap().j_col_ptrs.as_ptr(),
                pattern
            );
        }
        times.sort_by(f64::total_cmp);
        let identity = prepared.base().catalog().identities()[index];
        samples.push(json!({"outage_index":index,"outage_kind":format!("{:?}",identity.kind),"outage_id":identity.id,"iterations":iterations,"repeats":2001,"ordinary_acpf_seconds_median":times[1000],"ordinary_acpf_seconds_min":times[0],"ordinary_acpf_seconds_max":times[2000],"ordinary_seconds_per_iteration":times[1000]/iterations as f64,"first_acpf_including_pattern_symbolic_factor_seconds":first_seconds,"call_samples_sorted_seconds":times}));
    }
    let total_iters = pipeline["attempted_ac_iterations_samples"][0]
        .as_u64()
        .unwrap();
    let sample_time: f64 = samples
        .iter()
        .map(|r| r["ordinary_acpf_seconds_median"].as_f64().unwrap())
        .sum();
    let sample_iters: u64 = samples
        .iter()
        .map(|r| r["iterations"].as_u64().unwrap())
        .sum();
    let per_iteration = sample_time / sample_iters as f64;
    let estimate = per_iteration * total_iters as f64;
    let low = samples
        .iter()
        .map(|r| r["ordinary_seconds_per_iteration"].as_f64().unwrap())
        .fold(f64::INFINITY, f64::min)
        * total_iters as f64;
    let high = samples
        .iter()
        .map(|r| r["ordinary_seconds_per_iteration"].as_f64().unwrap())
        .fold(0., f64::max)
        * total_iters as f64;
    let wall = pipeline["run_wall_seconds_median"].as_f64().unwrap();
    let report = json!({"case":pipeline["case"],"buses":pipeline["buses"],"candidates":pipeline["candidates"],"actual_pf_cases":candidates.len(),"actual_pipeline_ac_iterations":total_iters,"pipeline_one_worker_wall_seconds_median":wall,"estimated_serial_numeric_seconds":estimate,"sample_cost_spread_estimate_seconds":[low,high],"ordinary_seconds_per_iteration_weighted":per_iteration,"pipeline_over_estimate_ratio":wall/estimate,"difference_seconds":wall-estimate,"initialization":"shared converged base AC voltage, applied to actual single-device outages","tolerance":1e-10,"max_iter":30,"timing_scope":"ordinary newton_pf entry to return, cached J pattern and KLU Symbolic/Numeric reused after first warm-up; matrix preparation and patching outside timing","interpretation":"sample-based serial numerical cost estimate; not a complete serial N-1 measurement or mathematical lower bound","samples":samples});
    let path = PathBuf::from(&args[3]);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
    println!(
        "case={} estimated_us={:.3} pipeline_us={:.3}",
        pipeline["case"],
        estimate * 1e6,
        wall * 1e6
    );
    Ok(())
}
