//! Complete contingency batches; numerical stage clocks + end-to-end wall clock.
#[path = "common/n1.rs"]
mod common_n1;
use rustpower::{n1::*, prelude::*};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    time::Instant,
};
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let case = args.get(1).map(String::as_str).unwrap_or("39");
    let workers: Vec<usize> = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("1,2,4")
        .split(',')
        .map(|s| s.parse().map_err(|e| format!("workers: {e}")))
        .collect::<Result<_, _>>()?;
    let batch_size = args
        .get(3)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("batch: {e}"))?
        .unwrap_or(8);
    let out = PathBuf::from(args.get(4).ok_or(
        "usage: bench_n1 CASE WORKERS_CSV BATCH OUTDIR [base-ac|dc] [status|details] [bridges|all]",
    )?);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let initialization = match args.get(5).map(String::as_str).unwrap_or("base-ac") {
        "base-ac" => N1Initialization::BaseAc,
        "dc" => N1Initialization::DcPerCase,
        _ => return Err("initialization must be base-ac or dc".into()),
    };
    let reporting = match args.get(6).map(String::as_str).unwrap_or("status") {
        "status" => N1Reporting::StatusOnly,
        "details" => N1Reporting::Detailed,
        _ => return Err("reporting must be status or details".into()),
    };
    let screen_bridges = match args.get(7).map(String::as_str).unwrap_or("bridges") {
        "bridges" => true,
        "all" => false,
        _ => return Err("topology screening must be bridges or all".into()),
    };
    let init = Instant::now();
    let net = match case {
        "39" => serde_json::from_str::<rustpower::io::pandapower::Network>(
            rustpower::testcases::case_ieee39::IEEE_39,
        )
        .map_err(|e| e.to_string())?,
        "9241" => rustpower::io::pandapower::load_csv_zip(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/cases/pegase9241/data.zip"
        ))
        .map_err(|e| e.to_string())?,
        _ => return Err("case must be 39 or 9241".into()),
    };
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    let model_seconds = init.elapsed().as_secs_f64();
    common_n1::install_watches(grid.world_mut());
    let prep = Instant::now();
    let prepared = prepare_n1_with_initialization(grid.world_mut(), initialization)?;
    let preparation_seconds = prep.elapsed().as_secs_f64();
    let check = Instant::now();
    common_n1::check_baseline(&prepared)?;
    let baseline_check_seconds = check.elapsed().as_secs_f64();
    println!(
        "BASELINE line_loading+bus_voltage violations=0; transformer loading monitoring disabled"
    );
    let candidates = prepared.active_branches().count();
    println!(
        "START case={case} buses={} candidates={candidates} shared_prepare={preparation_seconds:.6}s",
        prepared.base().baseline().v_init().len()
    );
    let mut runs = Vec::<Value>::new();
    let mut baseline = None::<(f64, usize, f64, BTreeMap<usize, String>)>;
    for p in workers {
        let mut records = Vec::<Value>::new();
        let mut failed_cases = BTreeMap::<usize, String>::new();
        let mut max_loading = 0.0_f64;
        let mut generated = 0usize;
        let mut islands = Vec::new();
        let selected = prepared
            .active_branches()
            .inspect(|_| {
                generated += 1;
                if generated % 1000 == 0 {
                    eprintln!(
                        "PROGRESS case={case} workers={p} generated={generated}/{candidates}"
                    );
                }
            })
            .filter(|&index| {
                if screen_bridges {
                    if let Some(cut) = prepared.topology().bridge_cut(index) {
                        if cut.slackless_bus_count() > 0 {
                            islands.push((index, cut));
                            return false;
                        }
                    }
                }
                true
            });
        let start = Instant::now();
        let summary = run_n1(
            &prepared,
            N1Config {
                workers: p,
                batch_size,
                reporting,
                ..Default::default()
            },
            selected,
            |batch| {
                for event in batch.events {
                    if event.stage != SolveStage::Converged {
                        failed_cases.insert(event.outage_index, format!("{:?}", event.stage));
                    }
                    let mut worst = 0.0_f64;
                    let mut worst_branch = None;
                    let mut voltage_hits = 0;
                    let mut voltage_bus_ids = Vec::new();
                    let mut min_vm = None::<f64>;
                    let mut max_vm = None::<f64>;
                    for (hit_index, hit) in event.hits.iter().enumerate() {
                        if let Some((_, id)) = hit.target.bus() {
                            voltage_hits += 1;
                            voltage_bus_ids.push(id);
                            if let Some(value) = *hit.value {
                                min_vm = Some(min_vm.map_or(value, |v| v.min(value)));
                                max_vm = Some(max_vm.map_or(value, |v| v.max(value)));
                            }
                        }
                        if let Some(flow) = event.hits.branch_for_hit(hit_index) {
                            if flow.loading_percent > worst {
                                worst = flow.loading_percent;
                                let (_, identity) =
                                    hit.target.branch().expect("branch result target");
                                worst_branch = Some(format!("{:?}:{}", identity.kind, identity.id));
                            }
                        }
                    }
                    max_loading = max_loading.max(worst);
                    records.push(json!({"case_id":event.case_id,"outage_index":event.outage_index,"outage_kind":format!("{:?}",event.outage.kind),"outage_id":event.outage.id,"stage":format!("{:?}",event.stage),"outcome":format!("{:?}",event.outcome),"iterations":event.iterations,"residual":event.residual,"hits":event.hits.len(),"violations":(reporting==N1Reporting::Detailed).then(||event.hits.violation_count()),"voltage_hits":(reporting==N1Reporting::Detailed).then_some(voltage_hits),"voltage_bus_ids":(reporting==N1Reporting::Detailed).then_some(voltage_bus_ids),"min_violating_vm":min_vm,"max_violating_vm":max_vm,"worst_loading_percent":(reporting==N1Reporting::Detailed).then_some(worst),"worst_branch":worst_branch,"error":event.error,"factor_calls":event.linear.factor_calls,"refactor_calls":event.linear.refactor_calls,"refactor_solve_retries":event.linear.refactor_solve_retries}));
                }
            },
        )?;
        let wall_seconds = start.elapsed().as_secs_f64();
        assert_eq!(summary.submitted + islands.len(), candidates);
        assert_eq!(summary.processed, summary.submitted);
        assert_eq!(failed_cases.len(), summary.failed);
        records.sort_by_key(|r| r["case_id"].as_u64().unwrap());
        let path = out.join(format!("case{case}-workers{p}-events.json"));
        std::fs::write(path, serde_json::to_vec_pretty(&records).unwrap())
            .map_err(|e| e.to_string())?;
        let island_records: Vec<_> = islands.iter().map(|(index, cut)| {
            let identity = prepared.base().catalog().identities()[*index];
            json!({"outage_index": index, "outage_kind": format!("{:?}", identity.kind), "outage_id": identity.id,
                "outcome": "Islanded", "slackless_buses": cut.slackless_bus_count(),
                "subtree_buses": cut.subtree_bus_count(), "other_buses": cut.other_bus_count(),
                "subtree_slacks": cut.subtree_slacks, "other_slacks": cut.other_slacks})
        }).collect();
        std::fs::write(
            out.join(format!("case{case}-workers{p}-islands.json")),
            serde_json::to_vec_pretty(&island_records).unwrap(),
        )
        .map_err(|e| e.to_string())?;
        if p == 1 {
            baseline = Some((
                wall_seconds,
                summary.attempted_iterations,
                summary.ac_iteration_time.as_secs_f64(),
                failed_cases.clone(),
            ));
        }
        let mut row = json!({"case":case,"workers":p,"batch_size":batch_size,"candidates":candidates,"submitted":summary.submitted,"processed":summary.processed,"failed":summary.failed,"dc_failed":summary.dc_failed,"newton_failed":summary.newton_failed,"completed_ac_iterations":summary.total_iterations,"attempted_ac_iterations":summary.attempted_iterations,"run_wall_seconds":wall_seconds,"throughput_per_second":summary.processed as f64/wall_seconds,"shared_preparation_seconds":preparation_seconds,"baseline_check_seconds":baseline_check_seconds,"baseline_violations":0,"loading_watch_scope":"lines_and_bus_voltage","initialization":format!("{initialization:?}"),"base_ac_iterations":prepared.base_solution().map(|s|s.iterations),"base_dc_seconds":prepared.base_solution().map(|s|s.timings.dc.as_secs_f64()),"base_ac_seconds":prepared.base_solution().map(|s|s.timings.ac.as_secs_f64()),"first_batch_from_initialized_world_seconds":preparation_seconds+baseline_check_seconds+wall_seconds,"input_load_and_main_world_init_seconds":model_seconds,"summed_dc_seconds":summary.dc_time.as_secs_f64(),"summed_ac_seconds":summary.ac_time.as_secs_f64(),"summed_ac_iteration_seconds":summary.ac_iteration_time.as_secs_f64(),"summed_patch_seconds":summary.patch_time.as_secs_f64(),"summed_watch_seconds":summary.watch_time.as_secs_f64(),"emitted_cases":summary.emitted_cases,"emitted_hits":summary.emitted_hits,"reporting":format!("{reporting:?}"),"passed":summary.passed(),"rejected":summary.rejected(),"violated_cases":summary.violated_cases,"violation_hits":(reporting==N1Reporting::Detailed).then_some(summary.violation_hits),"factor_calls":summary.linear.factor_calls,"refactor_calls":summary.linear.refactor_calls,"refactor_solve_retries":summary.linear.refactor_solve_retries,"worst_loading_percent":(reporting==N1Reporting::Detailed).then_some(max_loading)});
        row["topology_screening"] = json!(screen_bridges);
        row["islanded_cases"] = json!(islands.len());
        row["solver_rejected"] = json!(summary.rejected());
        row["rejected"] = json!(summary.rejected() + islands.len());
        row["classified"] = json!(summary.processed + islands.len());
        row["classification_throughput_per_second"] = json!(candidates as f64 / wall_seconds);
        if let Some((t1, i1, ac1, f1)) = &baseline {
            let pipeline_iteration = t1 / (*i1 as f64);
            let pure_iteration = ac1 / (*i1 as f64);
            let estimate = summary.attempted_iterations as f64 * pipeline_iteration;
            row["wall_speedup"] = json!(t1 / wall_seconds);
            row["wall_parallel_efficiency"] = json!(t1 / (wall_seconds * p as f64));
            row["single_worker_amortized_seconds_per_ac_attempt"] = json!(pipeline_iteration);
            row["single_worker_ac_seconds_per_attempt"] = json!(pure_iteration);
            row["iteration_normalized_serial_seconds"] = json!(estimate);
            row["iteration_normalized_speedup"] = json!(estimate / wall_seconds);
            row["iteration_normalized_efficiency"] = json!(estimate / (wall_seconds * p as f64));
            let keys: BTreeSet<_> = f1.keys().chain(failed_cases.keys()).copied().collect();
            let changed: Vec<_> = keys
                .into_iter()
                .filter(|id| f1.get(id) != failed_cases.get(id))
                .collect();
            row["failure_status_changes_from_single_worker"] = json!(changed);
        }
        println!("DONE {}", row);
        runs.push(row);
        std::fs::write(
            out.join(format!("case{case}-summary.json")),
            serde_json::to_vec_pretty(&runs).unwrap(),
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}
