//! Retry numerical failures from an event file with per-scenario DCPF initialization.
#[path = "common/n1.rs"]
mod common_n1;
use rustpower::{n1::*, prelude::*};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 {
        return Err(
            "usage: n1_retry_dc CASE INPUT_EVENTS_JSON OUTPUT_JSON [workers] [max_iter]".into(),
        );
    }
    let case = &args[1];
    let output = PathBuf::from(&args[3]);
    let workers = args
        .get(4)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("workers: {e}"))?
        .unwrap_or(4);
    let max_iter = args
        .get(5)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("max_iter: {e}"))?
        .unwrap_or(30);
    let input: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(&args[2]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let mut selected = BTreeMap::new();
    for record in input
        .into_iter()
        .filter(|row| row["outcome"] == "NumericalFailure")
    {
        let index = record["outage_index"]
            .as_u64()
            .and_then(|i| usize::try_from(i).ok())
            .ok_or("failure record needs a valid outage_index")?;
        if selected.insert(index, record).is_some() {
            return Err(format!("duplicate failure record for outage_index {index}"));
        }
    }
    if selected.is_empty() {
        return Err("input contains no numerical failure records".into());
    }
    let started = Instant::now();
    let net = common_n1::load_case(case)?;
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    common_n1::install_watches(grid.world_mut());
    let prepared = prepare_n1(grid.world_mut())?;
    let study = prepared.study().map_err(|e| e.to_string())?;
    let catalog = prepared.base().catalog();
    for (&index, record) in &selected {
        let identity = catalog
            .identities()
            .get(index)
            .ok_or_else(|| format!("outage_index {index} is outside this case"))?;
        if record["outage_id"].as_i64() != Some(identity.id)
            || record["outage_kind"].as_str() != Some(format!("{:?}", identity.kind).as_str())
        {
            return Err(format!(
                "outage_index {index} does not match the recorded device identity"
            ));
        }
        if !catalog.in_service()[index] {
            return Err(format!("outage_index {index} is inactive in this baseline"));
        }
        if prepared
            .topology()
            .bridge_cut(index)
            .is_some_and(|cut| cut.slackless_bus_count() > 0)
        {
            return Err(format!(
                "outage_index {index} belongs to the island handling path"
            ));
        }
    }
    let preparation_seconds = started.elapsed().as_secs_f64();
    let mut results = Vec::new();
    let retry_started = Instant::now();
    let faults = N1FaultSet::out_of_service(
        selected
            .keys()
            .map(|&index| catalog.source_entities()[index]),
    );
    let report = study.run(
        &faults,
        N1RunOptions { config: N1Config {
            workers,
            batch_size: 1,
            max_iter,
            ..Default::default()
        }, initialization: Some(N1Initialization::DcPerCase), emit_all: true },
        |batch| {
            for event in batch.events {
                results.push(json!({
                "outage_index": event.outage_index, "outage_kind": format!("{:?}", event.outage.kind), "outage_id": event.outage.id,
                "original": selected[&event.outage_index],
                "dc_retry": {"stage": format!("{:?}", event.stage), "outcome": format!("{:?}", event.outcome), "iterations": event.iterations,
                    "residual": event.residual, "error": event.error, "factor_calls": event.linear.factor_calls, "refactor_calls": event.linear.refactor_calls,
                    "refactor_solve_retries": event.linear.refactor_solve_retries, "violation_hits": event.hits.violation_count()}
            }));
            }
        },
    ).map_err(|e| e.to_string())?;
    let summary = report.solve;
    let retry_wall_seconds = retry_started.elapsed().as_secs_f64();
    assert_eq!(summary.processed, selected.len());
    assert_eq!(results.len(), selected.len());
    results.sort_by_key(|row| row["outage_index"].as_u64().unwrap());
    assert_eq!(
        results
            .iter()
            .map(|row| row["outage_index"].as_u64().unwrap() as usize)
            .collect::<Vec<_>>(),
        selected.keys().copied().collect::<Vec<_>>()
    );
    let report = json!({"case":case,"input_events":args[2],"initialization":"DcPerCase","workers":workers,"batch_size":1,"max_iter":max_iter,"tolerance":1e-10,
        "retried":summary.processed,"converged_after_dc":summary.processed-summary.failed,"passed_constraints":summary.passed(),"constraint_violations":summary.violated_cases,
        "dc_failed":summary.dc_failed,"newton_failed":summary.newton_failed,"preparation_seconds":preparation_seconds,"retry_wall_seconds":retry_wall_seconds,
        "attempted_ac_iterations":summary.attempted_iterations,"summed_dc_seconds":summary.dc_time.as_secs_f64(),"summed_ac_seconds":summary.ac_time.as_secs_f64(),
        "results":results});
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap())
        .map_err(|e| e.to_string())?;
    println!(
        "retried={} converged_after_dc={} dc_failed={} newton_failed={} retry_wall={retry_wall_seconds:.6}s",
        summary.processed,
        summary.processed - summary.failed,
        summary.dc_failed,
        summary.newton_failed
    );
    Ok(())
}
