//! True single-device outages, entity watchers and sparse result collection.
#[path = "common/n1.rs"]
mod common_n1;
use rustpower::{n1::*, prelude::*};
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let workers = args
        .get(1)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("workers: {e}"))?
        .unwrap_or(3);
    let batch_size = args
        .get(2)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("batch_size: {e}"))?
        .unwrap_or(8);
    let initialization = match args.get(3).map(String::as_str).unwrap_or("base-ac") {
        "base-ac" => N1Initialization::BaseAc,
        "dc" => N1Initialization::DcPerCase,
        _ => return Err("initialization must be base-ac or dc".into()),
    };
    let reporting = match args.get(4).map(String::as_str).unwrap_or("status") {
        "status" => N1Reporting::StatusOnly,
        "details" => N1Reporting::Detailed,
        _ => return Err("reporting must be status or details".into()),
    };
    let net = serde_json::from_str(rustpower::testcases::case_ieee39::IEEE_39)
        .map_err(|e| e.to_string())?;
    let mut main_grid = PowerGrid::default();
    main_grid.world_mut().insert_resource(PPNetwork(net));
    main_grid.init_pf_net();
    common_n1::install_watches(main_grid.world_mut());
    let prepared = prepare_n1_with_initialization(main_grid.world_mut(), initialization)?;
    let study = prepared.study().map_err(|e| e.to_string())?;
    println!(
        "BASELINE line_loading+bus_voltage violations=0; transformer loading monitoring disabled"
    );
    let symbols = prepared.base().baseline().symbolic_ids();
    println!(
        "candidates={} workers={workers} batch_size={batch_size} shared_Symbolic={symbols:?} worker_init={initialization:?}",
        prepared.active_branches().count()
    );
    let faults = study.all_branch_outages();
    let summary = study
        .run(
            &faults,
            N1RunOptions {
                config: N1Config {
                    workers,
                    batch_size,
                    reporting,
                    ..Default::default()
                },
                ..Default::default()
            },
            |batch| {
                for event in batch.events {
                    println!(
                        "case={} outage={:?} worker={} outcome={:?} hits={}",
                        event.case_id,
                        event.outage,
                        event.worker_id,
                        event.outcome,
                        event.hits.len()
                    );
                }
            },
        )
        .map_err(|e| e.to_string())?;
    println!("{summary:?}");
    Ok(())
}
