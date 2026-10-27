//! Rust-native N-1 screening and selected-rule detail output.
#[path = "common/n1.rs"]
mod common_n1;
use rustpower::{n1::*, prelude::*};
use std::{
    fs::File,
    io::{BufWriter, Write},
};

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let case = args.get(1).map(String::as_str).unwrap_or("39");
    let workers = args
        .get(2)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("workers: {e}"))?
        .unwrap_or(4);
    let batch_size = args
        .get(3)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("batch_size: {e}"))?
        .unwrap_or(8);
    let detailed = match args.get(4).map(String::as_str).unwrap_or("status") {
        "status" => false,
        "details" => true,
        _ => return Err("mode must be status or details".into()),
    };
    let output = args.get(5).map(String::as_str).unwrap_or("n1-events.json");
    let initialization = match args.get(6).map(String::as_str).unwrap_or("base-ac") {
        "base-ac" => N1Initialization::BaseAc,
        "dc" => N1Initialization::DcPerCase,
        _ => return Err("initialization must be base-ac or dc".into()),
    };
    let max_iter = args
        .get(7)
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("max_iter: {e}"))?
        .unwrap_or(30);
    let mut grid = PowerGrid::default();
    grid.world_mut()
        .insert_resource(PPNetwork(common_n1::load_case(case)?));
    grid.init_pf_net();
    common_n1::install_watches(grid.world_mut());
    let prepared = prepare_n1_with_initialization(grid.world_mut(), initialization)?;
    let study = prepared.study().map_err(|e| e.to_string())?;
    let faults = study.all_branch_outages();
    let options = N1RunOptions {
        config: N1Config {
            workers,
            batch_size,
            max_iter,
            ..Default::default()
        },
        ..Default::default()
    };
    // The collector writes each received batch, keeping retained output bounded.
    let mut writer = BufWriter::new(File::create(output).map_err(|e| e.to_string())?);
    writer.write_all(b"[\n").map_err(|e| e.to_string())?;
    let mut first = true;
    let mut write_error = None;
    let collect = |batch: N1EventBatch| {
        if write_error.is_some() {
            return;
        }
        for event in batch.events {
            let result = (|| -> Result<(), String> {
                if !first {
                    writer.write_all(b",\n").map_err(|e| e.to_string())?;
                }
                serde_json::to_writer(&mut writer, &common_n1::event_record(&event))
                    .map_err(|e| e.to_string())?;
                first = false;
                Ok(())
            })();
            if let Err(error) = result {
                write_error = Some(error);
                break;
            }
        }
    };
    let report = if detailed {
        study.details(&faults, options, collect)
    } else {
        study.run(&faults, options, collect)
    }
    .map_err(|e| e.to_string())?;
    if let Some(error) = write_error {
        return Err(error);
    }
    writer.write_all(b"\n]\n").map_err(|e| e.to_string())?;
    writer.flush().map_err(|e| e.to_string())?;
    println!(
        "processed={} passed={} violated={} numerical_failed={} islanded={} output={output}",
        report.processed(),
        report.passed(),
        report.violated(),
        report.numerical_failed(),
        report.islanded()
    );
    println!("{:?}", report.solve);
    Ok(())
}
