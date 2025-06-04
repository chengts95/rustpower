#![allow(deprecated)]
use std::{env, time::Instant};

use bevy_archive::prelude::{load_world_manifest, read_manifest_from_file};
use ecs::post_processing::PostProcessing;
use rustpower::{
    io::archive::aurora_format::ArchiveSnapshotRes,
    prelude::*,
    timeseries::{
        TimeSeriesDefaultPlugins,
        scheduled::{ScheduledStaticAction, ScheduledStaticActions},
        sim_time::{DeltaTime, Time},
        state::TimeSeriesData,
    },
};

#[macro_export]
macro_rules! timeit {
    ($name:ident, $times:expr, $block:expr) => {{
        use std::time::{Duration, Instant};
        let mut total_duration = Duration::new(0, 0);
        let mut max_duration = Duration::new(0, 0);
        let mut min_duration = Duration::new(u64::MAX, 999_999_999);

        for _ in 0..$times {
            let start_time = Instant::now();
            let _result = $block();
            let end_time = Instant::now();
            let duration = end_time - start_time;

            total_duration += duration;
            if duration > max_duration {
                max_duration = duration;
            }
            if duration < min_duration {
                min_duration = duration;
            }
        }

        let avg_duration = total_duration / $times;
        println!(
            " {} loops, {} - Average: {:?}, Max: {:?}, Min: {:?}",
            $times,
            stringify!($name),
            avg_duration,
            max_duration,
            min_duration
        );
    }};
}

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let file = format!("{}/cases/pegase9241/time_series.toml", dir);

    // Initialize the default ECS application with predefined plugins
    let mut pf_net = default_app();
    pf_net.add_plugins(TimeSeriesDefaultPlugins);
    pf_net.insert_resource(DeltaTime(15.0 * 60.0));
    pf_net.insert_resource(TimeSeriesData::default());
    pf_net.world_mut().spawn(ScheduledStaticActions {
        queue: vec![
            ScheduledStaticAction {
                execute_at: 30.0 * 60.0,
                action: rustpower::timeseries::scheduled::ScheduledActionKind::SetTargetPMW {
                    bus: 0,
                    value: 1000.0,
                },
            },
            ScheduledStaticAction {
                execute_at: 120.0 * 60.0,
                action: rustpower::timeseries::scheduled::ScheduledActionKind::SetTargetPMW {
                    bus: 9235,
                    value: 209.0,
                },
            },
        ]
        .into(),
    });
    let net = read_manifest_from_file(&file, None).unwrap();
    // Initialize the default ECS application with predefined plugins

    // pf_net.add_plugins(QLimPlugin);
    pf_net
        .world_mut()
        .resource_scope::<ArchiveSnapshotRes, _>(|world, registry| {
            load_world_manifest(world, &net, &registry.0.case_file_reg).unwrap();
        });
    // let archive = pf_net.to_case_file().unwrap();
    // archive.to_file("data_before.toml", None).unwrap();
    let t_end = 24.0 * 60.0 * 60.0;
    let tstart = Instant::now();
    while pf_net.world().resource::<Time>().0 < t_end {
        pf_net.update();
        //this will initalize the data for pf in the first run
        // Extract and validate the results
        // let results = pf_net.world().get_resource::<PowerFlowResult>().unwrap();
        // assert_eq!(results.converged, true);
        // println!("ECS APP converged within {} iterations", results.iterations);
    }
    let dur = Instant::now() - tstart;
    println!("ECS APP took {} ms", dur.as_millis());
    // Post-process and print the results
    pf_net.post_process();
    pf_net.print_res_bus();

    timeit!(pegase9241, 10, || pf_net.update());
}
