use std::env;

use bevy_archive::prelude::*;
use ecs::post_processing::PostProcessing;
use rustpower::{io::archive::aurora_format::ArchiveSnapshotRes, prelude::{ecs::powerflow::qlim::QLimPlugin, *}};

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
    let file = format!("{}/cases/pegase9241/pegase9241.toml", dir);

    // Initialize the default ECS application with predefined plugins
    let mut pf_net = default_app();

    let net = read_manifest_from_file(&file, None).unwrap();
    // Initialize the default ECS application with predefined plugins

    pf_net.add_plugins(QLimPlugin);
    pf_net
        .world_mut()
        .resource_scope::<ArchiveSnapshotRes, _>(|world, registry| {
            load_world_manifest(world, &net, &registry.0.case_file_reg).unwrap();
        });

    pf_net.update(); //this will initalize the data for pf in the first run
    // Extract and validate the results
    let results = pf_net.world().get_resource::<PowerFlowResult>().unwrap();
    assert_eq!(results.converged, true);
    println!("ECS APP converged within {} iterations", results.iterations);

    // Post-process and print the results
    pf_net.post_process();
    pf_net.print_res_bus();
    timeit!(pegase9241, 10, || pf_net.update());
}
