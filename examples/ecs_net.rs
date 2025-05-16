use std::env;

use ecs::{elements::PPNetwork, network::*, post_processing::*};
use rustpower::{
    io::{archive, pandapower::*},
    prelude::*,
};
use crate::archive::aurora_format::save_world_manifest;
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
    // t.add_type("vlim", Some(StorageTypeFlag::Table));

    // let v = VMagLimitPU(RangeUnit {
    //     min: 1.0.into(),
    //     max: 2.0.into(),
    // });
    // t.entities.push(0);
    // t.columns[0].push(serde_json::to_value(&v).unwrap());

    // println!("{:?}", serde_json::to_value(&t).unwrap());
    // let mut st = Vec::new();
    // unsafe {
    //     columnar_from_snapshot_unchecked(&t)
    //         .to_csv_writer(&mut st)
    //         .unwrap()
    // };
    // println!("{:?}", std::string::String::from_utf8(st));
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let zipfile = format!("{}/cases/IEEE118/data.zip", dir);
    let net = load_csv_zip(&zipfile).unwrap();

    let mut pf_net = PowerGrid::default();
    pf_net.world_mut().insert_resource(PPNetwork(net));
    pf_net.init_pf_net();
    pf_net.run_pf();
    assert_eq!(
        pf_net
            .world()
            .get_resource::<PowerFlowResult>()
            .unwrap()
            .converged,
        true
    );
    pf_net.post_process();
    pf_net.print_res_bus();
    pf_net.register_types();
    let reg = pf_net.get_snapshot_reg();
    let a = save_world_manifest(pf_net.world(), reg.unwrap());
    a.unwrap().to_file("IEEE118.toml", None).unwrap();
}
