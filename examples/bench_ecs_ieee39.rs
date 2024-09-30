use new_ecs::{
    elements::PPNetwork,
    network::{DataOps, PowerFlow, PowerFlowResult, PowerGrid},
    post_processing::PostProcessing,
};

use rustpower::{io::pandapower::Network, prelude::*};

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

#[allow(non_snake_case)]
fn main() {
    let file_path = test_ieee39::IEEE_39;
    let net: Network = serde_json::from_str(file_path).unwrap();
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

    timeit!(pf_ieee39, 100, || {
        pf_net.run_pf();
    });
    pf_net.post_process();
    pf_net.print_res_bus();
}
