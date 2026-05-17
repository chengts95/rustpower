use ecs::{
    elements::PPNetwork,
    network::{DataOps, PowerFlow, PowerGrid},
};
use rustpower::{io::pandapower::*, prelude::*, testcases::case_ieee39::IEEE_39};
use std::env;
use std::time::{Duration, Instant};

#[macro_export]
macro_rules! timeit {
    ($name:expr, $times:expr, $block:expr) => {{
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
            " {}: {} loops - Average: {:?}, Max: {:?}, Min: {:?}",
            $name,
            $times,
            avg_duration,
            max_duration,
            min_duration
        );
    }};
}

fn run_benchmark(name: &str, net: Network, iterations: u32) {
    let mut pf_net = PowerGrid::default();
    pf_net.world_mut().insert_resource(PPNetwork(net));
    pf_net.init_pf_net();
    
    // Warmup
    pf_net.run_pf();
    
    let res = pf_net
        .world()
        .get_resource::<PowerFlowResult>()
        .unwrap();
    
    if !res.converged {
        println!("{} did not converge!", name);
        return;
    }
    println!("{} converged in {} iterations", name, res.iterations);
    let vm: Vec<f64> = res.v.iter().map(|c| c.norm()).collect();
    let min_v = vm.iter().fold(f64::MAX, |a, &b| a.min(b));
    let max_v = vm.iter().fold(f64::MIN, |a, &b| a.max(b));
    println!("  Vm range: [{:.4}, {:.4}]", min_v, max_v);

    timeit!(name, iterations, || {
        pf_net.run_pf();
    });
}

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // IEEE 39
    let net_39: Network = serde_json::from_str(IEEE_39).unwrap();
    run_benchmark("IEEE 39", net_39, 100);

    // IEEE 118
    let zip_118 = format!("{}/cases/IEEE118/data.zip", dir);
    if std::path::Path::new(&zip_118).exists() {
        let net_118 = load_csv_zip(&zip_118).unwrap();
        run_benchmark("IEEE 118", net_118, 100);
    } else {
        println!("IEEE 118 data not found at {}", zip_118);
    }

    // PEGASE 9241
    let zip_9241 = format!("{}/cases/pegase9241/data.zip", dir);
    if std::path::Path::new(&zip_9241).exists() {
        let net_9241 = load_csv_zip(&zip_9241).unwrap();
        run_benchmark("PEGASE 9241", net_9241, 300);
    } else {
        println!("PEGASE 9241 data not found at {}", zip_9241);
    }
}
