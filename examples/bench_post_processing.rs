use std::env;
use std::time::{Duration, Instant};

use rustpower::io::pandapower::*;
use rustpower::prelude::*;
use rustpower::prelude::newtonpf::NewtonCache;
use rustpower::testcases::case_ieee39::IEEE_39;

struct BenchStat {
    avg: Duration,
    min: Duration,
    max: Duration,
}

fn bench_loop<F: FnMut()>(times: u32, mut f: F) -> BenchStat {
    let mut total = Duration::ZERO;
    let mut min = Duration::MAX;
    let mut max = Duration::ZERO;

    for _ in 0..times {
        let t0 = Instant::now();
        f();
        let el = t0.elapsed();
        total += el;
        if el < min {
            min = el;
        }
        if el > max {
            max = el;
        }
    }
    BenchStat {
        avg: total / times,
        min,
        max,
    }
}

fn run_case_bench(name: &str, net: Network, n_bus: usize, n_line: usize, n_trafo: usize, loops: u32) {
    println!("\n================================================================================");
    println!(
        " BENCHMARK: {} (Buses: {}, Lines: {}, Transformers: {}, Total Branches: {})",
        name,
        n_bus,
        n_line,
        n_trafo,
        n_line + n_trafo
    );
    println!("================================================================================");

    let mut pf_net = PowerGrid::default();
    pf_net.world_mut().insert_resource(PPNetwork(net));
    pf_net.world_mut().insert_resource(NewtonCache::default());
    pf_net.init_pf_net();

    // Warmup solve and post-process
    pf_net.run_pf();
    pf_net.post_process();

    let res = pf_net
        .world()
        .get_resource::<PowerFlowResult>()
        .expect("PowerFlowResult missing");

    if !res.converged {
        println!("  [ERROR] Network did not converge!");
        return;
    }
    println!("  Converged in {} iterations.", res.iterations);

    // 1. Bench Pure Solve
    let solve_stat = bench_loop(loops, || {
        pf_net.run_pf();
    });

    // 2. Bench Pure Post-processing
    let post_stat = bench_loop(loops, || {
        pf_net.post_process();
    });

    // 3. Bench Full Cycle (Solve + Post-processing in Hot Loop)
    let combined_stat = bench_loop(loops, || {
        pf_net.run_pf();
        pf_net.post_process();
    });

    let total_branches = (n_line + n_trafo) as f64;
    let post_us = post_stat.avg.as_nanos() as f64 / 1_000.0;
    let solve_us = solve_stat.avg.as_nanos() as f64 / 1_000.0;
    let combined_us = combined_stat.avg.as_nanos() as f64 / 1_000.0;
    let branch_throughput = (total_branches / (post_stat.avg.as_secs_f64())) / 1_000_000.0;
    let post_ratio = (post_us / combined_us) * 100.0;

    println!("  [1] Pure Solve:           avg = {:>9.2} µs (min: {:>9.2} µs, max: {:>9.2} µs)",
        solve_us, solve_stat.min.as_nanos() as f64 / 1000.0, solve_stat.max.as_nanos() as f64 / 1000.0);
    println!("  [2] Pure Post-Process:     avg = {:>9.2} µs (min: {:>9.2} µs, max: {:>9.2} µs)",
        post_us, post_stat.min.as_nanos() as f64 / 1000.0, post_stat.max.as_nanos() as f64 / 1000.0);
    println!("  [3] Hot Loop (Solve+Post):  avg = {:>9.2} µs (min: {:>9.2} µs, max: {:>9.2} µs)",
        combined_us, combined_stat.min.as_nanos() as f64 / 1000.0, combined_stat.max.as_nanos() as f64 / 1000.0);
    println!("  ------------------------------------------------------------------------------");
    println!("  * Post-processing overhead in hot loop: {:.2}%", post_ratio);
    println!("  * Post-processing branch throughput:    {:.2} Million branches / sec", branch_throughput);
    println!("  * Per-branch compute time:              {:.2} ns / branch", (post_stat.avg.as_nanos() as f64) / total_branches);
}

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());

    // 1. IEEE 39
    let net_39: Network = serde_json::from_str(IEEE_39).unwrap();
    let n_bus_39 = net_39.bus.len();
    let n_line_39 = net_39.line.as_ref().map(|v| v.len()).unwrap_or(0);
    let n_trafo_39 = net_39.trafo.as_ref().map(|v| v.len()).unwrap_or(0);
    run_case_bench("IEEE 39", net_39, n_bus_39, n_line_39, n_trafo_39, 1000);

    // 2. IEEE 118
    let zip_118 = format!("{}/cases/IEEE118/data.zip", dir);
    if std::path::Path::new(&zip_118).exists() {
        let net_118 = load_csv_zip(&zip_118).unwrap();
        let n_bus_118 = net_118.bus.len();
        let n_line_118 = net_118.line.as_ref().map(|v| v.len()).unwrap_or(0);
        let n_trafo_118 = net_118.trafo.as_ref().map(|v| v.len()).unwrap_or(0);
        run_case_bench("IEEE 118", net_118, n_bus_118, n_line_118, n_trafo_118, 1000);
    }

    // 3. PEGASE 9241
    let zip_9241 = format!("{}/cases/pegase9241/data.zip", dir);
    if std::path::Path::new(&zip_9241).exists() {
        let net_9241 = load_csv_zip(&zip_9241).unwrap();
        let n_bus_9241 = net_9241.bus.len();
        let n_line_9241 = net_9241.line.as_ref().map(|v| v.len()).unwrap_or(0);
        let n_trafo_9241 = net_9241.trafo.as_ref().map(|v| v.len()).unwrap_or(0);
        run_case_bench("PEGASE 9241", net_9241, n_bus_9241, n_line_9241, n_trafo_9241, 100);
    }
}
