use std::env;

use nalgebra::ComplexField;
use rustpower::{io::pandapower::*, prelude::*};

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
    let zipfile = format!("{}/cases/IEEE118/data.zip", dir);
    let net = load_csv_zip(zipfile).unwrap();
    let pf = PFNetwork::from(net);
    let v_init = pf.create_v_init();
    let tol = Some(1e-6);
    let max_it = Some(10);
    let v = pf.run_pf(v_init.clone(), max_it, tol);
    println!("{}",v);
    println!("Vm,\t angle");
    for (x, i) in v.iter().enumerate() {
        println!("{} {:.5}, {:.5}", x, i.modulus(), i.argument().to_degrees());
    }
    timeit!(pf_ieee118,100,|| _ = (&pf).run_pf(v_init.clone(), max_it, tol));
}
