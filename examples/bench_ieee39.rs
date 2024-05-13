use nalgebra::ComplexField;
use pf_module::{io::pandapower::Network, prelude::*};

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
    let file_path = test_ieee39::IEEE_39;
    let net: Network = serde_json::from_str(file_path).unwrap();
    let pf = PFNetwork::from(net);
    let v_init = pf.create_v_init();
    let tol = Some(1e-8);
    let max_it = Some(10);

    let v = (&pf).run_pf(v_init.clone(), max_it, tol);
    println!("Vm,\t angle");
    for i in v.iter() {
        println!("{:.5}, {:.5}", i.modulus(), i.argument().to_degrees());
    }
    timeit!(pf_ieee39,100,|| _ = (&pf).run_pf(v_init.clone(), max_it, tol));
}
