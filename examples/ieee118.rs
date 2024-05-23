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
    let folder = "D:/projects/rust/rustpower/out";
    let bus = folder.to_owned() + "/bus.csv";
    let gen = folder.to_owned() + "/gen.csv";
    let line = folder.to_owned() + "/line.csv";
    let shunt = folder.to_owned() + "/shunt.csv";
    let trafo = folder.to_owned() + "/trafo.csv";
    let extgrid = folder.to_owned() + "/extgrid.csv";
    let load = folder.to_owned() + "/load.csv";
    let mut net = Network::default();
    net.bus = load_pandapower_csv(bus);
    net.gen = Some(load_pandapower_csv(gen));
    net.line = Some(load_pandapower_csv(line));
    net.shunt = Some(load_pandapower_csv(shunt));
    net.trafo = Some(load_pandapower_csv(trafo));
    net.ext_grid = Some(load_pandapower_csv(extgrid));
    net.load = Some(load_pandapower_csv(load));
    net.sn_mva = 100.0;
    net.f_hz = 60.0;
    let pf = PFNetwork::from(net);
    let v_init = pf.create_v_init();
    let tol = Some(1e-6);
    let max_it = Some(10);
    let v = pf.run_pf(v_init.clone(), max_it, tol);
    println!("Vm,\t angle");
    for i in v.iter() {
        println!("{:.5}, {:.5}", i.modulus(), i.argument().to_degrees());
    }
}
