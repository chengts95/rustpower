use rustpower::prelude::*;
use rustpower::io::pandapower::load_csv_zip;
use std::env;
use std::sync::Arc;
use std::time::Instant;
use std::thread;

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let zip_9241 = format!("{}/cases/pegase9241/data.zip", dir);
    
    println!("Loading PEGASE 9241 data...");
    let net = load_csv_zip(&zip_9241).expect("Failed to load 9241 data");
    let net_arc = Arc::new(net);

    let num_instances = 100;
    println!("Starting {} parallel power flow instances...", num_instances);
    
    let start = Instant::now();
    let mut handles = vec![];

    for i in 0..num_instances {
        let net_clone = Arc::clone(&net_arc);
        let handle = thread::spawn(move || {
            let mut pf_net = PowerGrid::default();
            pf_net.world_mut().insert_resource(PPNetwork((*net_clone).clone()));
            
            pf_net.init_pf_net();
            pf_net.run_pf();
            
            let res = pf_net.world().get_resource::<PowerFlowResult>().unwrap();
            if !res.converged {
                println!("Instance {} failed to converge", i);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let duration = start.elapsed();
    println!("Total time for {} instances: {:?}", num_instances, duration);
    println!("Average time per instance: {:?}", duration / num_instances);

    // Explicitly drop the shared network data to see the true memory floor
    drop(net_arc);
    
    println!("\n--- MEMORY LEAK TEST ---");
    println!("All PowerGrid instances and shared data have been DROPPED.");
    println!("Please check Task Manager now.");
    println!("Waiting 5 seconds before exiting...");
    
    thread::sleep(std::time::Duration::from_secs(5));
    println!("Done.");
}
