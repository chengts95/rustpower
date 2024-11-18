use std::env;
use nalgebra::ComplexField;
use new_ecs::{elements::PPNetwork, network::*, plugin::default_app, post_processing::*};
use rustpower::{io::pandapower::*, prelude::*};

/// A utility macro to measure the execution time of a code block.
/// It runs the provided code block a specified number of times (`$times`) 
/// and calculates the average, maximum, and minimum execution duration.
///
/// # Parameters
/// - `$name`: Identifier for the measured task (used in the output).
/// - `$times`: Number of iterations to run the code block.
/// - `$block`: Code block to measure.
///
/// # Output
/// Prints the average, maximum, and minimum durations for the code block.
/// Useful for performance profiling.
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

/// Entry point of the program.
/// Demonstrates three different methods to perform power flow analysis:
/// 1. Object-Oriented Programming (OOP) approach.
/// 2. ECS with traits.
/// 3. ECS with plugins.
/// 
/// Each method uses a different approach to solve the same problem,
/// showcasing the flexibility and trade-offs of each design.
fn main() {
    // Set up the file path for the input data
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let zipfile = format!("{}/cases/IEEE118/data.zip", dir);

    // Load the power network data from a CSV file in a ZIP archive
    let net = load_csv_zip(&zipfile).unwrap();

    // Execute each method and compare their outputs
    run_pf_net_obj(net.clone());            // Traditional OOP method
    run_ecs_with_trait(net.clone());        // ECS with traits
    run_ecs_app_with_plugins(net.clone());  // ECS with plugins
}

/// Performs power flow analysis using a traditional Object-Oriented Programming (OOP) approach.
///
/// # How It Works
/// - Creates a power flow network (`PFNetwork`) from the input data.
/// - Initializes the voltage vector (`v_init`) with default values.
/// - Solves the power flow problem using Newton-Raphson iteration.
/// - Prints the convergence results and voltage magnitudes/angles.
///
/// # Limitations
/// - Fixed initialization and parameters (`tol`, `max_it`) reduce flexibility.
/// - Results may differ from `pandapower` due to algorithmic differences.
/// - This implementation is not recommended now due to scalibility issues and 
/// the treatment to shunt is not correct, causing significant differences to 
/// pandapower's results.
/// # Parameters
/// - `net`: The input power network (`Network`) to analyze.
fn run_pf_net_obj(net: Network) {
    // Create a power flow network object
    let pf = PFNetwork::from(net);

    // Initialize voltage vector with a default method
    let v_init = pf.create_v_init(); 
    let tol = Some(1e-6);  // Convergence tolerance
    let max_it = Some(10); // Maximum iterations

    // Solve the power flow problem
    let (v, iter) = pf.run_pf(v_init.clone(), max_it, tol);

    // Output the results
    println!("OOP converged within {} iterations", iter);
    println!("Vm,\t angle");
    for (x, i) in v.iter().enumerate() {
        println!("{} {:.5}, {:.5}", x, i.modulus(), i.argument().to_degrees());
    }
}

/// Performs power flow analysis using an ECS (Entity-Component-System) approach with plugins.
///
/// # How It Works
/// - Uses a Bevy App to manage entities and systems.
/// - Registers the power network as a resource.
/// - Can implement custom traits such as post process directly on Bevy App.
///
/// # Advantages
/// - Highly modular: Plugins can be added or replaced without changing the core logic.
/// - Decoupled systems and resources improve code maintainability and scalability.
///
/// # Parameters
/// - `net`: The input power network (`Network`) to analyze.
fn run_ecs_app_with_plugins(net: Network) {
    // Initialize the default ECS application with predefined plugins
    let mut pf_net = default_app();

    // Register the power network as a resource in the ECS world
    pf_net.world_mut().insert_resource(PPNetwork(net));

    // Run the ECS application (executes all registered systems)
    pf_net.update(); // first execution calls startup schedule to initalize pf resources.

    // Extract and validate the results
    let results = pf_net.world().get_resource::<PowerFlowResult>().unwrap();
    assert_eq!(results.converged, true);
    println!("ECS APP converged within {} iterations", results.iterations);

    // Post-process and print the results
    pf_net.post_process();
    pf_net.print_res_bus();
}

/// Performs power flow analysis using an ECS approach with traits.
///
/// # How It Works
/// - Initializes a custom ECS application for power flow analysis.
/// - Methods are defined as trait interfaces to register systems and resources.
/// - Solves the power flow problem and prints results.
///
/// # Advantages
/// - Flexible: Developers can define custom behaviors and resources.
/// - Allows fine-grained control over the ECS lifecycle.
///
/// # Parameters
/// - `net`: The input power network (`Network`) to analyze.
fn run_ecs_with_trait(net: Network) {
    // Create a new ECS-based power grid
    let mut ecs_trait_net = PowerGrid::default();

    // Register the power network as a resource
    ecs_trait_net.world_mut().insert_resource(PPNetwork(net));

    // Initialize the ECS application for power flow analysis
    ecs_trait_net.init_pf_net();

    // Solve the power flow problem
    ecs_trait_net.run_pf();

    // Extract and validate the results
    let results = ecs_trait_net
        .world()
        .get_resource::<PowerFlowResult>()
        .unwrap();
    assert_eq!(results.converged, true);
    println!(
        "ECS Trait object converged within {} iterations",
        results.iterations
    );

    // Post-process and print the results
    ecs_trait_net.post_process();
    ecs_trait_net.print_res_bus();
}
