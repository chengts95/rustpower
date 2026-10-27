//! Base-case smoke run: worker-local Apps, shared analysis and branch output.
use rustpower::{
    n1::{BaseBatch, BaseTask, prepare_base, run_base_batches},
    prelude::*,
};
fn main() -> Result<(), String> {
    let net = serde_json::from_str(rustpower::testcases::case_ieee39::IEEE_39)
        .map_err(|e| e.to_string())?;
    let mut main_grid = PowerGrid::default();
    main_grid.world_mut().insert_resource(PPNetwork(net));
    main_grid.init_pf_net();
    let prepared = prepare_base(main_grid.world_mut())?;
    println!(
        "Shared DC/AC Symbolic: {:?}",
        prepared.baseline().symbolic_ids()
    );
    let groups = (0..4).map(|g| {
        BaseBatch::new(
            (0..3)
                .map(|i| BaseTask {
                    case_id: g * 3 + i,
                    max_iter: 50,
                    tolerance: 1e-10,
                    emit: true, // explicit diagnostic output rule
                })
                .collect(),
        )
        .unwrap()
    });
    let summary = run_base_batches(&prepared, 3, groups, |result| {
        println!(
            "case={} worker={} converged={} iterations={} residual={:.3e} branches={}",
            result.case_id,
            result.worker_id,
            result.solution.converged,
            result.solution.iterations,
            result.solution.residual,
            result.branches.len()
        );
    })?;
    println!("{summary:?}");
    Ok(())
}
