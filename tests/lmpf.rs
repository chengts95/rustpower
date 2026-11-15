use rustpower::testcases::case_ieee39::IEEE_39;
use rustpower::{lm::LmpfRunOptions, prelude::*};

fn grid() -> PowerGrid {
    let net = serde_json::from_str::<rustpower::io::pandapower::Network>(IEEE_39).unwrap();
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    grid
}

#[test]
fn prepared_lmpf_converges_and_reuses_study() {
    let mut grid = grid();
    let mut study = grid.prepare_lmpf();
    assert_eq!(study.backend(), "qdldl");
    let first = study.run(LmpfRunOptions::default()).unwrap();
    assert!(first.converged);
    assert!(first.residual_norm < 1e-6);
    assert_eq!(first.voltage.len(), 39);

    let second = study.run(LmpfRunOptions::default()).unwrap();
    assert!(second.converged);
    for (a, b) in first.voltage.iter().zip(second.voltage.iter()) {
        assert!((a - b).norm() < 1e-12);
    }
}

#[test]
fn lmpf_rejects_invalid_options() {
    let mut grid = grid();
    let mut study = grid.prepare_lmpf();
    assert!(
        study
            .run(LmpfRunOptions {
                max_iter: 0,
                ..Default::default()
            })
            .is_err()
    );
}
