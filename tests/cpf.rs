use rustpower::prelude::*;
use rustpower::testcases::case_ieee39::IEEE_39;

fn grid() -> PowerGrid {
    let net = serde_json::from_str::<rustpower::io::pandapower::Network>(IEEE_39).unwrap();
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    grid
}

#[test]
fn ieee39_cpf_passes_nose_and_preserves_physical_bus_order() {
    let mut grid = grid();
    let report = grid.run_cpf(CpfOptions::default()).unwrap();
    assert!(report.passed_nose());
    assert!(report.points.len() > 3);
    let nose = report.nose.as_ref().unwrap();
    assert!((2.13..2.14).contains(&nose.lambda));
    assert_eq!(nose.voltage.len(), 39);
    assert!(report.stats.factorizations > 0);

    let base = grid.world().resource::<PowerFlowResult>();
    for (actual, expected) in report.points[0].voltage.iter().zip(base.v.iter()) {
        assert!((actual - expected).norm() < 1e-12);
    }
}

#[test]
fn truncated_trace_does_not_claim_a_nose() {
    let mut grid = grid();
    let report = grid
        .run_cpf(CpfOptions {
            max_points: 1,
            ..Default::default()
        })
        .unwrap();
    assert!(!report.passed_nose());
    assert!(report.nose.is_none());
    assert_eq!(report.termination, CpfTermination::MaxPointsReached);
    assert_eq!(report.lambda_max(), 1.0);
}
