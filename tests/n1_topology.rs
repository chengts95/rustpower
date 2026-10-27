#![cfg(feature = "experimental_n1")]
use rustpower::{n1::*, prelude::*};

#[test]
fn prepared_topology_uses_ybus_solver_order_and_fixed_slack_partition() {
    let net = serde_json::from_str(rustpower::testcases::case_ieee39::IEEE_39).unwrap();
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    let p = prepare_n1(grid.world_mut()).unwrap();
    let base = p.base().baseline();
    let topology = p.topology();
    let slack = base.npq() + base.npv();
    let slack_number = topology.dfs_number(slack).unwrap();
    let mut cuts = 0;
    for branch in p.active_branches() {
        if let Some(cut) = topology.bridge_cut(branch) {
            cuts += 1;
            assert_eq!(cut.component.len(), base.v_init().len());
            assert_eq!(cut.subtree_slacks + cut.other_slacks, 1);
            assert_eq!(
                cut.contains_in_subtree(slack_number),
                cut.subtree_slacks == 1
            );
            assert!(cut.slackless_bus_count() > 0);
            let from = p.base().catalog().from_buses()[branch];
            let to = p.base().catalog().to_buses()[branch];
            assert_ne!(
                cut.contains_in_subtree(topology.dfs_number(from).unwrap()),
                cut.contains_in_subtree(topology.dfs_number(to).unwrap())
            );
        }
    }
    assert!(cuts > 0);
    assert_eq!(cuts, topology.bridge_count());

    // Producer-side classification preserves branch indices for both output paths.
    let active_count = p.active_branches().count();
    let mut islands = Vec::new();
    let candidates = p.active_branches().filter(|&branch| {
        if let Some(cut) = topology.bridge_cut(branch) {
            if cut.slackless_bus_count() > 0 {
                islands.push((branch, cut));
                return false;
            }
        }
        true
    });
    let mut events = Vec::new();
    let summary = run_n1(
        &p,
        N1Config {
            workers: 2,
            max_iter: 0,
            ..Default::default()
        },
        candidates,
        |batch| events.extend(batch.events),
    )
    .unwrap();
    assert_eq!(islands.len(), cuts);
    assert_eq!(summary.submitted, active_count - cuts);
    assert_eq!(summary.processed, summary.submitted);
    assert!(
        events
            .iter()
            .all(|event| topology.bridge_cut(event.outage_index).is_none())
    );
}

#[test]
fn pegase9241_bridge_classification_separates_known_connected_failures() {
    let net = rustpower::io::pandapower::load_csv_zip(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/cases/pegase9241/data.zip"
    ))
    .unwrap();
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    let base = prepare_base(grid.world_mut()).unwrap();
    let started = std::time::Instant::now();
    let topology = N1Topology::from_baseline(base.baseline(), base.catalog()).unwrap();
    println!(
        "9241 topology-only preparation: {:?}; bridges={}",
        started.elapsed(),
        topology.bridge_count()
    );
    assert_eq!(topology.bridge_count(), 1665);
    let mut disconnected = 0;
    for branch in 0..base.catalog().len() {
        if let Some(cut) = topology.bridge_cut(branch) {
            disconnected += 1;
            assert_eq!(cut.component.len(), 9241);
            assert_eq!(cut.subtree_slacks + cut.other_slacks, 1);
            assert!(cut.slackless_bus_count() > 0);
        }
    }
    assert_eq!(disconnected, 1665);
    for (kind, ids) in [
        (BranchKind::Line, &[488, 3897, 7935, 10570][..]),
        (
            BranchKind::Transformer,
            &[26, 1411, 1423, 1424, 1446, 1447][..],
        ),
    ] {
        for &id in ids {
            let branch = base
                .catalog()
                .identities()
                .iter()
                .position(|key| key.kind == kind && key.id == id)
                .unwrap();
            assert!(topology.bridge_cut(branch).is_none(), "{kind:?} {id}");
        }
    }
}
