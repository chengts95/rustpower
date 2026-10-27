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
