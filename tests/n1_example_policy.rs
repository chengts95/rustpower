#![cfg(feature = "experimental_n1")]
#[path = "../examples/common/n1.rs"]
mod common_n1;
use rustpower::{n1::*, prelude::*};

#[test]
fn baseline_gate_uses_line_constraints_and_keeps_transformer_outages() {
    let mut net: rustpower::io::pandapower::Network =
        serde_json::from_str(rustpower::testcases::case_ieee39::IEEE_39).unwrap();
    for line in net.line.as_mut().unwrap() {
        line.max_i_ka = Some(1e6);
    }
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    // Change only postprocessing ratings after Ybus initialization: every transformer
    // now has an artificial loading violation which the example policy must omit.
    let mut ratings = grid
        .world_mut()
        .query::<&mut ecs::elements::trans::TransformerDevice>();
    for mut device in ratings.iter_mut(grid.world_mut()) {
        device.sn_mva = 1e-9;
    }
    let mut limits = grid
        .world_mut()
        .query::<&mut ecs::elements::bus::VmLimit<ecs::elements::units::PerUnit>>();
    for mut limit in limits.iter_mut(grid.world_mut()) {
        *limit = ecs::elements::bus::VmLimit::new(0.1, 2.0);
    }
    common_n1::install_watches(grid.world_mut());
    for mode in [N1Initialization::BaseAc, N1Initialization::DcPerCase] {
        let p = prepare_n1_with_initialization(grid.world_mut(), mode).unwrap();
        common_n1::check_baseline(&p).unwrap();
        let transformer = p
            .base()
            .catalog()
            .identities()
            .iter()
            .position(|id| id.kind == BranchKind::Transformer)
            .unwrap();
        assert!(p.active_branches().any(|i| i == transformer));
        if let Some(base) = p.base_solution() {
            assert!(
                p.base()
                    .catalog()
                    .compute_one(transformer, &base.voltage, None)
                    .unwrap()
                    .loading_percent
                    > 100.
            );
        }
        let line = p
            .base()
            .catalog()
            .identities()
            .iter()
            .position(|id| id.kind == BranchKind::Line)
            .unwrap();
        let entity = p.base().catalog().source_entities()[line];
        grid.world_mut().entity_mut(entity).insert(N1Watch {
            rules: vec![WatchRule {
                rule_id: 10,
                condition: WatchCondition::BranchLoadingAbove(-1.),
            }],
        });
        let bad = prepare_n1_with_initialization(grid.world_mut(), mode).unwrap();
        let error = common_n1::check_baseline(&bad).unwrap_err();
        assert!(
            error.contains("base violates 1 monitored constraints"),
            "{error}"
        );
        common_n1::install_watches(grid.world_mut());
        let bus = grid.world_mut().query_filtered::<bevy_ecs::entity::Entity,
            bevy_ecs::query::With<ecs::elements::bus::VmLimit<ecs::elements::units::PerUnit>>>()
            .iter(grid.world()).next().unwrap();
        *grid
            .world_mut()
            .get_mut::<ecs::elements::bus::VmLimit<ecs::elements::units::PerUnit>>(bus)
            .unwrap() = ecs::elements::bus::VmLimit::new(0.0, 0.1);
        let voltage_bad = prepare_n1_with_initialization(grid.world_mut(), mode).unwrap();
        assert!(
            common_n1::check_baseline(&voltage_bad)
                .unwrap_err()
                .contains("base violates 1 monitored constraints")
        );
        *grid
            .world_mut()
            .get_mut::<ecs::elements::bus::VmLimit<ecs::elements::units::PerUnit>>(bus)
            .unwrap() = ecs::elements::bus::VmLimit::new(0.1, 2.0);
    }
}
