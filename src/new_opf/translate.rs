//! OPF setup systems that translate pandapower data into OPF-only ECS components.
//!
//! These functions are the **only** place that touches `io::pandapower` types on the
//! OPF side. Downstream OPF computation reads exclusively from ECS components and
//! resources — never from `Network`. This keeps pandapower isolated to the I/O layer
//! and lets ECS-side data be persisted/restored via `bevy_archive` independently.
//!
//! Each function is additive: skip the call → the component is absent → OPF systems
//! that gate on the component simply don't act on that entity.
//!
//! Components and Resources required to already exist (populated by
//! `load_pandapower_net`):
//!   - `PandapowerEntityMap` resource (line_entities, trafo_entities, …)
//!   - `NodeLookup` resource (bus_id → Entity)
//!   - `VNominal` component on bus entities (nominal voltage in kV)

use bevy_ecs::prelude::*;

use crate::basic::ecs::elements::{NodeLookup, PandapowerEntityMap, VNominal};
use crate::io::pandapower::Network;

use super::components::BranchFlowLimit;

/// Attach `BranchFlowLimit` to each in-service line entity.
///
/// rate_a_pu = `max_i_ka * vbase_kv * √3 / sn_mva`, with `vbase_kv` read from the
/// from-bus entity's `VNominal`. Lines with `max_i_ka == 0` are treated as
/// unconstrained and get **no** component attached.
pub fn attach_line_flow_limits(world: &mut World, net: &Network) {
    let Some(lines) = net.line.as_deref() else { return };

    let line_ents: Vec<Entity> = {
        let map = world
            .get_resource::<PandapowerEntityMap>()
            .expect("PandapowerEntityMap missing; call load_pandapower_net first");
        map.line_entities.clone()
    };

    if line_ents.len() != lines.len() {
        panic!(
            "line_entities length ({}) does not match net.line length ({})",
            line_ents.len(),
            lines.len()
        );
    }

    // Pre-collect vbase[i] in kV for each line's from-bus, so the main loop is
    // a clean mutable pass without borrow-checker contortions.
    let vbases_kv: Vec<f64> = {
        let lookup = world
            .get_resource::<NodeLookup>()
            .expect("NodeLookup missing; bus_systems::init_node_lookup must run first");
        lines
            .iter()
            .map(|line| {
                let bus_ent = lookup
                    .get_entity(line.from_bus)
                    .unwrap_or_else(|| panic!("from_bus {} has no entity", line.from_bus));
                world
                    .entity(bus_ent)
                    .get::<VNominal>()
                    .map(|vn| vn.0.0)
                    .unwrap_or_else(|| panic!("bus entity {bus_ent:?} missing VNominal"))
            })
            .collect()
    };

    let sn_mva = net.sn_mva;
    for (i, line) in lines.iter().enumerate() {
        if !line.in_service || line.max_i_ka == 0.0 {
            continue;
        }
        let rate_pu = line.max_i_ka * vbases_kv[i] * 3f64.sqrt() / sn_mva;
        world
            .entity_mut(line_ents[i])
            .insert(BranchFlowLimit { rate_a_pu: rate_pu });
    }
}

/// Attach `BranchFlowLimit` to each in-service transformer entity.
///
/// rate_a_pu = `sn_mva * parallel * max_loading_percent / 100 / sn_mva_sys`.
/// Transformers with no `max_loading_percent` use the pandapower default of 100%.
pub fn attach_trafo_flow_limits(world: &mut World, net: &Network) {
    let Some(trafos) = net.trafo.as_deref() else { return };

    let trafo_ents: Vec<Entity> = {
        let map = world
            .get_resource::<PandapowerEntityMap>()
            .expect("PandapowerEntityMap missing; call load_pandapower_net first");
        map.trafo_entities.clone()
    };

    if trafo_ents.len() != trafos.len() {
        panic!(
            "trafo_entities length ({}) does not match net.trafo length ({})",
            trafo_ents.len(),
            trafos.len()
        );
    }

    let sn_mva_sys = net.sn_mva;
    for (i, trafo) in trafos.iter().enumerate() {
        if !trafo.in_service {
            continue;
        }
        let loading_pct = trafo.max_loading_percent.unwrap_or(100.0);
        let rate_pu =
            trafo.sn_mva * (trafo.parallel as f64) * loading_pct * 0.01 / sn_mva_sys;
        if rate_pu == 0.0 {
            continue;
        }
        world
            .entity_mut(trafo_ents[i])
            .insert(BranchFlowLimit { rate_a_pu: rate_pu });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::ecs::network::{DataOps, PowerGrid};
    use crate::io::pandapower::load_csv_zip;
    use crate::io::pandapower::ecs_net_conv::LoadPandapowerNet;
    use crate::basic::ecs::elements::bus_systems::init_node_lookup;
    use bevy_ecs::system::RunSystemOnce;

    #[test]
    fn test_attach_line_flow_limits_ieee118() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE118/data.zip", dir);
        let net = load_csv_zip(&path).unwrap();

        let mut pf_net = PowerGrid::default();
        pf_net.load_pandapower_net(&net);
        let _ = pf_net.world_mut().run_system_once(init_node_lookup);

        attach_line_flow_limits(pf_net.world_mut(), &net);

        // Verify component attached to every in-service line with a finite limit.
        let line_ents = pf_net
            .world()
            .resource::<PandapowerEntityMap>()
            .line_entities
            .clone();
        let lines = net.line.as_deref().unwrap_or(&[]);
        assert_eq!(line_ents.len(), lines.len());

        let mut n_attached = 0usize;
        for (i, line) in lines.iter().enumerate() {
            let comp = pf_net.world().entity(line_ents[i]).get::<BranchFlowLimit>();
            if !line.in_service || line.max_i_ka == 0.0 {
                assert!(comp.is_none(), "line {i} got a limit but shouldn't");
            } else {
                let bfl = comp.unwrap_or_else(|| panic!("line {i} missing limit"));
                // Spot check the formula against the from-bus vbase
                let bus_idx = net.bus.iter().position(|b| b.index == line.from_bus).unwrap();
                let vbase = net.bus[bus_idx].vn_kv;
                let expect = line.max_i_ka * vbase * 3f64.sqrt() / net.sn_mva;
                assert!((bfl.rate_a_pu - expect).abs() < 1e-12,
                    "line {i}: got {:.6}, expected {:.6}", bfl.rate_a_pu, expect);
                n_attached += 1;
            }
        }
        println!("IEEE118: {} of {} lines got BranchFlowLimit", n_attached, lines.len());
        assert!(n_attached > 0);
    }

    #[test]
    fn test_attach_trafo_flow_limits_ieee118() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE118/data.zip", dir);
        let net = load_csv_zip(&path).unwrap();

        let mut pf_net = PowerGrid::default();
        pf_net.load_pandapower_net(&net);
        let _ = pf_net.world_mut().run_system_once(init_node_lookup);

        attach_trafo_flow_limits(pf_net.world_mut(), &net);

        let trafo_ents = pf_net
            .world()
            .resource::<PandapowerEntityMap>()
            .trafo_entities
            .clone();
        let trafos = net.trafo.as_deref().unwrap_or(&[]);
        assert_eq!(trafo_ents.len(), trafos.len());

        let mut n_attached = 0usize;
        for (i, trafo) in trafos.iter().enumerate() {
            let comp = pf_net.world().entity(trafo_ents[i]).get::<BranchFlowLimit>();
            if !trafo.in_service {
                assert!(comp.is_none(), "trafo {i} oos got a limit but shouldn't");
                continue;
            }
            let loading = trafo.max_loading_percent.unwrap_or(100.0);
            let expect = trafo.sn_mva * (trafo.parallel as f64) * loading * 0.01 / net.sn_mva;
            if expect == 0.0 {
                assert!(comp.is_none());
                continue;
            }
            let bfl = comp.unwrap_or_else(|| panic!("trafo {i} missing limit"));
            assert!((bfl.rate_a_pu - expect).abs() < 1e-12,
                "trafo {i}: got {:.6}, expected {:.6}", bfl.rate_a_pu, expect);
            n_attached += 1;
        }
        println!("IEEE118: {} of {} trafos got BranchFlowLimit", n_attached, trafos.len());
    }

    #[test]
    fn test_branch_flow_limits_combined_ieee118() {
        // line and trafo both contribute BranchFlowLimit components — same world, no clashes.
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE118/data.zip", dir);
        let net = load_csv_zip(&path).unwrap();

        let mut pf_net = PowerGrid::default();
        pf_net.load_pandapower_net(&net);
        let _ = pf_net.world_mut().run_system_once(init_node_lookup);

        attach_line_flow_limits(pf_net.world_mut(), &net);
        attach_trafo_flow_limits(pf_net.world_mut(), &net);

        let n_branches_with_limit = pf_net
            .world_mut()
            .query::<&BranchFlowLimit>()
            .iter(pf_net.world())
            .count();
        println!("IEEE118 combined: {} branch entities carry BranchFlowLimit", n_branches_with_limit);
        assert!(n_branches_with_limit > 0);
    }
}
