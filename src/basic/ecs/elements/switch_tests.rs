use super::*;
use crate::basic::ecs::{
    plugin::{BasePFPlugin, SwitchPluginTypeA, SwitchPluginTypeB},
    post_processing::{LineResultData, PostProcessing, SBusResult, VBusResult},
    powerflow::{
        mutation::{set_gen_vm, set_load_p},
        pf_init::PFInit,
        prepare::prepare_solver_inputs,
        qlim::{QLimPlugin, QLimited},
        result_extract::VBusUpdatePlugin,
        structure_update::{FullRebuildEvent, StructureUpdatePlugin},
        systems::{PowerFlowConfig, PowerFlowResult},
    },
};
use bevy_app::App;

fn fixture(merged: bool, plugin_b: bool) -> (App, Option<Entity>, Entity, Entity, Entity) {
    let mut app = App::new();
    app.add_plugins((BasePFPlugin, StructureUpdatePlugin, VBusUpdatePlugin));
    if merged {
        if plugin_b {
            app.add_plugins(SwitchPluginTypeB);
        } else {
            app.add_plugins(SwitchPluginTypeA);
        }
    }
    let w = app.world_mut();
    w.insert_resource(PFCommonData {
        sbase: 100.0,
        f_hz: 50.0,
        wbase: 100.0 * std::f64::consts::PI,
    });
    for label in if merged {
        vec![10, 20, 50, 90]
    } else {
        vec![10, 20, 90]
    } {
        let e = w.spawn(BusID(label)).id();
        w.get_mut::<VNominal>(e).unwrap().0.0 = 10.0;
    }
    w.spawn((TargetBus(10), TargetVmPu(1.0), TargetVaDeg(0.0)));
    let load = w
        .spawn((TargetBus(20), TargetPMW(-10.0), TargetQMVar(-5.0)))
        .id();
    w.spawn((TargetBus(90), TargetPMW(-15.0), TargetQMVar(-8.0)));
    let generator = w
        .spawn((
            TargetBus(if merged { 50 } else { 20 }),
            TargetPMW(5.0),
            TargetVmPu(1.02),
        ))
        .id();
    let mut lines = Vec::new();
    for (f, t, x) in [(10, 20, 0.1), (if merged { 50 } else { 20 }, 90, 0.125)] {
        lines.push(
            w.spawn((
                FromBus(f),
                ToBus(t),
                LineParams {
                    r_ohm_per_km: 0.0,
                    x_ohm_per_km: x,
                    g_us_per_km: 0.0,
                    c_nf_per_km: 0.0,
                    length_km: 1.0,
                    df: 1.0,
                    parallel: 1,
                    max_i_ka: 1.0,
                },
            ))
            .id(),
        );
    }
    let switch = if merged {
        Some(
            w.spawn((
                Switch {
                    bus: 20,
                    element: 50,
                    et: SwitchType::SwitchTwoBuses,
                    z_ohm: 0.0,
                },
                SwitchState(true),
            ))
            .id(),
        )
    } else {
        None
    };
    w.run_schedule(PFInit);
    w.resource_mut::<PowerFlowConfig>().tol = Some(1e-10);
    (app, switch, load, generator, lines[1])
}
fn solve(app: &mut App) {
    app.update();
    assert!(app.world().resource::<PowerFlowResult>().converged);
    app.post_process();
}
fn assert_matches(merged: &App, reference: &App) {
    for label in [10, 20, 90] {
        let a = merged
            .world()
            .resource::<NodeLookup>()
            .get_entity(label)
            .unwrap();
        let b = reference
            .world()
            .resource::<NodeLookup>()
            .get_entity(label)
            .unwrap();
        assert!(
            (merged.world().get::<VBusResult>(a).unwrap().0
                - reference.world().get::<VBusResult>(b).unwrap().0)
                .norm()
                < 1e-9
        );
    }
    let w = merged.world();
    let lut = w.resource::<NodeLookup>();
    assert_eq!(
        w.get::<VBusResult>(lut.get_entity(20).unwrap()).unwrap().0,
        w.get::<VBusResult>(lut.get_entity(50).unwrap()).unwrap().0
    );
}
#[test]
fn transitive_roots_are_resolved_without_prior_find() {
    let mut uf = NodeMerge::new(&[0, 1, 2, 3, 4]);
    uf.union(0, 1);
    uf.union(2, 3);
    uf.union(0, 2);
    let m = uf.get_node_mapping(0);
    assert_eq!(m[&0], m[&3]);
    assert_ne!(m[&0], m[&4]);
}
#[test]
fn sparse_bus_merge_selects_pv_and_expands_results() {
    let (mut app, _, _, _, line) = fixture(true, false);
    let (mut reference, _, _, _, ref_line) = fixture(false, false);
    assert_eq!(app.world().resource::<PowerFlowMat>().npq, 1);
    assert_eq!(app.world().resource::<PowerFlowMat>().npv, 1);
    assert_eq!(
        app.world().resource::<NodeAggRes>().representatives,
        vec![0, 2, 3]
    );
    solve(&mut app);
    solve(&mut reference);
    assert_matches(&app, &reference);
    assert!(
        (app.world().get::<LineResultData>(line).unwrap().p_from_mw
            - reference
                .world()
                .get::<LineResultData>(ref_line)
                .unwrap()
                .p_from_mw)
            .abs()
            < 1e-8
    );
    let e = app.world().resource::<NodeLookup>().get_entity(20).unwrap();
    assert!((app.world().get::<SBusResult>(e).unwrap().0 - Complex::new(10.0, 5.0)).norm() < 1e-9);
}
#[test]
fn warm_edits_sum_injections_and_update_shared_voltage_without_remerge() {
    let (mut app, _, load, generator, _) = fixture(true, false);
    let (mut reference, _, ref_load, ref_gen, _) = fixture(false, false);
    solve(&mut app);
    solve(&mut reference);
    let rev = app.world().resource::<NodeLookup>().revision;
    set_load_p(app.world_mut(), load, 18.0);
    set_load_p(reference.world_mut(), ref_load, 18.0);
    set_gen_vm(app.world_mut(), generator, 1.01);
    set_gen_vm(reference.world_mut(), ref_gen, 1.01);
    solve(&mut app);
    solve(&mut reference);
    assert_matches(&app, &reference);
    assert_eq!(app.world().resource::<NodeLookup>().revision, rev);
}
#[test]
fn opening_reclosing_and_removing_switch_rebuilds_the_model() {
    let (mut app, switch, _, _, _) = fixture(true, false);
    let switch = switch.unwrap();
    solve(&mut app);
    app.world_mut().get_mut::<SwitchState>(switch).unwrap().0 = false;
    prepare_solver_inputs(app.world_mut());
    assert!(!app.world().contains_resource::<NodeAggRes>());
    assert_eq!(app.world().resource::<PowerFlowMat>().y_bus.nrows(), 4);
    app.world_mut().get_mut::<SwitchState>(switch).unwrap().0 = true;
    prepare_solver_inputs(app.world_mut());
    assert_eq!(app.world().resource::<PowerFlowMat>().y_bus.nrows(), 3);
    app.world_mut().despawn(switch);
    prepare_solver_inputs(app.world_mut());
    assert_eq!(app.world().resource::<PowerFlowMat>().y_bus.nrows(), 4);
    assert!(!app.world().contains_resource::<NodeAggRes>());
}
#[test]
fn finite_switch_uses_inverse_impedance_and_clears_stale_stamps() {
    for b in [false, true] {
        let (mut app, switch, _, _, _) = fixture(true, b);
        let switch = switch.unwrap();
        app.world_mut().get_mut::<Switch>(switch).unwrap().z_ohm = 0.2;
        prepare_solver_inputs(app.world_mut());
        assert_eq!(
            app.world().get::<Port4MatPatch>(switch).unwrap().0[(0, 0)],
            Complex::new(5.0, 0.0)
        );
        assert!(!app.world().contains_resource::<NodeAggRes>());
        app.world_mut().get_mut::<SwitchState>(switch).unwrap().0 = false;
        prepare_solver_inputs(app.world_mut());
        assert!(app.world().get::<Port4MatPatch>(switch).is_none());
    }
}
#[test]
fn q_limits_apply_to_merged_electrical_node_and_survive_repartition() {
    let (mut app, _, _, generator, _) = fixture(true, false);
    let (mut reference, _, _, ref_gen, _) = fixture(false, false);
    for (a, g) in [(&mut app, generator), (&mut reference, ref_gen)] {
        a.add_plugins(QLimPlugin);
        a.world_mut()
            .entity_mut(g)
            .insert(PQLim::new(-100.0, 100.0, -1.0, 1.0));
    }
    solve(&mut app);
    solve(&mut reference);
    assert_matches(&app, &reference);
    assert_eq!(app.world().resource::<PowerFlowMat>().npv, 0);
    assert_eq!(app.world().resource::<PowerFlowMat>().y_bus.nrows(), 3);
    let mut q = app.world_mut().query::<&QLimited>();
    assert_eq!(q.iter(app.world()).count(), 1);
    app.world_mut().write_message(FullRebuildEvent);
    solve(&mut app);
    assert_matches(&app, &reference);
}
#[test]
fn slack_priority_overrides_pv_in_same_group() {
    let (mut app, switch, _, generator, _) = fixture(true, false);
    // Merge Slack 10 and PV 50. PQ 20 stays separate.
    app.world_mut()
        .get_mut::<Switch>(switch.unwrap())
        .unwrap()
        .bus = 10;
    prepare_solver_inputs(app.world_mut());
    let mat = app.world().resource::<PowerFlowMat>();
    assert_eq!(mat.npv, 0);
    assert_eq!(mat.npq, 2);
    assert_eq!(app.world().resource::<NodeAggRes>().representatives[0], 0);
    solve(&mut app);
    let w = app.world();
    let e = w.resource::<NodeLookup>().get_entity(50).unwrap();
    assert!((w.get::<VBusPu>(e).unwrap().0 - Complex::new(1.0, 0.0)).norm() < 1e-12);
    set_gen_vm(app.world_mut(), generator, 1.08);
    solve(&mut app);
    assert!((app.world().get::<VBusPu>(e).unwrap().0 - Complex::new(1.0, 0.0)).norm() < 1e-12);
    app.world_mut().write_message(FullRebuildEvent);
    solve(&mut app);
    assert!((app.world().get::<VBusPu>(e).unwrap().0 - Complex::new(1.0, 0.0)).norm() < 1e-12);
}
#[test]
fn dc_initialization_matches_explicitly_fused_grid() {
    use crate::basic::ecs::dcpf::{DcpfNewtonPfPlugin, DcpfSolverActive};
    let (mut app, _, _, _, _) = fixture(true, false);
    let (mut reference, _, _, _, _) = fixture(false, false);
    for a in [&mut app, &mut reference] {
        a.add_plugins(DcpfNewtonPfPlugin);
        a.world_mut().insert_resource(DcpfSolverActive);
    }
    solve(&mut app);
    solve(&mut reference);
    assert_matches(&app, &reference);
}

#[test]
fn generators_on_different_merged_buses_share_q_limits() {
    let (mut app, _, _, generator, _) = fixture(true, false);
    let (mut reference, _, _, ref_generator, _) = fixture(false, false);
    for (a, g) in [(&mut app, generator), (&mut reference, ref_generator)] {
        a.add_plugins(QLimPlugin);
        a.world_mut()
            .entity_mut(g)
            .insert(PQLim::new(-100.0, 100.0, -1.0, 1.0));
        a.world_mut().spawn((
            TargetBus(20),
            TargetPMW(2.0),
            TargetVmPu(1.02),
            PQLim::new(-100.0, 100.0, -2.0, 2.0),
        ));
        a.world_mut().write_message(FullRebuildEvent);
    }
    solve(&mut app);
    solve(&mut reference);
    assert_matches(&app, &reference);
    assert_eq!(app.world().resource::<PowerFlowMat>().npv, 0);
    let mut q = app.world_mut().query::<&QLimited>();
    let clamps: Vec<_> = q.iter(app.world()).map(|q| q.q_pu).collect();
    assert_eq!(clamps.len(), 1);
    assert!((clamps[0].abs() - 0.03).abs() < 1e-12);
}
#[test]
fn plugin_without_switches_keeps_full_bus_space() {
    let (mut app, _, _, _, _) = fixture(false, false);
    app.add_plugins(SwitchPluginTypeA);
    solve(&mut app);
    assert!(!app.world().contains_resource::<NodeAggRes>());
    assert_eq!(app.world().resource::<PowerFlowMat>().y_bus.nrows(), 3);
}

#[test]
fn phase_shifter_dc_injections_use_merged_endpoint_indices() {
    use crate::basic::ecs::dcpf::build_dcpf_model;
    let (mut app, _, _, _, line) = fixture(true, false);
    let (mut reference, _, _, _, ref_line) = fixture(false, false);
    for (a, e) in [(&mut app, line), (&mut reference, ref_line)] {
        a.world_mut()
            .entity_mut(e)
            .remove::<LineParams>()
            .insert(TransformerDevice {
                df: 1.0,
                i0_percent: 0.0,
                pfe_kw: 0.0,
                vk_percent: 10.0,
                vkr_percent: 0.0,
                shift_degree: 5.0,
                sn_mva: 100.0,
                vn_hv_kv: 10.0,
                vn_lv_kv: 10.0,
                max_loading_percent: None,
                parallel: 1,
                tap: None,
            });
        a.world_mut().write_message(FullRebuildEvent);
        prepare_solver_inputs(a.world_mut());
    }
    let merged = app.world_mut().run_system_once(build_dcpf_model).unwrap();
    let fused = reference
        .world_mut()
        .run_system_once(build_dcpf_model)
        .unwrap();
    assert!((&merged.p_shift - &fused.p_shift).norm() < 1e-12);
    assert!(merged.p_shift.norm() > 0.1);
}
