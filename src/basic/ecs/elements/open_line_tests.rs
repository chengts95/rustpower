use super::*;
use crate::basic::ecs::{
    plugin::{BasePFPlugin, LineSwitchPlugin},
    post_processing::{LineResultData, PostProcessing},
    powerflow::{
        mutation::set_load_p,
        pf_init::PFInit,
        prepare::prepare_solver_inputs,
        result_extract::VBusUpdatePlugin,
        structure_update::{FullRebuildEvent, StructureUpdatePlugin},
        systems::{OriginalYBus, PowerFlowConfig, PowerFlowResult},
    },
};
use bevy_app::App;

fn params() -> LineParams {
    LineParams {
        r_ohm_per_km: 0.1,
        x_ohm_per_km: 0.2,
        g_us_per_km: 1000.0,
        c_nf_per_km: 4000.0,
        length_km: 1.0,
        df: 1.0,
        parallel: 1,
        max_i_ka: 1.0,
    }
}
fn fixture(enabled: bool) -> (App, Entity, Entity, Entity, Entity) {
    let mut app = App::new();
    app.add_plugins((BasePFPlugin, StructureUpdatePlugin, VBusUpdatePlugin));
    if enabled {
        app.add_plugins(LineSwitchPlugin);
    }
    let w = app.world_mut();
    w.insert_resource(PFCommonData {
        sbase: 100.0,
        f_hz: 50.0,
        wbase: 100.0 * std::f64::consts::PI,
    });
    for id in [10, 20, 30] {
        let e = w.spawn(BusID(id)).id();
        w.get_mut::<VNominal>(e).unwrap().0.0 = 10.0;
    }
    w.spawn((TargetBus(10), TargetVmPu(1.0), TargetVaDeg(0.0)));
    let load = w
        .spawn((TargetBus(20), TargetPMW(-5.0), TargetQMVar(-2.0)))
        .id();
    let line = w.spawn((LineID(42), FromBus(10), ToBus(20), params())).id();
    for (f, t) in [(10, 30), (30, 20)] {
        w.spawn((FromBus(f), ToBus(t), params()));
    }
    let switch = |bus| {
        (
            Switch {
                bus,
                element: 42,
                et: SwitchType::SwitchBusLine,
                z_ohm: 0.0,
            },
            SwitchState(true),
        )
    };
    let from = w.spawn(switch(10)).id();
    let to = w.spawn(switch(20)).id();
    w.run_schedule(PFInit);
    w.resource_mut::<PowerFlowConfig>().tol = Some(1e-10);
    (app, line, from, to, load)
}
fn solve(app: &mut App) {
    app.update();
    assert!(app.world().resource::<PowerFlowResult>().converged);
    app.post_process();
}
#[test]
fn terminal_reduction_is_idempotent_and_restores_current_parameters() {
    let (mut app, line, from, to, _) = fixture(true);
    for (f, t) in [
        (false, false),
        (true, false),
        (false, true),
        (true, true),
        (false, false),
    ] {
        app.world_mut().get_mut::<SwitchState>(from).unwrap().0 = !f;
        app.world_mut().get_mut::<SwitchState>(to).unwrap().0 = !t;
        prepare_solver_inputs(app.world_mut());
        let w = app.world();
        let patch = w.get::<Port4MatPatch>(line).unwrap().0;
        if f || t {
            assert_eq!(
                w.get::<OpenLineTerminals>(line).unwrap(),
                &OpenLineTerminals { from: f, to: t }
            );
            assert_eq!(patch[(0, 1)], Complex::new(0.0, 0.0));
            assert_eq!(patch[(1, 0)], Complex::new(0.0, 0.0));
            if f {
                assert_eq!(patch[(0, 0)], Complex::new(0.0, 0.0));
            }
            if t {
                assert_eq!(patch[(1, 1)], Complex::new(0.0, 0.0));
            }
        } else {
            assert!(w.get::<OpenLineModel>(line).is_none());
        }
        // Preparation may be repeated independently of normal line setup.
        prepare_switches(app.world_mut());
        assert_eq!(app.world().get::<Port4MatPatch>(line).unwrap().0, patch);
    }
    app.world_mut().get_mut::<SwitchState>(from).unwrap().0 = false;
    prepare_solver_inputs(app.world_mut());
    app.world_mut()
        .get_mut::<LineParams>(line)
        .unwrap()
        .c_nf_per_km = 8000.0;
    app.world_mut().write_message(FullRebuildEvent);
    prepare_solver_inputs(app.world_mut());
    let changed = app.world().get::<Port4MatPatch>(line).unwrap().0;
    prepare_switches(app.world_mut());
    assert_eq!(app.world().get::<Port4MatPatch>(line).unwrap().0, changed);
    app.world_mut().get_mut::<SwitchState>(from).unwrap().0 = true;
    prepare_solver_inputs(app.world_mut());
    let w = app.world();
    assert_eq!(
        w.get::<Port4MatPatch>(line).unwrap().0,
        super::super::line::line_systems::closed_line_patch(
            w.get::<LineParams>(line).unwrap(),
            w.resource::<PFCommonData>().wbase
        )
    );
}
#[test]
fn plugin_is_optional_and_runs_before_ybus_assembly() {
    let (mut app, line, from, _, _) = fixture(false);
    let closed = app.world().get::<Port4MatPatch>(line).unwrap().0;
    app.world_mut().get_mut::<SwitchState>(from).unwrap().0 = false;
    app.world_mut().write_message(FullRebuildEvent);
    prepare_solver_inputs(app.world_mut());
    assert_eq!(app.world().get::<Port4MatPatch>(line).unwrap().0, closed);
    assert!(app.world().get::<OpenLineModel>(line).is_none());
    app.add_plugins(LineSwitchPlugin);
    prepare_solver_inputs(app.world_mut());
    assert!(app.world().get::<OpenLineModel>(line).is_some());
    // Same physical matrix as the grid without this line, plus retained charging.
    let reduced = app.world().resource::<OriginalYBus>().0.clone();
    let charging = app.world().get::<Port4MatPatch>(line).unwrap().0[(1, 1)];
    app.world_mut().entity_mut(line).insert(OutOfService);
    app.world_mut().write_message(FullRebuildEvent);
    prepare_solver_inputs(app.world_mut());
    let disabled = &app.world().resource::<OriginalYBus>().0;
    for (r, c, value) in reduced.triplet_iter() {
        let delta = if r == 1 && c == 1 {
            charging
        } else {
            Complex::new(0.0, 0.0)
        };
        assert!((*value - disabled.get_entry(r, c).unwrap().into_value() - delta).norm() < 1e-12);
    }
}
#[test]
fn multiple_switches_any_open_wins_and_removal_reconnects() {
    let (mut app, line, from, _, _) = fixture(true);
    app.world_mut().spawn((
        Switch {
            bus: 10,
            element: 42,
            et: SwitchType::SwitchBusLine,
            z_ohm: 0.0,
        },
        SwitchState(true),
    ));
    app.world_mut().get_mut::<SwitchState>(from).unwrap().0 = false;
    prepare_solver_inputs(app.world_mut());
    assert!(app.world().get::<OpenLineTerminals>(line).unwrap().from);
    // Removing just the Switch component must not leave its cached reference active.
    app.world_mut().entity_mut(from).remove::<Switch>();
    prepare_solver_inputs(app.world_mut());
    assert!(app.world().get::<OpenLineModel>(line).is_none());
}
#[test]
fn policy_changes_disable_branch_and_warm_edits_do_not_rebuild() {
    let (mut app, line, from, _, load) = fixture(true);
    app.world_mut().get_mut::<SwitchState>(from).unwrap().0 = false;
    solve(&mut app);
    let revision = app.world().resource::<NodeLookup>().revision;
    assert!(app.world().get::<LineResultData>(line).unwrap().i_to_ka > 0.0);
    set_load_p(app.world_mut(), load, 7.0);
    solve(&mut app);
    assert_eq!(app.world().resource::<NodeLookup>().revision, revision);
    app.world_mut()
        .insert_resource(OpenLinePolicy::DisableBranch);
    solve(&mut app);
    assert_eq!(
        app.world().get::<Port4MatPatch>(line).unwrap().0,
        nalgebra::Matrix2::zeros()
    );
    assert_eq!(
        app.world().get::<LineResultData>(line).unwrap(),
        &LineResultData::default()
    );
    app.world_mut()
        .insert_resource(OpenLinePolicy::PreserveCharging);
    solve(&mut app);
    assert!(app.world().get::<LineResultData>(line).unwrap().i_to_ka > 0.0);
}
#[test]
fn disconnected_voltage_is_reconstructed_and_offline_line_stays_offline() {
    let (mut app, line, from, _, _) = fixture(true);
    app.world_mut().get_mut::<SwitchState>(from).unwrap().0 = false;
    solve(&mut app);
    let w = app.world();
    let model = w.get::<OpenLineModel>(line).unwrap();
    let bus = w.resource::<NodeLookup>().get_entity(20).unwrap();
    let voltage = model.voltage_ratio * w.get::<VBusPu>(bus).unwrap().0;
    let result = w.get::<LineResultData>(line).unwrap();
    assert_eq!(result.i_from_ka, 0.0);
    assert!((result.vm_from_pu - voltage.norm()).abs() < 1e-12);
    assert!((result.va_from_degree - voltage.arg().to_degrees()).abs() < 1e-12);
    app.world_mut().entity_mut(line).insert(OutOfService);
    app.world_mut().write_message(FullRebuildEvent);
    app.world_mut().get_mut::<SwitchState>(from).unwrap().0 = true;
    solve(&mut app);
    assert_eq!(
        app.world().get::<LineResultData>(line).unwrap(),
        &LineResultData::default()
    );
}
#[test]
fn line_entity_reference_does_not_redirect_after_label_reuse() {
    let (mut app, line, from, _, _) = fixture(true);
    assert_eq!(app.world().get::<SwitchLineRef>(from).unwrap().entity, line);
    app.world_mut().despawn(line);
    app.world_mut()
        .spawn((LineID(42), FromBus(10), ToBus(20), params()));
    assert!(
        super::super::topology::prepare_topology(app.world_mut())
            .unwrap_err()
            .contains("deleted line entity")
    );
}
#[test]
fn line_relabel_preserves_entity_reference() {
    let (mut app, line, from, _, _) = fixture(true);
    app.world_mut().get_mut::<LineID>(line).unwrap().0 = 1001;
    super::super::topology::prepare_topology(app.world_mut()).unwrap();
    assert_eq!(app.world().get::<Switch>(from).unwrap().element, 1001);
    assert_eq!(app.world().get::<SwitchLineRef>(from).unwrap().entity, line);
}

#[test]
fn startup_prepares_switches_after_line_setup_before_assembly() {
    let mut app = crate::basic::ecs::plugin::default_app();
    app.add_plugins(LineSwitchPlugin);
    let w = app.world_mut();
    w.insert_resource(PFCommonData {
        sbase: 100.0,
        f_hz: 50.0,
        wbase: 100.0 * std::f64::consts::PI,
    });
    for id in [10, 20, 30] {
        let e = w.spawn(BusID(id)).id();
        w.get_mut::<VNominal>(e).unwrap().0.0 = 10.0;
    }
    w.spawn((TargetBus(10), TargetVmPu(1.0), TargetVaDeg(0.0)));
    w.spawn((TargetBus(20), TargetPMW(-5.0), TargetQMVar(-2.0)));
    let line = w.spawn((LineID(42), FromBus(10), ToBus(20), params())).id();
    for (f, t) in [(10, 30), (30, 20)] {
        w.spawn((FromBus(f), ToBus(t), params()));
    }
    w.spawn((
        Switch {
            bus: 10,
            element: 42,
            et: SwitchType::SwitchBusLine,
            z_ohm: 0.0,
        },
        SwitchState(false),
    ));
    solve(&mut app);
    let expected = app.world().resource::<OriginalYBus>().0.clone();
    assert!(app.world().get::<OpenLineTerminals>(line).unwrap().from);
    app.world_mut().write_message(FullRebuildEvent);
    solve(&mut app);
    assert_eq!(app.world().resource::<OriginalYBus>().0, expected);
}

#[test]
fn deleting_bus_disables_connected_lines_and_their_switches() {
    let (mut app, line, from, to, load) = fixture(true);
    solve(&mut app);
    let w = app.world_mut();
    let bus = w.resource::<NodeLookup>().get_entity(20).unwrap();
    let removed = std::collections::HashSet::from([bus]);
    let disabled = topology::connected_elements_for_removal(w, &removed);
    assert!(disabled.contains(&line));
    assert!(disabled.contains(&from)); // Switch on the surviving end of this line.
    assert!(disabled.contains(&to));
    assert!(disabled.contains(&load));
    topology::validate_topology_with_disabled(w, &removed, &disabled).unwrap();
    for e in disabled {
        w.entity_mut(e).insert(OutOfService);
    }
    w.despawn(bus);
    w.write_message(FullRebuildEvent);
    solve(&mut app);
    assert!(app.world().get::<LineParams>(line).is_some());
    assert!(app.world().get::<FromBusMatrixIndex>(line).is_none());
    assert_eq!(
        app.world().get::<LineResultData>(line).unwrap().i_from_ka,
        0.0
    );
    assert!(app.world().get::<Switch>(from).is_some());
    assert!(topology::validate_activation(app.world_mut(), line).is_err());
    solve(&mut app);
    assert!(
        !app.world()
            .resource::<crate::basic::ecs::powerflow::structure_update::LastStructureAction>()
            .full_rebuild
    );
}

#[test]
fn transformer_switch_plugin_is_optional_and_rebuilds_from_current_parameters() {
    use crate::basic::ecs::plugin::TransformerSwitchPlugin;
    use crate::basic::ecs::post_processing::TrafoResultData;
    let mut app = App::new();
    app.add_plugins((BasePFPlugin, StructureUpdatePlugin, VBusUpdatePlugin));
    let w = app.world_mut();
    w.insert_resource(PFCommonData {
        sbase: 100.0,
        f_hz: 50.0,
        wbase: 100.0 * std::f64::consts::PI,
    });
    for (id, kv) in [(0, 110.0), (1, 20.0)] {
        let bus = w.spawn(BusID(id)).id();
        w.get_mut::<VNominal>(bus).unwrap().0.0 = kv;
    }
    w.spawn((TargetBus(0), TargetVmPu(1.02), TargetVaDeg(0.0)));
    w.spawn((TargetBus(1), TargetPMW(-5.0), TargetQMVar(-1.0)));
    let dev = TransformerDevice {
        df: 1.0,
        i0_percent: 0.8,
        pfe_kw: 80.0,
        vk_percent: 10.0,
        vkr_percent: 0.5,
        shift_degree: 17.0,
        sn_mva: 40.0,
        vn_hv_kv: 110.0,
        vn_lv_kv: 20.0,
        max_loading_percent: None,
        parallel: 1,
        tap: None,
    };
    let target = w
        .spawn((
            TransformerID(901),
            FromBus(0),
            ToBus(1),
            dev.clone(),
            TrafoResultData::default(),
        ))
        .id();
    w.spawn((
        TransformerID(903),
        FromBus(0),
        ToBus(1),
        dev,
        TrafoResultData::default(),
    ));
    let switch = w
        .spawn((
            Switch {
                bus: 0,
                element: 901,
                et: SwitchType::SwitchBusTransformer,
                z_ohm: 0.0,
            },
            SwitchState(false),
        ))
        .id();
    w.run_schedule(PFInit);
    assert!(w.get::<OpenTransformerModel>(target).is_none());
    let closed = w.get::<Port4MatPatch>(target).unwrap().0;
    app.add_plugins(TransformerSwitchPlugin);
    solve(&mut app);
    let patch = app.world().get::<Port4MatPatch>(target).unwrap().0;
    assert_eq!(patch[(0, 0)], Complex::new(0.0, 0.0));
    assert_eq!(patch[(0, 1)], Complex::new(0.0, 0.0));
    assert!(patch[(1, 1)].norm() > 0.0);
    assert_eq!(
        app.world().get::<TrafoResultData>(target).unwrap().i_hv_ka,
        0.0
    );
    solve(&mut app);
    assert!(
        !app.world()
            .resource::<crate::basic::ecs::powerflow::structure_update::LastStructureAction>()
            .full_rebuild
    );
    app.world_mut().get_mut::<SwitchState>(switch).unwrap().0 = true;
    solve(&mut app);
    assert_eq!(app.world().get::<Port4MatPatch>(target).unwrap().0, closed);
    app.world_mut().get_mut::<SwitchState>(switch).unwrap().0 = false;
    solve(&mut app);
    app.world_mut()
        .get_mut::<TransformerDevice>(target)
        .unwrap()
        .pfe_kw = 120.0;
    app.world_mut().write_message(FullRebuildEvent);
    solve(&mut app);
    let changed = app.world().get::<Port4MatPatch>(target).unwrap().0;
    assert!(changed[(1, 1)].re > patch[(1, 1)].re);
    app.world_mut().write_message(FullRebuildEvent);
    solve(&mut app);
    assert_eq!(app.world().get::<Port4MatPatch>(target).unwrap().0, changed);
    app.world_mut().entity_mut(target).insert(OutOfService);
    app.world_mut().write_message(FullRebuildEvent);
    solve(&mut app);
    assert_eq!(
        *app.world().get::<TrafoResultData>(target).unwrap(),
        TrafoResultData::default()
    );
}
