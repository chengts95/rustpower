use super::*;
use crate::basic::ecs::{
    dcpf::{DcpfNewtonPfPlugin, DcpfSolverActive, build_dcpf_model, ecs_run_dcpf_pf},
    elements::*,
    network::{SolverStage, apply_permutation},
    plugin::{BasePFPlugin, PowerFlowSolverSet},
    powerflow::{
        init::{PQBus, PVBus, SlackBus},
        mutation::ParamDiff,
        qlim::QLimPlugin,
        result_extract::VBusUpdatePlugin,
        structure_update::{
            FullRebuildEvent, NodeTypeChangeEvent, StructureUpdatePlugin, VoltageChangeEvent,
        },
        systems::{OriginalYBus, PowerFlowConfig, PowerFlowMat, PowerFlowResult, init_states},
    },
};
use bevy_app::prelude::*;
use bevy_ecs::system::RunSystemOnce;

fn three_bus() -> App {
    let mut app = App::new();
    app.add_plugins((BasePFPlugin, StructureUpdatePlugin, VBusUpdatePlugin));
    let world = app.world_mut();
    world.insert_resource(PFCommonData {
        sbase: 100.0,
        f_hz: 50.0,
        wbase: 100.0 * std::f64::consts::PI,
    });
    let mut lookup = NodeLookup::default();
    for (id, s, vm) in [
        (0, Complex64::new(-0.2, -0.1), 1.0),
        (1, Complex64::new(0.1, 0.0), 1.03),
        (2, Complex64::new(0.0, 0.0), 1.0),
    ] {
        let mut e = world.spawn((BusID(id), SBusInjPu(s), VBusPu(Complex64::new(vm, 0.0))));
        match id {
            0 => {
                e.insert(PQBus);
            }
            1 => {
                e.insert(PVBus);
            }
            _ => {
                e.insert(SlackBus);
            }
        }
        lookup.insert(id, e.id());
    }
    world.insert_resource(lookup);
    for (from, to, b) in [(0, 1, 10.0), (1, 2, 7.0), (0, 2, 5.0)] {
        world.spawn(AdmittanceBranch {
            y: Admittance(Complex64::new(0.0, -b)),
            port: Port2::new(from, to),
            v_base: VBase(10.0),
        });
    }
    world.run_system_once(init_states).unwrap();
    world.run_system_once(apply_permutation).unwrap();
    world.resource_mut::<PowerFlowConfig>().tol = Some(1e-10);
    app
}

#[test]
fn parameter_messages_are_consumed_once_across_entry_points() {
    let mut app = three_bus();
    let world = app.world_mut();
    world.write_message(ParamDiff::Injection {
        bus: 0,
        dp_mw: -5.0,
        dq_mvar: 0.0,
    });
    prepare_solver_inputs(world); // standalone DC boundary
    assert!((world.resource::<PowerFlowMat>().s_bus[0].re + 0.25).abs() < 1e-12);
    let model = world.run_system_once(build_dcpf_model).unwrap();
    assert_eq!(model.n_active, 2);
    app.update(); // AC boundary must not apply the same delta a second time
    assert!((app.world().resource::<PowerFlowMat>().s_bus[0].re + 0.25).abs() < 1e-12);
    assert!(app.world().resource::<PowerFlowResult>().converged);
    assert!(
        app.world()
            .resource::<Messages<VoltageChangeEvent>>()
            .is_empty()
    );
}

#[test]
fn same_size_reordering_invalidates_caches_without_restamping_ybus() {
    let mut app = three_bus();
    app.update();
    let world = app.world_mut();
    let model = world.run_system_once(build_dcpf_model).unwrap();
    world.insert_resource(model);
    let original_ptr = world.resource::<OriginalYBus>().0.values().as_ptr();
    let revision = world.resource::<DcpfModelRevision>().0;
    let pq = world.resource::<NodeLookup>().get_entity(0).unwrap();
    let pv = world.resource::<NodeLookup>().get_entity(1).unwrap();
    world.entity_mut(pq).remove::<PQBus>().insert(PVBus);
    world.entity_mut(pv).remove::<PVBus>().insert(PQBus);
    world.write_message(NodeTypeChangeEvent);
    prepare_solver_inputs(world);
    assert_eq!(world.resource::<PowerFlowMat>().from_perm, vec![1, 0, 2]);
    assert_eq!(
        world.resource::<OriginalYBus>().0.values().as_ptr(),
        original_ptr
    );
    assert!(world.resource::<NewtonCache>().j_pattern.is_none());
    assert!(!world.contains_resource::<crate::basic::dcpf::DcpfModel>());
    assert!(world.resource::<DcpfModelRevision>().0 > revision);
    app.update();
    assert!(app.world().resource::<PowerFlowResult>().converged);
}

#[test]
fn continuation_does_not_repeat_dc_initialization() {
    let mut app = three_bus();
    app.add_plugins(DcpfNewtonPfPlugin);
    app.world_mut().insert_resource(DcpfSolverActive);
    app.update();
    let world = app.world_mut();
    assert!(world.resource::<PowerFlowResult>().converged);
    world.resource_mut::<SolvePass>().continuation = true;
    world.resource_mut::<PowerFlowConfig>().max_it = Some(0);
    // With an already converged AC voltage, a continuation needs zero NR steps.
    // Running DC again would change its angles and fail this zero-step solve.
    world.remove_resource::<crate::basic::dcpf::DcpfModel>();
    world.run_system_once(ecs_run_dcpf_pf).unwrap();
    let result = world.resource::<PowerFlowResult>();
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
}

#[derive(Resource, Default)]
struct Passes(usize, Option<DVector<Complex64>>);

#[derive(Resource, Default)]
struct InputCalls(usize);

fn count_input_calls(mut calls: ResMut<InputCalls>) {
    calls.0 += 1;
}

fn record_pass(
    mat: Res<PowerFlowMat>,
    result: Res<PowerFlowResult>,
    pass: Res<SolvePass>,
    mut seen: ResMut<Passes>,
) {
    if pass.continuation {
        let previous = seen.1.as_ref().unwrap();
        for (idx, &orig) in mat.from_perm.iter().enumerate() {
            assert!((mat.v_bus_init[idx] - previous[orig]).norm() < 1e-12);
        }
    }
    seen.0 += 1;
    seen.1 = Some(result.v.clone());
}

#[test]
fn qlim_reorders_and_continues_ac_then_publishes_final_voltage() {
    let mut app = three_bus();
    app.add_plugins((QLimPlugin, DcpfNewtonPfPlugin));
    app.init_resource::<Passes>();
    app.init_resource::<InputCalls>();
    app.add_systems(Update, count_input_calls.in_set(SolverStage::BeforeSolve));
    app.add_systems(
        Update,
        record_pass
            .after(PowerFlowSolverSet)
            .in_set(SolverStage::Solve),
    );
    app.world_mut().insert_resource(DcpfSolverActive);
    app.world_mut().spawn((
        TargetBus(1),
        TargetPMW(10.0),
        TargetVmPu(1.03),
        PQLim::new(-100.0, 100.0, 0.0, 0.0),
    ));
    crate::basic::ecs::elements::topology::prepare_topology(app.world_mut()).unwrap();
    app.update();
    let world = app.world();
    assert_eq!(world.resource::<Passes>().0, 2);
    assert_eq!(world.resource::<InputCalls>().0, 1);
    assert_eq!(world.resource::<PowerFlowMat>().npq, 2);
    assert!(world.resource::<PowerFlowResult>().converged);
    assert!(!world.contains_resource::<crate::basic::dcpf::DcpfModel>());
    for (bus, entity) in world.resource::<NodeLookup>().iter() {
        assert_eq!(
            world.get::<VBusPu>(entity).unwrap().0,
            world.resource::<PowerFlowResult>().v[bus as usize]
        );
    }
    app.update();
    assert_eq!(app.world().resource::<InputCalls>().0, 2);
    assert!(!app.world().resource::<SolvePass>().continuation);
    assert!(
        app.world()
            .contains_resource::<crate::basic::dcpf::DcpfModel>()
    );
}

#[test]
fn qlim_parameter_updates_preserve_bound_and_symbolic_cache() {
    use super::super::qlim::QLimited;

    let mut app = three_bus();
    app.add_plugins(QLimPlugin);
    app.world_mut().spawn((
        TargetBus(1),
        TargetPMW(10.0),
        TargetVmPu(1.03),
        PQLim::new(-100.0, 100.0, -100.0, 1.0),
    ));
    crate::basic::ecs::elements::topology::prepare_topology(app.world_mut()).unwrap();
    app.update();
    let world = app.world_mut();
    assert!(world.resource::<PowerFlowResult>().converged);
    let bus = world.resource::<NodeLookup>().get_entity(1).unwrap();
    assert!(world.get::<PQBus>(bus).is_some());
    assert_eq!(world.get::<QLimited>(bus).unwrap().q_pu, 0.01);
    let pattern = world
        .resource::<NewtonCache>()
        .j_pattern
        .as_ref()
        .unwrap()
        .j_col_ptrs
        .as_ptr();
    let ybus = world.resource::<PowerFlowMat>().y_bus.values().as_ptr();
    let permutation = world.resource::<PowerFlowMat>().from_perm.clone();
    let revision = world.resource::<DcpfModelRevision>().0;
    // Change physical injection on the limited bus; add its clamp exactly once.
    world.write_message(ParamDiff::Injection {
        bus: 1,
        dp_mw: -2.0,
        dq_mvar: -3.0,
    });
    prepare_solver_inputs(world);
    assert_eq!(world.resource::<DcpfModelRevision>().0, revision);
    let index = permutation.iter().position(|&i| i == 1).unwrap();
    assert!((world.get::<SBusInjPu>(bus).unwrap().0.im + 0.03).abs() < 1e-12);
    assert!((world.resource::<PowerFlowMat>().s_bus[index].im + 0.02).abs() < 1e-12);
    world.write_message(ParamDiff::VoltageMag {
        bus: 1,
        vm_pu: 0.95,
    });
    prepare_solver_inputs(world);
    assert!(world.get::<PVBus>(bus).is_none());
    assert!(world.get::<PQBus>(bus).is_some());
    assert_eq!(world.get::<QLimited>(bus).unwrap().q_pu, 0.01);
    assert_eq!(world.resource::<PowerFlowMat>().from_perm, permutation);
    assert_eq!(
        world.resource::<PowerFlowMat>().y_bus.values().as_ptr(),
        ybus
    );
    assert_eq!(
        world
            .resource::<NewtonCache>()
            .j_pattern
            .as_ref()
            .unwrap()
            .j_col_ptrs
            .as_ptr(),
        pattern
    );
    app.update();
    let world = app.world();
    assert!(world.resource::<PowerFlowResult>().converged);
    assert!(!world.resource::<SolvePass>().continuation);
    assert_eq!(
        world
            .resource::<NewtonCache>()
            .j_pattern
            .as_ref()
            .unwrap()
            .j_col_ptrs
            .as_ptr(),
        pattern
    );
    let result = world.resource::<PowerFlowResult>();
    let mut v = DVector::zeros(3);
    for (i, &original) in permutation.iter().enumerate() {
        v[i] = result.v[original];
    }
    let s = v.component_mul(&(&world.resource::<PowerFlowMat>().y_bus * &v).conjugate());
    assert!((s[index].im + 0.02).abs() < 1e-9);
}

#[test]
fn time_series_reuses_voltage_in_solver_order() {
    let mut app = three_bus();
    app.add_plugins(crate::timeseries::state::StateTransferPlugin);
    let world = app.world_mut();
    let bus0 = world.resource::<NodeLookup>().get_entity(0).unwrap();
    let bus1 = world.resource::<NodeLookup>().get_entity(1).unwrap();
    world.entity_mut(bus0).remove::<PQBus>().insert(PVBus);
    world.entity_mut(bus1).remove::<PVBus>().insert(PQBus);
    world.write_message(NodeTypeChangeEvent);
    app.update();
    let world = app.world_mut();
    assert!(world.resource::<PowerFlowResult>().converged);
    assert_eq!(world.resource::<PowerFlowMat>().from_perm, vec![1, 0, 2]);
    let solved = world.resource::<PowerFlowResult>().v.clone();
    assert!((world.resource::<PowerFlowMat>().v_bus_init[0] - solved[1]).norm() < 1e-12);
    assert!((world.resource::<PowerFlowMat>().v_bus_init[1] - solved[0]).norm() < 1e-12);
    // Run the next solve directly: the transferred voltage must already solve
    // the equations, without any corrective Newton iteration or component sync.
    world.resource_mut::<PowerFlowConfig>().max_it = Some(0);
    world
        .run_system_once(crate::basic::ecs::network::ecs_run_pf)
        .unwrap();
    assert!(world.resource::<PowerFlowResult>().converged);
    assert_eq!(world.resource::<PowerFlowResult>().v, solved);
}

#[test]
fn clearing_and_full_rebuild_preserve_cache_policy() {
    let mut app = App::new();
    app.add_plugins((BasePFPlugin, StructureUpdatePlugin, VBusUpdatePlugin));
    app.world_mut().insert_resource(PFCommonData {
        sbase: 100.0,
        f_hz: 50.0,
        wbase: 100.0 * std::f64::consts::PI,
    });
    let path = format!("{}/cases/IEEE118/data.zip", env!("CARGO_MANIFEST_DIR"));
    app.world_mut().insert_resource(PPNetwork(
        crate::io::pandapower::load_csv_zip(&path).unwrap(),
    ));
    app.update();
    assert!(app.world().resource::<NewtonCache>().j_pattern.is_some());
    clear_solver_caches(app.world_mut());
    assert!(app.world().resource::<SolverCachePolicy>().enabled);
    app.update();
    assert!(app.world().resource::<NewtonCache>().j_pattern.is_some());
    app.world_mut().write_message(FullRebuildEvent);
    app.update();
    assert!(app.world().resource::<NewtonCache>().j_pattern.is_some());
    assert!(app.world().resource::<PowerFlowResult>().converged);
    app.world_mut().resource_mut::<SolverCachePolicy>().enabled = false;
    app.world_mut().write_message(FullRebuildEvent);
    app.update();
    assert!(!app.world().contains_resource::<NewtonCache>());
    assert!(app.world().resource::<PowerFlowResult>().converged);
}

#[test]
fn relabeling_native_buses_preserves_ac_dc_results_and_publication() {
    let mut app = three_bus();
    app.update();
    assert!(app.world().resource::<PowerFlowResult>().converged);
    let mut expected: Vec<_> = app
        .world_mut()
        .query::<(Entity, &VBusPu)>()
        .iter(app.world())
        .map(|(e, v)| (e, v.0))
        .collect();
    expected.sort_by_key(|(e, _)| *e);
    let world = app.world_mut();
    for (old, label) in [(0, 1_000_000_000_000), (1, 42), (2, 10)] {
        let entity = world.resource::<NodeLookup>().get_entity(old).unwrap();
        world.get_mut::<BusID>(entity).unwrap().0 = label;
    }
    world.run_system_once(init_states).unwrap();
    world.run_system_once(apply_permutation).unwrap();
    app.update();
    assert!(app.world().resource::<PowerFlowResult>().converged);
    for (e, v) in expected {
        assert!((app.world().get::<VBusPu>(e).unwrap().0 - v).norm() < 1e-9);
    }
    assert_eq!(app.world().resource::<PowerFlowMat>().y_bus.nrows(), 3);
    app.add_plugins(DcpfNewtonPfPlugin);
    app.update();
    assert!(app.world().resource::<PowerFlowResult>().converged);
}

#[test]
fn warm_numeric_edits_do_not_rebuild_entity_or_index_caches() {
    let mut app = three_bus();
    app.update();
    let revision = app.world().resource::<NodeLookup>().revision;
    let bus = app.world().resource::<NodeLookup>().get_entity(0).unwrap();
    for dp in [-1.0, 1.0, -1.0] {
        app.world_mut().write_message(ParamDiff::InjectionEntity {
            bus,
            dp_mw: dp,
            dq_mvar: 0.0,
        });
        app.update();
        assert!(app.world().resource::<PowerFlowResult>().converged);
        assert_eq!(app.world().resource::<NodeLookup>().revision, revision);
        assert_eq!(app.world().get::<BusMatrixIndex>(bus).unwrap().0, 0);
    }
}
