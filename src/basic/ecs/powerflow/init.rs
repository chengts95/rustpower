use crate::basic::ecs::elements::*;

use crate::basic::ecs::network::apply_permutation;
use crate::basic::ecs::plugin::{AfterPFInitStage, BeforePFInitStage, PFInitStage};

use bevy_ecs::prelude::*;

use bevy_app::{plugin_group, prelude::*};
use bevy_ecs::component::Mutable;
use bevy_ecs::system::SystemParam;
use nalgebra::{Complex, SimdComplexField};

use super::systems::{PowerFlowMat, init_states};

/// Marks an entity as a PQ bus (load bus).
/// Typically used when a node has specified active and reactive power injection,
/// but no fixed voltage magnitude or angle.
#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct PQBus;

/// Marks an entity as a PV bus (generator bus).
/// These nodes maintain a specified voltage magnitude and active power,
/// while the reactive power is determined by the solver.
#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct PVBus;

/// Marks an entity as a Slack (reference) bus.
/// This node balances the power mismatch in the system and holds a fixed voltage magnitude and angle.
#[derive(Component)]
#[component(storage = "SparseSet")]
pub struct SlackBus;

/// ECS system parameter for node operations. Provides convenient access to:
/// - Entities with a target bus and a tag component (`T`)
/// - Corresponding mutable state (`T1`) to inject into
/// - Node lookup resource for entity resolution
/// - Global power flow metadata
#[derive(SystemParam)]
pub struct NodeOp<'w, 's, T: Component, T1: Component<Mutability = Mutable>> {
    elements: Query<'w, 's, (&'static TargetBus, &'static T), Without<OutOfService>>,
    buses: Query<'w, 's, &'static mut T1>,
    node: ResMut<'w, NodeLookup>,
    common: Res<'w, PFCommonData>,
}

impl<'w, 's, T: Component, T1: Component<Mutability = Mutable>> NodeOp<'w, 's, T, T1> {
    /// Applies a user-defined injection function to each active element,
    /// based on the corresponding bus and per-unit scaling from system base power.
    ///
    /// # Arguments
    /// * `f` - Closure receiving the source data, the target mutable state, and base scaling.
    pub fn inject<F>(&mut self, mut f: F)
    where
        F: FnMut(&T, &mut T1, f64),
    {
        let s_base_frac = 1.0 / self.common.sbase;
        for (target_bus, val) in self.elements.iter() {
            let entity = self.node.get_entity(target_bus.0).unwrap();
            let mut target = self.buses.get_mut(entity).unwrap();
            f(val, &mut target, s_base_frac);
        }
    }
}

/// Labels all non-out-of-service, untagged buses as PQ buses by default.
/// Excludes PV, Slack, and out-of-service nodes.
fn label_pq_nodes(
    mut cmd: Commands,
    query: Query<
        Entity,
        (
            With<BusID>,
            Without<PVBus>,
            Without<PQBus>,
            Without<SlackBus>,
            Without<OutOfService>,
        ),
    >,
) {
    for entity in &query {
        cmd.entity(entity).insert(PQBus);
    }
}

/// Labels PV nodes (voltage-controlled generator nodes) based on available voltage and active power targets.
fn label_pv_nodes(
    mut cmd: Commands,
    nodes: Res<NodeLookup>,
    query: Query<&TargetBus, (With<TargetPMW>, With<TargetVmPu>, Without<OutOfService>)>,
) {
    for target_bus in &query {
        if let Some(entity) = nodes.get_entity(target_bus.0) {
            cmd.entity(entity).insert(PVBus);
        }
    }
}

/// Labels Slack nodes based on angle and voltage specification.
/// These nodes serve as the phase and magnitude reference for all others.
fn label_slack_nodes(
    mut cmd: Commands,
    nodes: Res<NodeLookup>,
    query: Query<&TargetBus, (With<TargetVaDeg>, With<TargetVmPu>, Without<OutOfService>)>,
) {
    for target_bus in &query {
        if let Some(entity) = nodes.get_entity(target_bus.0) {
            cmd.entity(entity).insert(SlackBus);
        }
    }
}

/// Injects active power (P in MW) into the system as per-unit complex real part at SBus nodes.
pub fn p_mw_inj(mut target_p: NodeOp<TargetPMW, SBusInjPu>) {
    target_p.inject(|val, state, sbase_frac| {
        state.0.re += val.0 * sbase_frac;
    });
}

/// Injects voltage magnitude and angle into VBus nodes,
/// reconstructing the complex per-unit voltage vector from separate magnitude and angle components.
pub fn v_inj(mut v: ParamSet<(NodeOp<TargetVmPu, VBusPu>, NodeOp<TargetVaDeg, VBusPu>)>) {
    let target_vm = v.p0();
    let mut buses = target_vm.buses;
    target_vm
        .elements
        .iter()
        .for_each(|(target_bus, target_vm_pu)| {
            let entity = target_vm.node.get_entity(target_bus.0).unwrap();
            let mut data = buses.get_mut(entity).unwrap();
            data.0 = data.0.simd_signum() * Complex::new(target_vm_pu.0, 0.0);
        });

    let target_va = v.p1();
    let mut buses = target_va.buses;
    target_va
        .elements
        .iter()
        .for_each(|(target_bus, target_va_deg)| {
            let entity = target_va.node.get_entity(target_bus.0).unwrap();
            let mut data = buses.get_mut(entity).unwrap();
            data.0 = data.0.simd_modulus() * Complex::from_polar(1.0, target_va_deg.0.to_radians());
        });
}

/// Injects reactive power (Q in MVar) into the system as per-unit complex imaginary part at SBus nodes.
pub fn q_mvar_inj(mut target_q: NodeOp<TargetQMVar, SBusInjPu>) {
    target_q.inject(|val, state, sbase_frac| {
        state.0.im += val.0 * sbase_frac;
    });
}

/// Plugin for tagging buses based on their operational role (PQ, PV, Slack).
#[derive(Default)]
pub struct NodeTaggingPlugin;

/// Plugin for initializing matrix construction systems for power flow computation.
#[derive(Default)]
pub struct MatBuilderPlugin;

// #[derive(Default)]
// pub struct ResultExtractPlugin;

impl Plugin for MatBuilderPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            Startup,
            (
                AfterPFInitStage.after(PFInitStage),
                BeforePFInitStage.before(PFInitStage),
            ),
        );
        app.add_systems(
            Startup,
            (
                init_states.run_if(not(resource_exists::<PowerFlowMat>)),
                apply_permutation,
            )
                .chain()
                .in_set(AfterPFInitStage),
        );
    }
}

impl Plugin for NodeTaggingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Startup,
            ((label_pv_nodes, label_slack_nodes), label_pq_nodes)
                .chain()
                .in_set(PFInitStage),
        );

        app.add_systems(Startup, (p_mw_inj, q_mvar_inj, v_inj).in_set(PFInitStage));
    }
}

plugin_group! {
    /// Doc comments and annotations are supported: they will be added to the generated plugin
    /// group.
    #[derive(Debug)]
    pub struct BasePFInitPlugins {
        :ElementSetupPlugin,
        // Identify PV PQ Ext Nodes.
        :NodeTaggingPlugin,
        // Build the power flow matrix.
        :MatBuilderPlugin


    }
}

// pub fn extract_pf_result(res: Res<PowerFlowResult>,
//        mat: Res<PowerFlowMat>,nodes: Res<NodeLookup>, mut v_bus:Query<&mut VBusPu>, mut s_bus:Query<&mut SBusPu>) {
//     let cv = &res.v;
//     let mis = &cv.component_mul(&(&mat.y_bus * cv).conjugate());
//     let sbus_res = mis.clone();
//     let sbus_res = &mat.reorder.transpose() * sbus_res;
//     let v = &mat.reorder.transpose() * &res.v;

// }
