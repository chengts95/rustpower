use std::ops::DerefMut;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::{
    basic::{
        ecs::{
            elements::*,
            powerflow::init::{PQBus, PVBus},
        },
        sparse::cast::Cast,
    },
    prelude::ecs::network::SolverStage::*,
};

use super::{
    nonlinear_schedule::*,
    structure_update::{NodeTypeChangeEvent, StructureUpdatePlugin},
    systems::{PowerFlowMat, PowerFlowResult},
};

fn modify_qlim_system(
    mut cmd: Commands,
    buses: Res<NodeLookup>,
    common: Res<PFCommonData>,
    res: Res<PowerFlowResult>,
    mut mat: ResMut<PowerFlowMat>,
    mut res_convergence: ResMut<ConvergedResult>,
    node_agg: Option<Res<NodeAggRes>>,
    generators: Query<(&TargetBus, &PQLim), (With<TargetPMW>, With<TargetVmPu>)>,
    mut pf_bus: Query<&mut SBusInjPu, With<PVBus>>,
    mut event: EventWriter<NodeTypeChangeEvent>,
) {
    // This system may have trouble since multiple generators can be connected to the same bus.
    let cv = &res.v;
    let mis = &cv.component_mul(&(&mat.y_bus * cv).conjugate());
    let sbus_res = mis;
    let inv_order = &mat.reorder.transpose();
    let sbus_res = inv_order * sbus_res;

    let sbus_res = match &node_agg {
        Some(node_agg) => &node_agg.expand_mat.cast() * &sbus_res,
        None => sbus_res,
    };
    let mut structure_change = false;
    generators
        .iter()
        .map(|d| {
            let bus = d.0.0;
            let e = buses.get_entity(bus).unwrap();
            (e, bus, d.1)
        })
        .for_each(|(e, bus, lim)| {
            if !pf_bus.contains(e) {
                return;
            }
            let mut q_target = pf_bus.get_mut(e).unwrap();
            let q_mvar = (sbus_res[bus as usize].im - q_target.0.im) * common.sbase;
            let qlim = &lim.q;
            if q_mvar < qlim.min {
                structure_change = true;
                cmd.entity(e).remove::<PVBus>().insert(PQBus);
                q_target.deref_mut().0.im = qlim.min / common.sbase;
            }
            if q_mvar > qlim.max {
                structure_change = true;
                cmd.entity(e).remove::<PVBus>().insert(PQBus);
                q_target.deref_mut().0.im = qlim.max / common.sbase;
            }
        });
    if structure_change {
        // println!("QLim modified the bus types.");
        mat.v_bus_init.clone_from(&res.v);
        res_convergence.converged = NonlinearConvType::Continue;
        event.write(NodeTypeChangeEvent);
    }
}
#[derive(Default)]
pub struct QLimPlugin;

impl Plugin for QLimPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<StructureUpdatePlugin>() {
            //panic!("QLimPlugin requires StructureUpdatePlugin to be added before it.");
            app.add_plugins(StructureUpdatePlugin);
        }
        if !app.is_plugin_added::<NonLinearSchedulePlugin>() {
            //panic!("QLimPlugin requires StructureUpdatePlugin to be added before it.");
            app.add_plugins(NonLinearSchedulePlugin);
        }
        app.add_systems(Update, modify_qlim_system.in_set(AfterSolve));
    }
}
