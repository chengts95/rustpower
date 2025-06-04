use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::basic::{
    ecs::{elements::*, network::ecs_run_pf},
    sparse::cast::Cast,
};

use super::{
    structure_update::VoltageChangeEvent,
    systems::{PowerFlowMat, PowerFlowResult},
};
use crate::prelude::ecs::network::SolverStage::Solve;

pub fn extract_powerflow_results(
    mat: Res<PowerFlowMat>,
    res: Res<PowerFlowResult>,
    buses: Res<NodeLookup>,
    mut q: Query<&mut VBusPu>,
    node_agg: Option<Res<NodeAggRes>>,
    mut event: EventWriter<VoltageChangeEvent>,
) {
    let v = &mat.reorder.transpose() * &res.v;
    let v = match &node_agg {
        Some(node_agg) => &node_agg.expand_mat_v.cast() * &v,
        None => v,
    };
    for i in 0..v.len() {
        let entity = buses.get_entity(i as i64).unwrap();
        if let Ok(mut bus) = q.get_mut(entity) {
            bus.0 = v[i];
        }
    }
    event.write(VoltageChangeEvent);
}

#[derive(Default)]
pub struct VBusUpdatePlugin;

impl Plugin for VBusUpdatePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            extract_powerflow_results.after(ecs_run_pf).in_set(Solve),
        );
    }
}
