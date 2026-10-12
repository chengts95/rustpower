use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::basic::{ecs::elements::*, sparse::cast::Cast};

use super::systems::{PowerFlowMat, PowerFlowResult};

pub fn extract_powerflow_results(
    res: Res<PowerFlowResult>,
    buses: Res<NodeLookup>,
    mut q: Query<&mut VBusPu>,
    node_agg: Option<Res<NodeAggRes>>,
    mut mat: ResMut<PowerFlowMat>,
) {
    // A diverged voltage vector would poison the warm start of the next solve.
    if !res.converged {
        return;
    }
    let v = match &node_agg {
        Some(node_agg) => &node_agg.expand_mat_v.cast() * &res.v,
        None => res.v.clone_owned(),
    };
    for i in 0..v.len() {
        let entity = buses.get_entity(i as i64).unwrap();
        if let Ok(mut bus) = q.get_mut(entity) {
            bus.0 = v[i];
        }
    }
    // Result publication is not an input edit. Preserve warm-start state
    // directly without sending a voltage-change event back into preparation.
    for idx in 0..mat.from_perm.len() {
        mat.v_bus_init[idx] = res.v[mat.from_perm[idx]];
    }
}

#[derive(Default)]
pub struct VBusUpdatePlugin;

impl Plugin for VBusUpdatePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, extract_powerflow_results);
    }
}
