use bevy_app::{App, First, Plugin, PostUpdate};
use bevy_ecs::prelude::*;
use nalgebra::{Complex, DVector};
use serde::{Deserialize, Serialize};

use crate::{
    basic::ecs::{
        elements::{BusID, SBusInjPu, VBusPu},
        powerflow::{
            structure_update::{SBusChangeEvent, StructureUpdatePlugin, VoltageChangeEvent},
            systems::PowerFlowMat,
        },
    },
    prelude::PowerFlowResult,
};

use super::sim_time::Time;

#[derive(Default, Resource, Serialize, Deserialize)]
pub struct TimeSeriesData {
    pub t: Vec<f64>,                      // in seconds
    pub data: Vec<DVector<Complex<f64>>>, // time series data
}

pub fn state_transfer(mut data: ResMut<PowerFlowMat>, pf_result: Res<PowerFlowResult>) {
    data.v_bus_init.clone_from(&pf_result.v);
}
pub fn state_preserve(
    time: Res<Time>,
    mut data: ResMut<TimeSeriesData>,
    pf_result: Res<PowerFlowResult>,
) {
    data.t.push(time.0);
    data.data.push(pf_result.v.clone());
    println!("t:{} v[0]:{}", time.0, data.data.last().unwrap()[0]);
}

pub fn state_update(
    mut voltage: EventWriter<VoltageChangeEvent>,
    mut sbus: EventWriter<SBusChangeEvent>,
    v: Query<&BusID, Changed<VBusPu>>,
    s: Query<&VBusPu, Changed<SBusInjPu>>,
) {
    if !v.is_empty() {
        voltage.write_default();
    }
    if !s.is_empty() {
        sbus.write_default();
    }
}
#[derive(Default)]
pub struct StateTransferPlugin;

impl Plugin for StateTransferPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<StructureUpdatePlugin>() {
            app.add_plugins(StructureUpdatePlugin);
        }
        app.add_systems(First, state_update);
        app.add_systems(
            PostUpdate,
            (
                state_transfer,
                state_preserve.run_if(resource_exists::<TimeSeriesData>),
            ),
        );
    }
}
