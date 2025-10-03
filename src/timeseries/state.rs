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

/// Stores a sequence of time-tagged voltage states over the course of the simulation.
///
/// This resource enables archiving the voltage vector (`VBusPu`) at each time step,
/// for later analysis, plotting, or exporting.
#[derive(Default, Resource, Serialize, Deserialize)]
pub struct TimeSeriesData {
    /// Simulation time vector (in seconds).
    pub t: Vec<f64>,
    /// Voltage state vector snapshots at each timestamp.
    pub data: Vec<DVector<Complex<f64>>>,
}

/// Updates the solver’s initial voltage vector using the latest simulation result.
///
/// This enables iterative solvers to reuse the previous converged solution as a warm start.
pub fn state_transfer(mut data: ResMut<PowerFlowMat>, pf_result: Res<PowerFlowResult>) {
    data.v_bus_init.clone_from(&pf_result.v);
}
/// Appends the current voltage vector and simulation time to the [`TimeSeriesData`] archive.
///
/// This system is conditional: it only runs if the resource [`TimeSeriesData`] exists.
pub fn state_preserve(
    time: Res<Time>,
    mut data: ResMut<TimeSeriesData>,
    pf_result: Res<PowerFlowResult>,
) {
    data.t.push(time.0);
    data.data.push(pf_result.v.clone());
}

/// Emits structural update events if the voltage or injection vectors have changed.
///
/// This system ensures proper triggering of rebuild logic without direct component comparison.
pub fn state_update(
    mut voltage: MessageWriter<VoltageChangeEvent>,
    mut sbus: MessageWriter<SBusChangeEvent>,
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

/// Plugin for managing simulation state transfer and archiving time series data.
///
/// This plugin serves two main purposes:
/// 1. **State Transfer**: Propagates the converged voltage vector to the next iteration.
/// 2. **State Preservation**: Records voltage states over time into [`TimeSeriesData`].
/// 3. **Change Detection**: Monitors voltage/injection changes and triggers structural update events.
///
/// # Dependencies
/// Automatically enables [`StructureUpdatePlugin`] to handle event propagation.
///
/// # System Scheduling
/// - `state_update` runs in the `First` schedule to flag early any component changes.
/// - `state_transfer` always runs in `PostUpdate`, updating the solver initial guess.
/// - `state_preserve` runs conditionally in `PostUpdate`, only if `TimeSeriesData` exists.
///
/// # Usage
/// Add this plugin to enable voltage vector replay or export functionality.
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
