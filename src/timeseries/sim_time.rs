use bevy_app::{App, First, Plugin};
use bevy_ecs::prelude::*;
use derive_more::derive::*;
use serde::{Deserialize, Serialize};
/// Represents a fixed simulation time step in seconds.
///
/// This resource defines how much simulation time advances per frame.
/// It is typically set at the beginning and remains constant throughout the simulation.
#[derive(PartialEq, Default, From, Into, Add, Mul, Div, Resource, Serialize, Deserialize)]
pub struct DeltaTime(pub f64); // in seconds

/// Represents the current simulation time in seconds.
///
/// This resource is incremented every frame using the [`DeltaTime`] value.
/// It can be queried in systems to determine the current simulated time.
#[derive(PartialEq, Default, From, Into, Add, Mul, Div, Resource, Serialize, Deserialize)]
pub struct Time(pub f64); // in seconds

impl Time {
    /// Returns the total elapsed simulation time in seconds.
    pub fn elapsed_seconds(&self) -> f64 {
        self.0
    }
}
/// Advances the simulation time by one step using the configured [`DeltaTime`] value.
///
/// This system updates the [`Time`] resource by adding `dt` at the start of every frame.
///
/// # Scheduling
/// Runs in the [`First`] schedule, ensuring it executes before most other systems.
pub fn advance(mut t: ResMut<Time>, dt: Res<DeltaTime>) {
    t.0 += dt.0;
}
/// Provides global simulation time tracking for deterministic, fixed-step simulations.
///
/// # Features:
/// - Adds the [`Time`] resources.
/// - Automatically advances [`Time`] by [`DeltaTime`] every frame.
/// - Runs the time advancement system early in the frame (`First` schedule).
///
/// # Example Usage:
/// ```rust
/// use bevy_ecs::prelude::*;
/// use bevy_app::prelude::*;
/// use rustpower::timeseries::sim_time::Time;
/// fn print_time(t: Res<Time>) {
///     println!("Sim Time: {}", t.elapsed_seconds());
/// }
/// let mut app = App::default();
/// app.add_systems(Update, print_time);
/// ```
///
/// # Scheduling Behavior:
/// - [`advance`] is run in the [`First`] schedule to ensure systems in `Update` see the updated time.
#[derive(Default)]
pub struct TimePlugin;

impl Plugin for TimePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Time>();
        app.add_systems(First, advance);
    }
}
