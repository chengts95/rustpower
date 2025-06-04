use bevy_app::{App, First, Plugin};
use bevy_ecs::prelude::*;
use derive_more::derive::*;
use serde::{Deserialize, Serialize};
#[derive(PartialEq, Default, From, Into, Add, Mul, Div, Resource, Serialize, Deserialize)]
pub struct DeltaTime(pub f64); // in seconds
#[derive(PartialEq, Default, From, Into, Add, Mul, Div, Resource, Serialize, Deserialize)]
pub struct Time(pub f64); // in seconds

impl Time{
    pub fn elapsed_seconds(&self) -> f64 {
        self.0
    }
}

pub fn advance(mut t: ResMut<Time>, dt: Res<DeltaTime>) {
    t.0 += dt.0;
}
#[derive(Default)]
pub struct TimePlugin;
impl Plugin for TimePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Time>();
        app.add_systems(First, advance);
    }
}
