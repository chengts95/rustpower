use crate::basic::system::*;
use crate::io::pandapower;
use bevy_ecs::prelude::*;
use derive_more::{Deref, DerefMut};
#[derive(Debug, Component)]
pub struct PiBranch {}

#[derive(Debug, Resource, Deref, DerefMut)]
pub struct PPNetwork(pub pandapower::Network);


