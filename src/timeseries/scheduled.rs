use bevy_app::App;
use bevy_app::Plugin;
use bevy_app::PostUpdate;
use bevy_ecs::prelude::*;
use nalgebra::Complex;
use nalgebra::SimdComplexField;
use serde::Deserialize;
use serde::Serialize;
use std::collections::VecDeque;

use crate::basic::ecs::elements::*;
use crate::timeseries::sim_time::Time;

/// Represents a dynamic ECS-side action scheduled for execution at a specific simulation time.
///
/// The `action` is a boxed closure that mutates the world directly.
pub struct ScheduledDynAction {
    /// Time at which this action should execute (in seconds).
    pub execute_at: f64,
    /// Action closure to execute on the world.
    pub action: Box<dyn FnMut(&mut World) + Send + Sync>,
}

/// ECS component storing a queue of dynamic scheduled actions to be executed in the future.
#[derive(Component)]
pub struct ScheduledDynActions {
    pub queue: VecDeque<ScheduledDynAction>,
}

/// Enum representing a static, serializable scheduled event.
///
/// Each variant corresponds to a well-defined ECS mutation,
/// such as changing a target power or voltage at a specified bus.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ScheduledActionKind {
    /// Set real power target (P) in MW for a bus.
    SetTargetPMW { bus: i64, value: f64 },
    /// Set reactive power target (Q) in MVar for a bus.
    SetTargetQMvar { bus: i64, value: f64 },
    /// Set voltage magnitude in per-unit.
    SetTargetVM { bus: i64, value: f64 },
    /// Set voltage angle in degrees.
    SetTargetVa { bus: i64, value: f64 },
}

/// Represents one static action with an execution timestamp.
///
/// These actions are deterministic and serializable.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ScheduledStaticAction {
    pub execute_at: f64,
    pub action: ScheduledActionKind,
}

/// ECS component storing a queue of static scheduled actions.
#[derive(Component, Serialize, Deserialize, Clone)]
pub struct ScheduledStaticActions {
    pub queue: VecDeque<ScheduledStaticAction>,
}

/// Resource used to track and log all executed scheduled actions.
#[derive(Resource, Default, Serialize, Deserialize, Clone, Debug)]
pub struct ScheduledLog {
    pub executed: Vec<ScheduledStaticAction>,
}

/// Safely mutates a component of type `T` on the given entity by queueing the change in a deferred system.
///
/// # Panics
/// If the entity does not have the specified component `T`.
fn write_component<T, F>(commands: &mut Commands, entity: Entity, func: F)
where
    T: Component<Mutability = bevy_ecs::component::Mutable> + 'static,
    F: FnOnce(&mut T) + Send + Sync + 'static,
{
    commands.queue(move |world: &mut World| {
        if let Some(mut comp) = world.entity_mut(entity).get_mut::<T>() {
            func(&mut comp);
        } else {
            panic!(
                "Entity {:?} missing component {:?}",
                entity,
                std::any::type_name::<T>()
            );
        }
    });
}

/// Executes scheduled static actions that are due at the current simulation time.
///
/// For each [`ScheduledStaticActions`] component:
/// - If the current time >= `execute_at`, performs the associated [`ScheduledActionKind`].
/// - Applies changes via deferred `commands.queue(...)`.
/// - Logs all executed actions in [`ScheduledLog`] for traceability.

fn scheduled_action_system(
    time: Res<Time>,
    common: Res<PFCommonData>,
    lut: Res<NodeLookup>,
    mut log: ResMut<ScheduledLog>,
    mut commands: Commands,
    mut query: Query<&mut ScheduledStaticActions>,
) {
    let now = time.elapsed_seconds();
    let sbase_frac = 1.0 / common.sbase;
    for mut sched in &mut query {
        while let Some(action) = sched.queue.front() {
            if action.execute_at <= now {
                let act = sched.queue.pop_front().unwrap();
                let action = act.action.clone();
                match action {
                    ScheduledActionKind::SetTargetPMW { bus, value } => {
                        let entity = lut.get_entity(bus).unwrap();
                        write_component::<SBusInjPu, _>(&mut commands, entity, move |a| {
                            a.0.re = value * sbase_frac;
                        });
                    }
                    ScheduledActionKind::SetTargetQMvar { bus, value } => {
                        let entity = lut.get_entity(bus).unwrap();
                        write_component::<SBusInjPu, _>(&mut commands, entity, move |a| {
                            a.0.im = value * sbase_frac;
                        });
                    }
                    ScheduledActionKind::SetTargetVM { bus, value } => {
                        let entity = lut.get_entity(bus).unwrap();
                        write_component::<VBusPu, _>(&mut commands, entity, move |a| {
                            let angle = a.0.simd_argument();
                            a.0 = Complex::from_polar(value, angle);
                        });
                    }
                    ScheduledActionKind::SetTargetVa { bus, value } => {
                        let entity = lut.get_entity(bus).unwrap();
                        write_component::<VBusPu, _>(&mut commands, entity, move |a| {
                            let mag = a.0.norm();
                            a.0 = Complex::from_polar(mag, value.to_radians());
                        });
                    }
                }
                log.executed.push(act);
            } else {
                break;
            }
        }
    }
}
// fn scheduled_dyn_action_system(
//     time: Res<Time>,
//     mut commands: Commands,
//     mut query: Query<&mut ScheduledDynActions>,
// ) {
//     let now = time.elapsed_seconds();

//     for mut sched in &mut query {
//         while let Some(action) = sched.queue.front() {
//             if action.execute_at <= now {
//                 let mut act = sched.queue.pop_front().unwrap();
//                 commands.queue(move |world: &mut World| {
//                     (act.action)(world);
//                 });
//             } else {
//                 break;
//             }
//         }
//     }
// }

/// Plugin for handling time-scheduled system actions in ECS-based simulations.
///
/// Supports:
/// - Static event scheduling (serializable actions applied at exact simulation times)
/// - Logging of all executed actions for auditing
/// - (Optional) Dynamic scheduling via runtime closures [see `ScheduledDynActions`]
///
/// # Resources:
/// - [`ScheduledLog`]: Stores executed action history
///
/// # Systems:
/// - [`scheduled_action_system`] runs during [`PostUpdate`] phase to execute time-due actions.
///
/// # Future Extensions:
/// - Enable dynamic action system
/// - Add filtering by tag/group
#[derive(Default)]
pub struct ScheduledEventPlugin;

impl Plugin for ScheduledEventPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScheduledLog>();
        app.add_systems(PostUpdate, scheduled_action_system);
    }
}
