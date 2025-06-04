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

pub struct ScheduledDynAction {
    pub execute_at: f64,
    pub action: Box<dyn FnMut(&mut World) + Send + Sync>,
}
#[derive(Component)]
pub struct ScheduledDynActions {
    pub queue: VecDeque<ScheduledDynAction>,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ScheduledActionKind {
    SetTargetPMW { bus: i64, value: f64 },
    SetTargetQMvar { bus: i64, value: f64 },
    SetTargetVM { bus: i64, value: f64 },
    SetTargetVa { bus: i64, value: f64 },
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ScheduledStaticAction {
    pub execute_at: f64,
    pub action: ScheduledActionKind,
}
#[derive(Component, Serialize, Deserialize, Clone)]
pub struct ScheduledStaticActions {
    pub queue: VecDeque<ScheduledStaticAction>,
}
#[derive(Resource, Default, Serialize, Deserialize, Clone, Debug)]
pub struct ScheduledLog {
    pub executed: Vec<ScheduledStaticAction>,
}

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
#[derive(Default)]
pub struct ScheduledEventPlugin;

impl Plugin for ScheduledEventPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScheduledLog>();
        app.add_systems(PostUpdate, scheduled_action_system);
    }
}
