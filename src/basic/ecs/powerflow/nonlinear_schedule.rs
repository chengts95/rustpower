use bevy_app::{MainScheduleOrder, prelude::*};
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::{ExecutorKind, ScheduleLabel};

use super::systems::PowerFlowResult;
use crate::basic::ecs::network::ecs_run_pf;
use crate::prelude::ecs::network::SolverStage::Solve;

// `NonLinearErrorCheck` is the schedule for checking NR convergence and triggering further iteration
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NonLinearErrorCheck;
/// Convergence tracking status
#[derive(Resource, Clone, Default)]
pub struct ConvergedResult {
    pub converged: NonlinearConvType, // iteration status
}
/// Enum for nonlinear system convergence type
#[derive(Clone, Debug, PartialEq, Default)]
pub enum NonlinearConvType {
    #[default]
    Converged, // error is below threshold
    Continue, // more iterations needed
    MaxIter,  // max iterations reached
}

pub struct NonLinearSchedulePlugin;
pub fn update_convergence(mut res: ResMut<ConvergedResult>, pf_res: Res<PowerFlowResult>) {
    if pf_res.converged {
        res.converged = NonlinearConvType::Converged;
    } else {
        res.converged = NonlinearConvType::MaxIter;
    }
}
pub fn run_outer_iteration(
    world: &mut World,
    mut run_at_least_once: Local<bool>,
    mut cached_nr_idx: Local<usize>,
) {
    if !*run_at_least_once {
        world.resource_scope(|world, order: Mut<MainScheduleOrder>| {
            for &label in &order.startup_labels {
                let _ = world.try_run_schedule(label);
            }
            *cached_nr_idx = order
                .labels
                .iter()
                .enumerate()
                .find(|x| x.1.intern() == PreUpdate.intern())
                .map(|x| x.0)
                .unwrap();
        });

        *run_at_least_once = true;
    }

    world.resource_scope(|world, order: Mut<MainScheduleOrder>| {
        let mut index = 0;

        while index < order.labels.len() {
            let label = order.labels[index];

            let _ = world.try_run_schedule(label);

            index += 1;
            if label == NonLinearErrorCheck.intern() {
                let c = world.resource_mut::<ConvergedResult>();
                match c.converged {
                    NonlinearConvType::Converged => {}
                    NonlinearConvType::MaxIter => {
                        panic!("Max Iteration reached");
                    }
                    _ => {
                        index = *cached_nr_idx;
                    }
                }
            }
        }
    });
}

impl Plugin for NonLinearSchedulePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConvergedResult>();

        let mut main_schedule = Schedule::new(Main);
        main_schedule.set_executor_kind(ExecutorKind::SingleThreaded);
        app.add_schedule(main_schedule);

        // 2. 设置 NR 检查阶段（Update 之后）
        let mut nl_post_schedule = Schedule::new(NonLinearErrorCheck);
        nl_post_schedule.set_executor_kind(ExecutorKind::SingleThreaded);
        app.add_schedule(nl_post_schedule);

        // 3. 注册主迭代驱动系统
        app.add_systems(Main, run_outer_iteration);
        app.add_systems(Update, update_convergence.after(ecs_run_pf).in_set(Solve));

        // 4. 修改调度顺序（从 bevy_app::MainScheduleOrder）
        let mut order = app.world_mut().resource_mut::<MainScheduleOrder>();

        order.insert_after(Update, NonLinearErrorCheck); // 类似 PostUpdate
    }
}
