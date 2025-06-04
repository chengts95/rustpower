use bevy_app::{MainScheduleOrder, prelude::*};
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::{ExecutorKind, ScheduleLabel};

use super::systems::PowerFlowResult;
use crate::basic::ecs::network::ecs_run_pf;
use crate::prelude::ecs::network::SolverStage::Solve;

/// A custom schedule label used to trigger nonlinear error checking after each solver pass.
/// Typically placed after the `Update` stage to determine convergence and whether further iterations are needed.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NonLinearErrorCheck;

/// Stores the convergence status of the current iteration process.
/// Updated after each NR solve pass.
#[derive(Resource, Clone, Default)]
pub struct ConvergedResult {
    pub converged: NonlinearConvType, // Tracks current convergence status
}

/// Represents the state of convergence for a nonlinear system.
/// - `Converged`: The iteration has reached a solution.
/// - `Continue`: Iteration should proceed.
/// - `MaxIter`: Maximum number of iterations has been reached.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum NonlinearConvType {
    #[default]
    Converged,
    Continue,
    MaxIter,
}

/// Plugin responsible for setting up custom iteration and convergence checking schedules
/// used in nonlinear solvers such as Newton-Raphson for power flow analysis.
pub struct NonLinearSchedulePlugin;

/// Updates the convergence status resource (`ConvergedResult`) based on the outcome of power flow computation.
/// This system is expected to run after each nonlinear solve pass.
pub fn update_convergence(mut res: ResMut<ConvergedResult>, pf_res: Res<PowerFlowResult>) {
    if pf_res.converged {
        res.converged = NonlinearConvType::Converged;
    } else {
        res.converged = NonlinearConvType::MaxIter;
    }
}

/// Runs the sequence of schedules for one nonlinear iteration cycle.
/// Starts from `Startup`, executes `Main`-ordered schedules in sequence,
/// and jumps back to `Update` if convergence is not yet achieved.
/// This effectively implements a loop over schedule stages until convergence.
///
/// # Behavior
/// - Executes all labels in `MainScheduleOrder`
/// - When `NonLinearErrorCheck` is reached:
///   - If converged: stop
///   - If max iterations: panic
///   - Else: rewind to `PreUpdate` stage and repeat
pub fn run_outer_iteration(
    world: &mut World,
    mut run_at_least_once: Local<bool>,
    mut cached_nr_idx: Local<usize>,
) {
    // First-time setup: run all Startup stages and locate `PreUpdate` index
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

    // Main iteration loop over schedules in `MainScheduleOrder`
    world.resource_scope(|world, order: Mut<MainScheduleOrder>| {
        let mut index = 0;

        while index < order.labels.len() {
            let label = order.labels[index];
            let _ = world.try_run_schedule(label);

            index += 1;

            if label == NonLinearErrorCheck.intern() {
                let c = world.resource_mut::<ConvergedResult>();
                match c.converged {
                    NonlinearConvType::Converged => {
                        // Exit iteration loop
                    }
                    NonlinearConvType::MaxIter => {
                        panic!("Max Iteration reached");
                    }
                    _ => {
                        // Rewind to `PreUpdate` and continue iteration
                        index = *cached_nr_idx;
                    }
                }
            }
        }
    });
}

impl Plugin for NonLinearSchedulePlugin {
    fn build(&self, app: &mut App) {
        // 1. Initialize convergence result resource
        app.init_resource::<ConvergedResult>();

        // 2. Register the main iteration schedule (label = `Main`)
        let mut main_schedule = Schedule::new(Main);
        main_schedule.set_executor_kind(ExecutorKind::SingleThreaded); // deterministic
        app.add_schedule(main_schedule);

        // 3. Add a custom post-update stage for NR convergence checking
        let mut nl_post_schedule = Schedule::new(NonLinearErrorCheck);
        nl_post_schedule.set_executor_kind(ExecutorKind::SingleThreaded);
        app.add_schedule(nl_post_schedule);

        // 4. Register outer iteration driver and convergence updater systems
        app.add_systems(Main, run_outer_iteration);
        app.add_systems(Update, update_convergence.after(ecs_run_pf).in_set(Solve));

        // 5. Insert NonLinearErrorCheck into the schedule order after Update
        let mut order = app.world_mut().resource_mut::<MainScheduleOrder>();
        order.insert_after(Update, NonLinearErrorCheck);
    }
}
