/// The `powerflow` module contains all logic related to the steady-state power flow calculation
/// in the ECS-based power system simulation framework.
///
/// It integrates initialization routines, system-level solving stages,
/// generator reactive limits enforcement, result extraction, and structural updates.
///
/// This module is a key part of the simulation backend, handling the Newton-Raphson iteration
/// and constraint scheduling mechanisms in coordination with ECS world data.
pub mod init; // System and resource initialization logic
pub mod nonlinear_schedule;
pub mod qlim; // Generator reactive power limit handling
pub mod result_extract; // Snapshot and result extraction into simulation state
pub mod structure_update; // Dynamic structural updates triggered by simulation stages
pub mod systems; // Core system stages for power flow iteration // Scheduler for non-linear solve steps (e.g., Q-limit enforcement)

/// Re-exports commonly used symbols from `init` and `systems` for easy access.

pub mod prelude {
    pub use super::init::*;
    pub use super::systems::*;
}
