use std::time::{Duration, Instant};

use nalgebra::DVector;
use num_complex::Complex64;

use crate::basic::{
    ecs::powerflow::systems::PowerFlowMat,
    solver::{DefaultLmSolver, Solve},
};

use super::{LmDriver, LmOptions};

/// Solver settings for one LMPF run. The prepared matrix structure is reused.
#[derive(Clone, Debug)]
pub struct LmpfRunOptions {
    pub tol: f64,
    pub max_iter: usize,
    /// Use the exact second-order Hessian. False selects Gauss-Newton LM.
    pub exact: bool,
    pub damping: LmOptions,
}

impl Default for LmpfRunOptions {
    fn default() -> Self {
        Self {
            tol: 1e-6,
            max_iter: 100,
            exact: true,
            damping: LmOptions::default(),
        }
    }
}

impl LmpfRunOptions {
    pub fn validate(&self) -> Result<(), String> {
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err("LMPF tolerance must be finite and positive".into());
        }
        if self.max_iter == 0 {
            return Err("LMPF max_iter must be positive".into());
        }
        self.damping.validate().map_err(str::to_owned)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmpfStatus {
    Converged,
    NotConverged,
}

impl LmpfStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::NotConverged => "not-converged",
        }
    }
}

/// Independent result of one LMPF run, in physical bus order.
#[derive(Clone, Debug)]
pub struct LmpfReport {
    pub status: LmpfStatus,
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: f64,
    pub voltage: DVector<Complex64>,
    pub elapsed: Duration,
}

/// Reusable fixed-topology LMPF study with private numerical workspaces.
pub struct PreparedLmpf {
    mat: PowerFlowMat,
    driver: LmDriver,
    solver: DefaultLmSolver,
}

impl PreparedLmpf {
    pub fn backend(&self) -> &'static str {
        if cfg!(feature = "qdldl") {
            "qdldl"
        } else {
            "default"
        }
    }

    pub fn new(mat: PowerFlowMat) -> Self {
        let driver = LmDriver::build(
            &mat.y_bus,
            mat.npv,
            mat.npq,
            mat.s_bus.iter().copied().collect(),
        );
        Self {
            mat,
            driver,
            solver: DefaultLmSolver::default(),
        }
    }

    pub fn run(&mut self, options: LmpfRunOptions) -> Result<LmpfReport, String> {
        options.validate()?;
        let started = Instant::now();
        let mut voltage = self.mat.v_bus_init.iter().copied().collect::<Vec<_>>();
        let result = self.driver.solve_lm_with_options(
            &self.mat.y_bus,
            &mut self.solver,
            &mut voltage,
            options.exact,
            options.tol,
            options.max_iter,
            &options.damping,
        );
        let from_perm = self.mat.from_perm.clone();
        let mut physical = DVector::zeros(voltage.len());
        for (solver, &original) in from_perm.iter().enumerate() {
            physical[original] = voltage[solver];
        }
        let status = if result.converged {
            LmpfStatus::Converged
        } else {
            LmpfStatus::NotConverged
        };
        Ok(LmpfReport {
            status,
            converged: result.converged,
            iterations: result.iterations,
            residual_norm: result.res_inf,
            voltage: physical,
            elapsed: started.elapsed(),
        })
    }

    pub fn reset(&mut self) {
        self.solver.reset();
    }
}
