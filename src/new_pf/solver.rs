use bevy_ecs::prelude::*;
use nalgebra::*;
use nalgebra_sparse::*;
use num_complex::Complex64;

use crate::basic::solver::Solve;
use crate::new_pf::systems::{NetworkOperators, PFOrder};

/// Core Newton-Raphson Solver (Free Function)
///
/// This function is decoupled from Bevy systems for maximum performance and testability.
/// It assumes Ybus is already permuted.
pub fn run_newton_pf<S: Solve>(
    ybus: &CscMatrix<Complex64>,
    sbus: &DVector<Complex64>,
    v_init: &DVector<Complex64>,
    npv: usize,
    npq: usize,
    solver: &mut S,
    max_iter: usize,
    tol: f64,
) -> Result<(DVector<Complex64>, usize), String> {
    crate::basic::newtonpf::newton_pf(
        ybus,
        sbus,
        v_init,
        npv,
        npq,
        Some(tol),
        Some(max_iter),
        solver,
        None,
    )
    .map_err(|(message, _, _)| message)
}

/// Thin Bevy System Wrapper
pub fn newton_pf_system(
    ops: ResMut<NetworkOperators>,
    _order: Res<PFOrder>,
    // TODO: Add queries for current voltage and setpoints
) {
    let Some(_ybus) = &ops.ybus else { return };
    // Integration with ECS components would go here.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct NoLinearSolve;
    impl Solve for NoLinearSolve {
        fn solve(
            &mut self,
            _: &mut [usize],
            _: &mut [usize],
            _: &mut [f64],
            _: &mut [f64],
            _: usize,
        ) -> Result<(), &'static str> {
            panic!("the initial residual already satisfies the infinity-norm tolerance");
        }
        fn reset(&mut self) {}
    }

    #[test]
    fn initial_convergence_uses_inf_norm() {
        let ybus = CscMatrix::identity(2);
        let v = DVector::from_element(2, Complex64::new(1.0, 0.0));
        // Both retained residuals are below tol, but their Euclidean norm is above it.
        let sbus = DVector::from_vec(vec![
            Complex64::new(0.25, -0.75),
            Complex64::new(99.0, 99.0),
        ]);
        let (actual, iterations) =
            run_newton_pf(&ybus, &sbus, &v, 0, 1, &mut NoLinearSolve, 0, 1.0).unwrap();
        assert_eq!(iterations, 0);
        assert_eq!(actual, v);
    }
}
