//! Opt-in AMD preparation around the unchanged native Newton solver.
use super::solver::NewtonSolver;
use crate::basic::ybus_ordering::default_backend;
use rustpower_amd::klu::{KluOrdering, YbusOrderingBackend, YbusOrderingStats};
use std::ops::{Deref, DerefMut};

pub struct AmdNewtonSolver {
    pub inner: NewtonSolver,
    pub ordering: KluOrdering,
}
impl Default for AmdNewtonSolver {
    fn default() -> Self {
        Self::new()
    }
}
impl AmdNewtonSolver {
    pub fn new() -> Self {
        Self {
            inner: NewtonSolver::new(),
            ordering: KluOrdering::new(default_backend()),
        }
    }
    pub fn solve(&mut self, max_iter: usize, tol: f64) -> Result<(bool, f64), &'static str> {
        let mat = self
            .inner
            .mat
            .as_ref()
            .ok_or("Solver context not initialized; call setup_context first")?;
        crate::basic::ybus_ordering::prepare_klu(
            &mut self.ordering,
            &mut self.inner.linear_solver,
            &mat.y_bus,
            mat.npv,
            mat.npq,
            Some(&mut self.inner.cache),
        );
        self.inner.solve(max_iter, tol)
    }
    pub fn set_ybus_ordering(&mut self, enabled: bool) {
        self.ordering
            .set_ybus_ordering(&mut self.inner.linear_solver.0, enabled);
    }
    pub fn set_ybus_ordering_backend(&mut self, backend: YbusOrderingBackend) {
        self.ordering
            .set_ybus_ordering_backend(&mut self.inner.linear_solver.0, backend);
    }
    pub fn ybus_ordering_stats(&self) -> YbusOrderingStats {
        self.ordering.ybus_ordering_stats()
    }
    pub fn ybus_ordering_backend(&self) -> YbusOrderingBackend {
        self.ordering.ybus_ordering_backend()
    }
}
impl Deref for AmdNewtonSolver {
    type Target = NewtonSolver;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl DerefMut for AmdNewtonSolver {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
