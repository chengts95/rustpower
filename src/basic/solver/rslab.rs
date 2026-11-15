use rslab::{
    AutoSettings, AutoSolver, CscMatrix, GeneralCsc, LdltSolver, LdltSymbolic, SolverSettings,
};

use super::Solve;

#[derive(Default)]
pub struct RslabLuSolver {
    solver: Option<AutoSolver<'static, f64>>,
}

impl RslabLuSolver {
    pub fn resolve(&mut self, b: &mut [f64]) -> Result<(), &'static str> {
        let solver = self
            .solver
            .as_ref()
            .ok_or("RSLAB LU has no factorization")?;
        let solution = solver.solve(b).map_err(|_| "RSLAB LU solve failed")?;
        b.copy_from_slice(&solution);
        Ok(())
    }
}

impl Solve for RslabLuSolver {
    fn solve(
        &mut self,
        ap: &mut [usize],
        ai: &mut [usize],
        ax: &mut [f64],
        b: &mut [f64],
        n: usize,
    ) -> Result<(), &'static str> {
        if n == 0 {
            return Ok(());
        }
        if self.solver.is_none() {
            let matrix = GeneralCsc {
                n,
                col_ptr: ap.to_vec(),
                row_idx: ai.to_vec(),
                values: ax.to_vec(),
            };
            self.solver = Some(
                AutoSolver::factor(&matrix, &AutoSettings::default())
                    .map_err(|_| "RSLAB LU factorization failed")?,
            );
        } else {
            let solver = self.solver.as_mut().expect("solver was initialized");
            solver.values_mut().copy_from_slice(ax);
            solver
                .refactor(&AutoSettings::default())
                .map_err(|_| "RSLAB LU refactorization failed")?;
        }
        self.resolve(b)
    }

    fn resolve(&mut self, b: &mut [f64]) -> Result<(), &'static str> {
        RslabLuSolver::resolve(self, b)
    }

    fn reset(&mut self) {
        self.solver = None;
    }
}

#[derive(Default)]
pub struct RslabLdltSolver {
    symbolic: Option<LdltSymbolic>,
    factor: Option<LdltSolver<f64>>,
}

impl RslabLdltSolver {
    pub fn resolve(&mut self, b: &mut [f64]) -> Result<(), &'static str> {
        let factor = self
            .factor
            .as_ref()
            .ok_or("RSLAB LDLT has no factorization")?;
        let solution = factor.solve(b).map_err(|_| "RSLAB LDLT solve failed")?;
        b.copy_from_slice(&solution);
        Ok(())
    }
}

impl Solve for RslabLdltSolver {
    fn solve(
        &mut self,
        ap: &mut [usize],
        ai: &mut [usize],
        ax: &mut [f64],
        b: &mut [f64],
        n: usize,
    ) -> Result<(), &'static str> {
        if n == 0 {
            return Ok(());
        }
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        col_ptr.push(0);
        for column in 0..n {
            for p in ap[column]..ap[column + 1] {
                if ai[p] >= column {
                    row_idx.push(ai[p]);
                    values.push(ax[p]);
                }
            }
            col_ptr.push(row_idx.len());
        }
        let matrix = CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        };
        let settings = SolverSettings::default();
        if self.symbolic.is_none() {
            self.symbolic = Some(
                LdltSymbolic::analyze(&matrix, &settings)
                    .map_err(|_| "RSLAB LDLT analysis failed")?,
            );
        }
        self.factor = Some(
            self.symbolic
                .as_ref()
                .expect("symbolic was initialized")
                .factor(&matrix, &settings)
                .map_err(|_| "RSLAB LDLT factorization failed")?,
        );
        self.resolve(b)
    }

    fn resolve(&mut self, b: &mut [f64]) -> Result<(), &'static str> {
        RslabLdltSolver::resolve(self, b)
    }

    fn reset(&mut self) {
        self.symbolic = None;
        self.factor = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_and_ldlt_solve_and_resolve() {
        let mut ap = [0, 2, 4];
        let mut ai = [0, 1, 0, 1];
        for (name, mut b) in [("lu", [1.0, 2.0]), ("ldlt", [1.0, 2.0])] {
            let mut second = b;
            match name {
                "lu" => {
                    let mut solver = RslabLuSolver::default();
                    let mut values = [2.0, 1.0, 1.0, 3.0];
                    solver
                        .solve(&mut ap, &mut ai, &mut values, &mut b, 2)
                        .unwrap();
                    solver.resolve(&mut second).unwrap();
                }
                _ => {
                    let mut solver = RslabLdltSolver::default();
                    let mut values = [2.0, 1.0, 1.0, 3.0];
                    solver
                        .solve(&mut ap, &mut ai, &mut values, &mut b, 2)
                        .unwrap();
                    solver.resolve(&mut second).unwrap();
                }
            }
            assert!((b[0] - 0.2).abs() < 1e-12);
            assert!((b[1] - 0.6).abs() < 1e-12);
            assert_eq!(b, second);
        }
    }
}
