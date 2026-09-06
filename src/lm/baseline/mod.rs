//! COO装配基线：每次重建COO，同时复用CSC输入缓冲区及线性求解器。

#[cfg(feature = "qdldl")]
pub mod aug_coo;
#[cfg(feature = "qdldl")]
pub mod full_slice;
#[cfg(feature = "qdldl")]
mod tests;

/// COO转换仍每次执行；固定输入地址使QDLDL能够复用符号分析。
#[cfg(feature = "qdldl")]
#[derive(Default)]
struct CooSystem {
    cols: Vec<usize>,
    rows: Vec<usize>,
    values: Vec<f64>,
    solver: crate::basic::solver::QDLDLSolver,
}

#[cfg(feature = "qdldl")]
impl CooSystem {
    fn update(&mut self, matrix: &nalgebra_sparse::CscMatrix<f64>) {
        use crate::basic::solver::Solve;
        if self.cols == matrix.col_offsets() && self.rows == matrix.row_indices() {
            self.values.copy_from_slice(matrix.values());
        } else {
            self.solver.reset();
            self.cols = matrix.col_offsets().to_vec();
            self.rows = matrix.row_indices().to_vec();
            self.values = matrix.values().to_vec();
        }
    }

    fn solve(&mut self, rhs: &mut [f64]) -> bool {
        use crate::basic::solver::Solve;
        self.solver
            .solve(
                &mut self.cols,
                &mut self.rows,
                &mut self.values,
                rhs,
                rhs.len(),
            )
            .is_ok()
    }
}
