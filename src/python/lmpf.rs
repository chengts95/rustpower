use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::lm::{LmpfRunOptions, PreparedLmpf};

use super::grid::PowerGrid;

#[pyclass(name = "LmpfReport")]
pub struct Report {
    inner: crate::lm::LmpfReport,
}

#[pymethods]
impl Report {
    #[getter]
    fn status(&self) -> &'static str {
        self.inner.status.as_str()
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    fn residual_norm(&self) -> f64 {
        self.inner.residual_norm
    }

    #[getter]
    fn voltage<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<num_complex::Complex64>> {
        self.inner.voltage.as_slice().to_vec().into_pyarray(py)
    }

    #[getter]
    fn elapsed_seconds(&self) -> f64 {
        self.inner.elapsed.as_secs_f64()
    }

    fn __bool__(&self) -> bool {
        self.inner.converged
    }

    fn __repr__(&self) -> String {
        format!(
            "LmpfReport(converged={}, iterations={}, residual_norm={:.4e})",
            self.inner.converged, self.inner.iterations, self.inner.residual_norm
        )
    }
}

#[pyclass(name = "LmpfStudy", unsendable)]
pub struct Study {
    owner: Py<PowerGrid>,
    prepared: PreparedLmpf,
}

#[pymethods]
impl Study {
    #[pyo3(signature = (*, tol=1e-6, max_iter=100, exact=true, initial_mu=1e-2, mu_increase=2.0, mu_decrease=3.0, max_trials=30))]
    fn run(
        &mut self,
        tol: f64,
        max_iter: usize,
        exact: bool,
        initial_mu: f64,
        mu_increase: f64,
        mu_decrease: f64,
        max_trials: usize,
    ) -> PyResult<Report> {
        let options = LmpfRunOptions {
            tol,
            max_iter,
            exact,
            damping: crate::lm::LmOptions {
                initial_mu,
                mu_increase,
                mu_decrease,
                max_trials,
                ..Default::default()
            },
        };
        options
            .validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        self.prepared
            .run(options)
            .map(|inner| Report { inner })
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    fn reset(&mut self) {
        self.prepared.reset();
    }

    #[getter]
    fn backend(&self) -> &'static str {
        self.prepared.backend()
    }

    #[getter]
    fn owner(&self, py: Python<'_>) -> Py<PowerGrid> {
        self.owner.clone_ref(py)
    }
}

pub fn prepare(grid: Py<PowerGrid>, py: Python<'_>, backend: &str) -> PyResult<Study> {
    let backend = match backend {
        "qdldl" => crate::lm::LmpfBackend::Qdldl,
        #[cfg(feature = "rslab")]
        "rslab-ldlt" => crate::lm::LmpfBackend::RslabLdlt,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "LMPF backend must be 'qdldl' or 'rslab-ldlt'",
            ));
        }
    };
    let prepared = grid.borrow_mut(py).inner.prepare_lmpf_with_backend(backend);
    Ok(Study {
        owner: grid,
        prepared,
    })
}
