use crate::cpf::{CpfOptions, CpfReport as NativeReport};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::{prelude::*, types::PyDict};

#[pyclass(name = "CpfNose")]
pub struct Nose {
    lambda: f64,
    voltage: Vec<num_complex::Complex64>,
}

#[pymethods]
impl Nose {
    #[getter]
    fn lambda_(&self) -> f64 {
        self.lambda
    }

    #[getter]
    fn voltage<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<num_complex::Complex64>> {
        self.voltage.clone().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!("CpfNose(lambda={:.8})", self.lambda)
    }
}

#[pyclass(name = "CpfReport")]
pub struct Report {
    inner: NativeReport,
}

#[pymethods]
impl Report {
    #[getter]
    fn passed_nose(&self) -> bool {
        self.inner.passed_nose()
    }

    #[getter]
    fn lambda_max(&self) -> f64 {
        self.inner.lambda_max()
    }

    #[getter]
    fn stop_reason(&self) -> &'static str {
        self.inner.stop_reason()
    }

    #[getter]
    fn nose(&self) -> Option<Nose> {
        self.inner.nose.as_ref().map(|nose| Nose {
            lambda: nose.lambda,
            voltage: nose.voltage.as_slice().to_vec(),
        })
    }

    #[getter]
    fn lambda_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner
            .points
            .iter()
            .map(|point| point.lambda)
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    #[getter]
    fn voltages<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<num_complex::Complex64>>> {
        let rows = self
            .inner
            .points
            .iter()
            .map(|point| point.voltage.as_slice().to_vec())
            .collect::<Vec<_>>();
        PyArray2::from_vec2(py, &rows).map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "CPF result has inconsistent voltage dimensions: {error}"
            ))
        })
    }

    #[getter]
    fn corrector_iterations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        self.inner
            .points
            .iter()
            .map(|point| point.corrector_iterations as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    #[getter]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = PyDict::new(py);
        stats.set_item("factorizations", self.inner.stats.factorizations)?;
        stats.set_item("resolves", self.inner.stats.resolves)?;
        stats.set_item("elapsed_seconds", self.inner.stats.elapsed.as_secs_f64())?;
        Ok(stats)
    }

    fn __repr__(&self) -> String {
        format!(
            "CpfReport(points={}, passed_nose={}, stop_reason='{}')",
            self.inner.points.len(),
            self.inner.passed_nose(),
            self.inner.stop_reason()
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    grid: &mut crate::basic::ecs::network::PowerGrid,
    tol: f64,
    step0: f64,
    step_max: f64,
    step_min: f64,
    step_grow: f64,
    step_shrink: f64,
    max_corr_iter: usize,
    max_points: usize,
    base_max_iter: usize,
) -> PyResult<Report> {
    let options = CpfOptions {
        tol,
        step0,
        step_max,
        step_min,
        step_grow,
        step_shrink,
        max_corr_iter,
        max_points,
        base_max_iter,
    };
    options
        .validate()
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    grid.run_cpf(options)
        .map(|inner| Report { inner })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}
