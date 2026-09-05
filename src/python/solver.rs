//! Python bindings for the native Newton-Raphson power flow solver.

#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArrayMethods};
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
fn extract_complex(obj: &Bound<'_, PyAny>) -> PyResult<num_complex::Complex64> {
    if let Ok(c) = obj.downcast::<pyo3::types::PyComplex>() {
        Ok(num_complex::Complex64::new(c.real(), c.imag()))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(num_complex::Complex64::new(f, 0.0))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected complex or float value",
        ))
    }
}

/// Unified index extraction supporting int32, int64, usize numpy arrays, and Python lists.
#[cfg(feature = "python")]
fn extract_indices(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(arr) = obj.downcast::<numpy::PyArray1<i64>>() {
        let ro = arr.readonly();
        Ok(ro.as_slice()?.iter().map(|&x| x as usize).collect())
    } else if let Ok(arr) = obj.downcast::<numpy::PyArray1<i32>>() {
        let ro = arr.readonly();
        Ok(ro.as_slice()?.iter().map(|&x| x as usize).collect())
    } else if let Ok(arr) = obj.downcast::<numpy::PyArray1<usize>>() {
        let ro = arr.readonly();
        Ok(ro.as_slice()?.to_vec())
    } else if let Ok(list) = obj.extract::<Vec<usize>>() {
        Ok(list)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected 1D integer array or list of indices (int32, int64, or usize)",
        ))
    }
}

/// Zero-copy (when possible) access to 1D complex or float arrays.
#[cfg(feature = "python")]
fn with_complex_slice<R>(
    obj: &Bound<'_, PyAny>,
    f: impl FnOnce(&[num_complex::Complex64]) -> PyResult<R>,
) -> PyResult<R> {
    if let Ok(arr) = obj.downcast::<numpy::PyArray1<num_complex::Complex64>>() {
        let ro = arr.readonly();
        f(ro.as_slice()?)
    } else if let Ok(arr) = obj.downcast::<numpy::PyArray1<f64>>() {
        let ro = arr.readonly();
        let vec: Vec<num_complex::Complex64> = ro
            .as_slice()?
            .iter()
            .map(|&x| num_complex::Complex64::new(x, 0.0))
            .collect();
        f(&vec)
    } else if let Ok(seq) = obj.downcast::<pyo3::types::PySequence>() {
        let len = seq.len()?;
        let mut vec = Vec::with_capacity(len);
        for i in 0..len {
            let item = seq.get_item(i)?;
            vec.push(extract_complex(&item)?);
        }
        f(&vec)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Expected 1D complex or float array",
        ))
    }
}

#[cfg(feature = "python")]
fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    let msg = e.to_string();
    if msg.contains("out of bounds") || msg.contains("out of range") {
        PyErr::new::<pyo3::exceptions::PyIndexError, _>(msg)
    } else if msg.contains("mismatch") {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)
    } else {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg)
    }
}

use crate::native::solver::NewtonSolver as NativeNewtonSolver;

/// High-performance Newton-Raphson power flow solver.
///
/// Direct wrapper around native Rust `crate::native::solver::NewtonSolver`.
#[cfg(feature = "python")]
#[pyclass(unsendable)]
pub struct NewtonSolver {
    inner: NativeNewtonSolver,
}

// ============================================================================
// 1. 状态管理与初始化配置 (State Management & Setup Context)
// ============================================================================
#[cfg(feature = "python")]
#[pymethods]
impl NewtonSolver {
    #[new]
    fn new() -> Self {
        Self {
            inner: NativeNewtonSolver::new(),
        }
    }

    /// Setup solver context by specifying node type partitions directly.
    ///
    /// Accepts integer arrays (int32/int64/usize or list) and complex arrays (complex128/float64).
    /// Automatically concatenates `[pq, pv, slack_bus]` to construct the `[PQ | PV | Slack]`
    /// solver permutation vector, builds inverse mapping, and infers `npv` and `npq`.
    #[pyo3(signature = (y_indptr, y_indices, y_data, s_bus, v_init, pq, pv, slack_bus))]
    fn setup_from_nodes(
        &mut self,
        y_indptr: &Bound<'_, PyAny>,
        y_indices: &Bound<'_, PyAny>,
        y_data: &Bound<'_, PyAny>,
        s_bus: &Bound<'_, PyAny>,
        v_init: &Bound<'_, PyAny>,
        pq: &Bound<'_, PyAny>,
        pv: &Bound<'_, PyAny>,
        slack_bus: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let indptr = extract_indices(y_indptr)?;
        let indices = extract_indices(y_indices)?;
        let pq_vec = extract_indices(pq)?;
        let pv_vec = extract_indices(pv)?;
        let slack_vec = extract_indices(slack_bus)?;

        with_complex_slice(y_data, |data| {
            with_complex_slice(s_bus, |s_raw| {
                with_complex_slice(v_init, |v_raw| {
                    self.inner
                        .setup_from_nodes(
                            v_raw.len(),
                            &indptr,
                            &indices,
                            data,
                            s_raw,
                            v_raw,
                            &pq_vec,
                            &pv_vec,
                            &slack_vec,
                        )
                        .map_err(to_py_err)
                })
            })
        })
    }

    /// Setup solver context using explicitly provided permutation vectors.
    ///
    /// Accepts integer arrays (int32/int64/usize or list) and complex arrays (complex128/float64).
    #[pyo3(signature = (y_indptr, y_indices, y_data, s_bus, v_init, p_vec_in, p_inv_in, npv, npq))]
    fn setup_context(
        &mut self,
        y_indptr: &Bound<'_, PyAny>,
        y_indices: &Bound<'_, PyAny>,
        y_data: &Bound<'_, PyAny>,
        s_bus: &Bound<'_, PyAny>,
        v_init: &Bound<'_, PyAny>,
        p_vec_in: &Bound<'_, PyAny>,
        p_inv_in: &Bound<'_, PyAny>,
        npv: usize,
        npq: usize,
    ) -> PyResult<()> {
        let indptr = extract_indices(y_indptr)?;
        let indices = extract_indices(y_indices)?;
        let p_vec = extract_indices(p_vec_in)?;
        let p_inv = extract_indices(p_inv_in)?;

        with_complex_slice(y_data, |data| {
            with_complex_slice(s_bus, |s_raw| {
                with_complex_slice(v_init, |v_raw| {
                    // p_vec is from_perm (new -> old), p_inv is to_perm (old -> new)
                    self.inner
                        .setup_context(
                            v_raw.len(),
                            &indptr,
                            &indices,
                            data,
                            s_raw,
                            v_raw,
                            p_inv,
                            p_vec,
                            npv,
                            npq,
                        )
                        .map_err(to_py_err)
                })
            })
        })
    }

    /// Update the entire initial voltage vector (in original bus order).
    fn set_v_init(&mut self, v_init: &Bound<'_, PyAny>) -> PyResult<()> {
        with_complex_slice(v_init, |v_slice| {
            self.inner.set_v_init(v_slice).map_err(to_py_err)
        })
    }

    /// Update initial voltage for a single bus (in original bus index).
    fn set_v_init_at(&mut self, bus: usize, v: &Bound<'_, PyAny>) -> PyResult<()> {
        let val = extract_complex(v)?;
        self.inner.set_v_init_at(bus, val).map_err(to_py_err)
    }

    /// Batch update initial voltage for specified bus indices.
    fn update_v_init_batch(
        &mut self,
        bus_indices: &Bound<'_, PyAny>,
        v_values: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let buses = extract_indices(bus_indices)?;
        with_complex_slice(v_values, |values| {
            self.inner.update_v_init_batch(&buses, values).map_err(to_py_err)
        })
    }

    /// Initial voltage vector in original bus order.
    #[getter]
    fn v_init<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let v = self.inner.v_init().map_err(to_py_err)?;
        Ok(v.into_pyarray(py))
    }

    #[setter]
    fn set_v_init_property(&mut self, v_init: &Bound<'_, PyAny>) -> PyResult<()> {
        self.set_v_init(v_init)
    }

    /// Update the entire bus power injection vector Sbus (in original bus order).
    fn set_s_bus(&mut self, s_bus: &Bound<'_, PyAny>) -> PyResult<()> {
        with_complex_slice(s_bus, |s_slice| {
            self.inner.set_s_bus(s_slice).map_err(to_py_err)
        })
    }

    /// Update Sbus injection for a single bus (in original bus index).
    fn set_s_bus_at(&mut self, bus: usize, s: &Bound<'_, PyAny>) -> PyResult<()> {
        let val = extract_complex(s)?;
        self.inner.set_s_bus_at(bus, val).map_err(to_py_err)
    }

    /// Batch update Sbus injections for specified bus indices.
    fn update_s_bus_batch(
        &mut self,
        bus_indices: &Bound<'_, PyAny>,
        s_values: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let buses = extract_indices(bus_indices)?;
        with_complex_slice(s_values, |values| {
            self.inner.update_s_bus_batch(&buses, values).map_err(to_py_err)
        })
    }

    /// Bus power injection vector Sbus in original bus order.
    #[getter]
    fn s_bus<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let s = self.inner.s_bus().map_err(to_py_err)?;
        Ok(s.into_pyarray(py))
    }

    #[setter]
    fn set_s_bus_property(&mut self, s_bus: &Bound<'_, PyAny>) -> PyResult<()> {
        self.set_s_bus(s_bus)
    }

    /// Explicitly clear cached symbolic pattern, factorizations, and Newton buffers.
    fn clear_cache(&mut self) -> PyResult<()> {
        self.inner.clear_cache();
        Ok(())
    }

    /// Reset internal cache and linear solver state (alias for clear_cache).
    fn reset_cache(&mut self) -> PyResult<()> {
        self.inner.clear_cache();
        Ok(())
    }

    /// Permutation mapping: original bus index -> permuted solver index (`p_inv`).
    #[getter]
    fn to_perm<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<usize>>> {
        let perm = self.inner.to_perm().map_err(to_py_err)?;
        Ok(perm.to_vec().into_pyarray(py))
    }

    /// Alias for `to_perm` (`p_inv`).
    #[getter]
    fn p_inv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<usize>>> {
        self.to_perm(py)
    }

    /// Inverse permutation mapping: permuted solver index -> original bus index (`p_vec`).
    #[getter]
    fn from_perm<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<usize>>> {
        let perm = self.inner.from_perm().map_err(to_py_err)?;
        Ok(perm.to_vec().into_pyarray(py))
    }

    /// Alias for `from_perm` (`p_vec`).
    #[getter]
    fn p_vec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<usize>>> {
        self.from_perm(py)
    }

    /// Number of PV buses configured in the solver.
    #[getter]
    fn npv(&self) -> PyResult<usize> {
        self.inner.npv().map_err(to_py_err)
    }

    /// Number of PQ buses configured in the solver.
    #[getter]
    fn npq(&self) -> PyResult<usize> {
        self.inner.npq().map_err(to_py_err)
    }

    /// Total number of buses configured in the solver.
    #[getter]
    fn n_buses(&self) -> PyResult<usize> {
        self.inner.n_buses().map_err(to_py_err)
    }

    // ========================================================================
    // 2. 核心数值求解 (Core Numerical Solve)
    // ========================================================================

    /// Run the Newton-Raphson power flow solver. Returns True if converged.
    ///
    /// Detailed results and convergence statistics can be inspected via:
    /// - `.converged` (bool)
    /// - `.residual_norm` (float, maximum power mismatch ||F||_inf)
    /// - `.iterations` (int)
    #[pyo3(signature = (max_iter=10, tol=1e-6))]
    fn solve(
        &mut self,
        max_iter: usize,
        tol: f64,
    ) -> PyResult<bool> {
        let (converged, _residual) = self.inner.solve(max_iter, tol).map_err(to_py_err)?;
        Ok(converged)
    }

    /// Maximum power mismatch norm ||F||_inf after the last solve.
    fn get_residual(&self) -> PyResult<f64> {
        Ok(self.inner.residual())
    }

    /// Maximum power mismatch norm ||F||_inf after the last solve.
    #[getter]
    fn residual_norm(&self) -> PyResult<f64> {
        Ok(self.inner.residual())
    }

    /// Get whether the last solve converged.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        self.inner.converged().map_err(to_py_err)
    }

    /// Get the number of iterations taken by the solver.
    fn get_iterations(&self) -> PyResult<usize> {
        self.inner.iterations().map_err(to_py_err)
    }

    /// Number of iterations taken by the solver.
    #[getter(iterations)]
    fn py_iterations(&self) -> PyResult<usize> {
        self.inner.iterations().map_err(to_py_err)
    }

    // ========================================================================
    // 3. 后处理与电气数据读取 (Post-Processing & Data Extraction)
    // ========================================================================

    /// Get the final complex bus voltages in original order.
    fn get_voltage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let v = self.inner.voltage().map_err(to_py_err)?;
        Ok(v.into_pyarray(py))
    }

    /// Get the final voltage magnitudes (p.u.) in original bus order.
    fn get_voltage_magnitude<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let vm = self.inner.vm().map_err(to_py_err)?;
        Ok(vm.into_pyarray(py))
    }

    /// Voltage magnitude vector in original bus order.
    #[getter]
    fn vm<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        self.get_voltage_magnitude(py)
    }

    /// Get the final voltage angles in original bus order.
    #[pyo3(signature = (deg=false))]
    fn get_voltage_angle<'py>(&self, py: Python<'py>, deg: bool) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let va = self.inner.va(deg).map_err(to_py_err)?;
        Ok(va.into_pyarray(py))
    }

    /// Voltage angle vector in radians in original bus order.
    #[getter]
    fn va<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        self.get_voltage_angle(py, false)
    }

    /// Voltage angle vector in degrees in original bus order.
    #[getter]
    fn va_deg<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        self.get_voltage_angle(py, true)
    }

    /// Get the calculated bus power injection vector (S_calc = V * (Ybus * V)^*) in original bus order.
    fn get_scalc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let s = self.inner.scalc().map_err(to_py_err)?;
        Ok(s.into_pyarray(py))
    }

    /// Calculated complex bus power injections S_calc in original bus order.
    #[getter(scalc)]
    fn py_scalc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        self.get_scalc(py)
    }

    /// Get calculated active power injection P_calc in original bus order.
    fn get_p_injections<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let p = self.inner.p_calc().map_err(to_py_err)?;
        Ok(p.into_pyarray(py))
    }

    /// Calculated active power injection P_calc in original bus order.
    #[getter]
    fn p_calc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        self.get_p_injections(py)
    }

    /// Get calculated reactive power injection Q_calc in original bus order.
    fn get_q_injections<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let q = self.inner.q_calc().map_err(to_py_err)?;
        Ok(q.into_pyarray(py))
    }

    /// Calculated reactive power injection Q_calc in original bus order.
    #[getter]
    fn q_calc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        self.get_q_injections(py)
    }

    /// In-place result extraction directly into caller-provided NumPy arrays.
    ///
    /// Zero heap allocations: directly scatters computed electrical quantities into
    /// the provided pre-allocated 1D arrays in original bus order.
    /// Any argument passed as `None` (default) is skipped.
    ///
    /// # Parameters
    /// - `v`: Optional 1D `complex128` array for complex voltages.
    /// - `vm`: Optional 1D `float64` array for voltage magnitudes (p.u.).
    /// - `va`: Optional 1D `float64` array for voltage angles (radians).
    /// - `va_deg`: Optional 1D `float64` array for voltage angles (degrees).
    /// - `scalc`: Optional 1D `complex128` array for complex power injections.
    /// - `p_calc`: Optional 1D `float64` array for active power injections.
    /// - `q_calc`: Optional 1D `float64` array for reactive power injections.
    #[pyo3(signature = (v=None, vm=None, va=None, va_deg=None, scalc=None, p_calc=None, q_calc=None))]
    fn extract_results(
        &self,
        v: Option<&Bound<'_, numpy::PyArray1<num_complex::Complex64>>>,
        vm: Option<&Bound<'_, numpy::PyArray1<f64>>>,
        va: Option<&Bound<'_, numpy::PyArray1<f64>>>,
        va_deg: Option<&Bound<'_, numpy::PyArray1<f64>>>,
        scalc: Option<&Bound<'_, numpy::PyArray1<num_complex::Complex64>>>,
        p_calc: Option<&Bound<'_, numpy::PyArray1<f64>>>,
        q_calc: Option<&Bound<'_, numpy::PyArray1<f64>>>,
    ) -> PyResult<()> {
        let mut v_rw = v.map(|arr| arr.readwrite());
        let mut vm_rw = vm.map(|arr| arr.readwrite());
        let mut va_rw = va.map(|arr| arr.readwrite());
        let mut va_deg_rw = va_deg.map(|arr| arr.readwrite());
        let mut scalc_rw = scalc.map(|arr| arr.readwrite());
        let mut p_calc_rw = p_calc.map(|arr| arr.readwrite());
        let mut q_calc_rw = q_calc.map(|arr| arr.readwrite());

        let v_slice = match v_rw.as_mut() {
            Some(rw) => Some(rw.as_slice_mut().map_err(to_py_err)?),
            None => None,
        };
        let vm_slice = match vm_rw.as_mut() {
            Some(rw) => Some(rw.as_slice_mut().map_err(to_py_err)?),
            None => None,
        };
        let va_slice = match va_rw.as_mut() {
            Some(rw) => Some(rw.as_slice_mut().map_err(to_py_err)?),
            None => None,
        };
        let va_deg_slice = match va_deg_rw.as_mut() {
            Some(rw) => Some(rw.as_slice_mut().map_err(to_py_err)?),
            None => None,
        };
        let scalc_slice = match scalc_rw.as_mut() {
            Some(rw) => Some(rw.as_slice_mut().map_err(to_py_err)?),
            None => None,
        };
        let p_calc_slice = match p_calc_rw.as_mut() {
            Some(rw) => Some(rw.as_slice_mut().map_err(to_py_err)?),
            None => None,
        };
        let q_calc_slice = match q_calc_rw.as_mut() {
            Some(rw) => Some(rw.as_slice_mut().map_err(to_py_err)?),
            None => None,
        };

        self.inner
            .extract_results(
                v_slice,
                vm_slice,
                va_slice,
                va_deg_slice,
                scalc_slice,
                p_calc_slice,
                q_calc_slice,
            )
            .map_err(to_py_err)?;

        Ok(())
    }
}
