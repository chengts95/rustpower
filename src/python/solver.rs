#[cfg(feature = "python")]
use crate::basic::ecs::elements::PFCommonData;
#[cfg(feature = "python")]
use crate::basic::ecs::network::PowerFlowSolver;
#[cfg(feature = "python")]
use crate::basic::ecs::powerflow::systems::{PowerFlowConfig, PowerFlowMat, PowerFlowResult};
#[cfg(feature = "python")]
use bevy_app::App;
#[cfg(feature = "python")]
use nalgebra::DVector;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArrayMethods};
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Low-level Newton-Raphson power flow solver.
///
/// This class provides direct access to the underlying solver logic, bypassing
/// the PowerGrid high-level abstraction. It expects pre-built Y-bus matrices
/// and handles permutations manually.
#[cfg(feature = "python")]
#[pyclass(unsendable)]
pub struct NewtonSolver {
    app: App,
    p_vec: Vec<usize>,
    p_inv: Vec<usize>,
}

#[cfg(feature = "python")]
#[pymethods]
impl NewtonSolver {
    /// Create a new NewtonSolver instance with default config.
    #[new]
    fn new() -> Self {
        let mut app = App::new();
        app.insert_resource(PowerFlowSolver::default());
        app.insert_resource(PFCommonData {
            sbase: 100.0,
            f_hz: 50.0,
            wbase: 2.0 * std::f64::consts::PI * 50.0,
        });
        app.insert_resource(PowerFlowConfig {
            max_it: Some(10),
            tol: Some(1e-8),
        });

        Self {
            app,
            p_vec: Vec::new(),
            p_inv: Vec::new(),
        }
    }

    /// Optimized context setup using the transpose trick.
    /// Converts the provided Y-bus matrix from CSR to CSC format, applies the given permutations,
    ///
    /// y_indptr, y_indices, y_data: CSR representation of the Y-bus matrix.
    /// s_bus: Complex power injections.
    /// v_init: Initial voltage guess.
    /// p_vec, p_inv: Permutation vectors.
    /// npv, npq: Number of PV and PQ buses.
    #[pyo3(signature = (y_indptr, y_indices, y_data, s_bus, v_init, p_vec_in, p_inv_in, npv, npq))]
    fn setup_context(
        &mut self,
        y_indptr: Bound<'_, numpy::PyArray1<i32>>,
        y_indices: Bound<'_, numpy::PyArray1<i32>>,
        y_data: Bound<'_, numpy::PyArray1<num_complex::Complex64>>,
        s_bus: Bound<'_, numpy::PyArray1<num_complex::Complex64>>,
        v_init: Bound<'_, numpy::PyArray1<num_complex::Complex64>>,
        p_vec_in: Bound<'_, numpy::PyArray1<i64>>,
        p_inv_in: Bound<'_, numpy::PyArray1<i64>>,
        npv: usize,
        npq: usize,
    ) -> PyResult<()> {
        let n = v_init.len()?;

        let p_vec: Vec<usize> = p_vec_in
            .readonly()
            .as_slice()?
            .iter()
            .map(|&x| x as usize)
            .collect();
        let p_inv: Vec<usize> = p_inv_in
            .readonly()
            .as_slice()?
            .iter()
            .map(|&x| x as usize)
            .collect();

        let indptr: Vec<usize> = y_indptr
            .readonly()
            .as_slice()?
            .iter()
            .map(|&x| x as usize)
            .collect();
        let indices: Vec<usize> = y_indices
            .readonly()
            .as_slice()?
            .iter()
            .map(|&x| x as usize)
            .collect();
        let y_data_ro = y_data.readonly();
        let data = y_data_ro.as_slice()?;

        // Use the ultra-fast O(NNZ) sort-free permutation utility
        let y_perm_csc = crate::basic::sparse::utils::permute_csr_to_csc_sort_free(
            n, &indptr, &indices, data, &p_vec, &p_inv,
        );

        let s_raw = DVector::from_vec(s_bus.readonly().as_slice()?.to_vec());
        let v_raw = DVector::from_vec(v_init.readonly().as_slice()?.to_vec());

        // Permute Vectors
        let mut s_perm = DVector::from_element(n, num_complex::Complex64::new(0.0, 0.0));
        let mut v_perm = DVector::from_element(n, num_complex::Complex64::new(0.0, 0.0));
        for (i, &old_idx) in p_vec.iter().enumerate() {
            s_perm[i] = s_raw[old_idx];
            v_perm[i] = v_raw[old_idx];
        }

        self.app.insert_resource(PowerFlowMat {
            y_bus: y_perm_csc,
            s_bus: s_perm,
            v_bus_init: v_perm,
            npv,
            npq,
            to_perm: p_vec.clone(),
            from_perm: p_inv.clone(),
        });

        self.p_vec = p_vec;
        self.p_inv = p_inv;
        Ok(())
    }

    /// Enable or disable Jacobian caching for this solver.
    /// When enabled, the solver reuses the symbolic factorization and matrix patterns
    /// across multiple solves, significantly speeding up consecutive computations on
    /// the same network structure.
    #[pyo3(signature = (enable=true))]
    fn enable_cache(&mut self, enable: bool) -> PyResult<()> {
        let world = self.app.world_mut();
        if enable {
            if !world.contains_resource::<crate::basic::newtonpf::NewtonCache>() {
                world.insert_resource(crate::basic::newtonpf::NewtonCache::default());
            }
        } else {
            world.remove_resource::<crate::basic::newtonpf::NewtonCache>();
        }
        Ok(())
    }

    /// Run the solver. Returns True if converged.
    #[pyo3(signature = (max_iter=10, tol=1e-6))]
    fn solve(&mut self, max_iter: usize, tol: f64) -> PyResult<bool> {
        use bevy_ecs::prelude::*;
        use bevy_ecs::system::RunSystemOnce;

        // Update the resource
        let mut cfg = self
            .app
            .world_mut()
            .get_resource_or_insert_with(|| PowerFlowConfig::default());
        cfg.max_it = Some(max_iter);
        cfg.tol = Some(tol);

        let converged = self
            .app
            .world_mut()
            .run_system_once(
                |mat: ResMut<PowerFlowMat>,
                 mut solver_res: ResMut<PowerFlowSolver>,
                 cfg: Res<PowerFlowConfig>,
                 mut cache: Option<ResMut<crate::basic::newtonpf::NewtonCache>>,
                 mut cmd: Commands| {
                    let mat_ref = mat.into_inner();
                    let result = crate::basic::newton_pf(
                        &mat_ref.y_bus,
                        &mat_ref.s_bus,
                        &mut mat_ref.v_bus_init,
                        mat_ref.npv,
                        mat_ref.npq,
                        cfg.tol,
                        cfg.max_it,
                        &mut solver_res.solver,
                        cache.as_deref_mut(),
                    );

                    let (converged, its, v_final) = match result {
                        Ok((v, i)) => (true, i, v),
                        Err((_err, v, i)) => (false, i, v),
                    };

                    cmd.insert_resource(PowerFlowResult {
                        v: v_final,
                        iterations: its,
                        converged,
                    });

                    converged
                },
            )
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Solver system failed or resources missing",
                )
            })?;

        Ok(converged)
    }

    /// Get the final complex bus voltages in original order.
    fn get_voltage<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let world = self.app.world();
        let res = world.get_resource::<PowerFlowResult>().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Solve has not been run")
        })?;

        let n = res.v.len();
        let mut v_final = vec![num_complex::Complex64::new(0.0, 0.0); n];
        for (i, &val) in res.v.as_slice().iter().enumerate() {
            // Restore original order using p_vec mapping
            // Since v_perm[i] = v_orig[p_vec[i]]
            v_final[self.p_vec[i]] = val;
        }

        Ok(v_final.into_pyarray(py))
    }

    /// Get the number of iterations taken by the solver.
    fn get_iterations(&self) -> PyResult<usize> {
        let world = self.app.world();
        let res = world.get_resource::<PowerFlowResult>().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Solve has not been run")
        })?;
        Ok(res.iterations)
    }

    /// Get the calculated bus power injection vector (S_calc = V * (Ybus * V)^*) in original bus order.
    /// If NewtonCache is enabled and populated, this extracts S_calc directly without matrix multiplication.
    /// If cache is empty or not enabled, it performs one matrix multiplication and vector conjugate product.
    fn get_scalc<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let world = self.app.world();
        let res = world.get_resource::<PowerFlowResult>().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Solve has not been run")
        })?;
        let mat = world.get_resource::<PowerFlowMat>().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("PowerFlowMat not initialized")
        })?;

        let n = res.v.len();
        let mut s_orig = vec![num_complex::Complex64::new(0.0, 0.0); n];

        if let Some(cache) = world.get_resource::<crate::basic::newtonpf::NewtonCache>() {
            if cache.s_calc.len() == n {
                for (i, &val) in cache.s_calc.as_slice().iter().enumerate() {
                    s_orig[self.p_vec[i]] = val;
                }
                return Ok(s_orig.into_pyarray(py));
            }
        }

        // Fallback: If cache is empty, do one csc_matvec_and_scalc
        let mut ibus = vec![num_complex::Complex64::new(0.0, 0.0); n];
        let mut scalc_perm = vec![num_complex::Complex64::new(0.0, 0.0); n];
        crate::basic::newtonpf::csc_matvec_and_scalc(
            mat.y_bus.col_offsets(),
            mat.y_bus.row_indices(),
            mat.y_bus.values(),
            res.v.as_slice(),
            &mut ibus,
            &mut scalc_perm,
        );

        for (i, &val) in scalc_perm.iter().enumerate() {
            s_orig[self.p_vec[i]] = val;
        }

        Ok(s_orig.into_pyarray(py))
    }
}
