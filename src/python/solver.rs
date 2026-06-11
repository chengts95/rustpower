#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::{IntoPyArray, PyArrayMethods};
#[cfg(feature = "python")]
use crate::basic::ecs::powerflow::systems::{PowerFlowMat, PowerFlowResult, PowerFlowConfig};
#[cfg(feature = "python")]
use crate::basic::ecs::elements::PFCommonData;
#[cfg(feature = "python")]
use crate::basic::ecs::network::PowerFlowSolver;
#[cfg(feature = "python")]
use bevy_app::App;
#[cfg(feature = "python")]
use nalgebra::DVector;
#[cfg(feature = "python")]
use nalgebra_sparse::{CscMatrix, CooMatrix, CsrMatrix};

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

    /// Optimized context setup using the Double-Transpose trick.
    /// Maps Y_csc -> Y_t_csr, permutes in CSR, then swaps back to CSC for KLU.
    ///
    /// y_indptr, y_indices, y_data: CSC representation of the Y-bus matrix.
    /// s_bus: Complex power injections.
    /// v_init: Initial voltage guess.
    /// p_vec, p_inv: Permutation vectors.
    /// npv, npq: Number of PV and PQ buses.
    #[pyo3(signature = (y_indptr, y_indices, y_data, s_bus, v_init, p_vec, p_inv, npv, npq))]
    fn setup_context(
        &mut self,
        y_indptr: Bound<'_, numpy::PyArray1<i32>>,
        y_indices: Bound<'_, numpy::PyArray1<i32>>,
        y_data: Bound<'_, numpy::PyArray1<num_complex::Complex64>>,
        s_bus: Bound<'_, numpy::PyArray1<num_complex::Complex64>>,
        v_init: Bound<'_, numpy::PyArray1<num_complex::Complex64>>,
        p_vec: Vec<usize>,
        p_inv: Vec<usize>,
        npv: usize,
        npq: usize,
    ) -> PyResult<()> {
        let n = v_init.len()?;
        
        let indptr: Vec<usize> = y_indptr.readonly().as_slice()?.iter().map(|&x| x as usize).collect();
        let indices: Vec<usize> = y_indices.readonly().as_slice()?.iter().map(|&x| x as usize).collect();
        let data = y_data.readonly().as_slice()?.to_vec();

        // Use the ultra-fast O(NNZ) sort-free permutation utility
        let y_perm_csc = crate::basic::sparse::utils::permute_csr_to_csc_sort_free(
            n,
            &indptr,
            &indices,
            &data,
            &p_vec,
            &p_inv,
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
            reorder: CsrMatrix::from(&CscMatrix::from(&CooMatrix::new(n, n))),
            to_perm: p_vec.clone(),
            from_perm: p_inv.clone(),
        });

        self.p_vec = p_vec;
        self.p_inv = p_inv;
        Ok(())
    }

    /// Run the solver. Returns True if converged.
    fn solve(&mut self) -> PyResult<bool> {
        let world = self.app.world_mut();
        let (max_it, tol) = {
             let cfg = world.get_resource::<PowerFlowConfig>().cloned().unwrap_or_default();
             (cfg.max_it, cfg.tol)
        };

        let mut mat = world.remove_resource::<PowerFlowMat>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Context not initialized"))?;
        let mut solver_res = world.remove_resource::<PowerFlowSolver>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Solver resource missing"))?;

        let result = crate::basic::newton_pf(
            &mat.y_bus,
            &mat.s_bus,
            &mut mat.v_bus_init,
            mat.npv,
            mat.npq,
            tol,
            max_it,
            &mut solver_res.solver,
        );
        
        let (converged, its, v_final) = match result {
            Ok((v, i)) => (true, i, v),
            Err((_err, v, i)) => (false, i, v),
        };

        world.insert_resource(mat);
        world.insert_resource(solver_res);
        world.insert_resource(PowerFlowResult {
            v: v_final,
            iterations: its,
            converged,
        });

        Ok(converged)
    }

    /// Get the final complex bus voltages in original order.
    fn get_voltage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let world = self.app.world();
        let res = world.get_resource::<PowerFlowResult>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Solve has not been run"))?;
        
        let n = res.v.len();
        let mut v_final = vec![num_complex::Complex64::new(0.0, 0.0); n];
        for (i, &val) in res.v.as_slice().iter().enumerate() {
            // Restore original order using p_vec mapping
            // Since v_perm[i] = v_orig[p_vec[i]]
            v_final[self.p_vec[i]] = val;
        }
        Ok(v_final.into_pyarray(py))
    }
}
