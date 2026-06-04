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
        
        // 1. Unpack Python CSR inputs
        let indptr: Vec<usize> = y_indptr.readonly().as_slice()?.iter().map(|&x| x as usize).collect();
        let indices: Vec<usize> = y_indices.readonly().as_slice()?.iter().map(|&x| x as usize).collect();
        let data = y_data.readonly().as_slice()?.to_vec();

        // 2. Direct O(NNZ) Sort-Free CSR to CSC Permutation
        // 
        // =====================================================================================
        // MATHEMATICAL ALGORITHM: The "Sort-Free Scatter" Trick
        // =====================================================================================
        // Goal: Compute Y_new = P * Y_old * P^T, where Y_old is CSR, and output Y_new as CSC.
        // P maps old indices to new indices: i_new = p_inv[i_old], or equivalently i_old = p_vec[i_new].
        // 
        // Naive approaches are slow:
        // 1. Generic matrix multiplication (P * Y * P^T) involves heavy structure lookups.
        // 2. Direct scattering (i_new = p_inv[i_old], j_new = p_inv[j_old]) usually produces 
        //    unsorted row indices in the resulting CSC columns, requiring an expensive O(NNZ log(NNZ)) sort.
        // 
        // The Optimal O(NNZ) Solution:
        // In a valid CSC matrix, elements within each column MUST be strictly sorted by their row index.
        // To achieve this *without sorting*, we make the outer loop iterate over `new_row` in strictly
        // ascending order (0, 1, 2, ..., N-1). 
        // 
        // For each `new_row`, we map back to the `old_row` = p_vec[new_row].
        // Because Y_old is in CSR format, we can instantly retrieve all non-zero elements of `old_row` in O(1).
        // Each element has an `old_col`, which maps to `new_col` = p_inv[old_col].
        // We then scatter the value into the `new_col` bucket.
        // 
        // Why this guarantees sorted columns:
        // Because we process `new_row` sequentially, any element we drop into a `new_col` bucket
        // is guaranteed to have a `new_row` index strictly greater than the element we dropped into 
        // that same bucket during an earlier iteration. 
        // Thus, the CSC structure is perfectly formed and sorted in a single O(NNZ) pass!
        // =====================================================================================
        
        let nnz = data.len();
        
        // Step 2a: Count non-zeros per new column to pre-allocate CSC indptr
        let mut nnz_per_new_col = vec![0; n];
        for &old_col in indices.iter() {
            let new_col = p_inv[old_col];
            nnz_per_new_col[new_col] += 1;
        }

        // Step 2b: Build CSC col_offsets (indptr)
        let mut csc_indptr = vec![0; n + 1];
        for i in 0..n {
            csc_indptr[i + 1] = csc_indptr[i] + nnz_per_new_col[i];
        }

        // Step 2c: Scatter data (Magically Sorted by design)
        let mut current_col_head = csc_indptr.clone();
        let mut csc_indices = vec![0; nnz];
        let mut csc_data = vec![num_complex::Complex64::new(0.0, 0.0); nnz];

        // The outer loop guarantees ascending `new_row` insertion!
        for new_row in 0..n {
            let old_row = p_vec[new_row];
            let start = indptr[old_row];
            let end = indptr[old_row + 1];
            
            for idx in start..end {
                let old_col = indices[idx];
                let val = data[idx];
                let new_col = p_inv[old_col];
                
                let insert_idx = current_col_head[new_col];
                csc_indices[insert_idx] = new_row;
                csc_data[insert_idx] = val;
                current_col_head[new_col] += 1;
            }
        }

        let y_perm_csc = CscMatrix::try_from_csc_data(n, n, csc_indptr, csc_indices, csc_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to build CSC: {}", e)))?;
        
        let s_raw = DVector::from_vec(s_bus.readonly().as_slice()?.to_vec());
        let v_raw = DVector::from_vec(v_init.readonly().as_slice()?.to_vec());

        // 3. Permute Vectors
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
            Err((_err, v)) => (false, 0, v),
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
