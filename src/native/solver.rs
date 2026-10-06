//! High-performance native Newton-Raphson power flow solver.
//!
//! Provides a pure, stateful numerical solver object [`NewtonSolver`] without any ECS World overhead.
//! Reuses the core [`PowerFlowMat`] and [`PowerFlowResult`] representations for clean data modeling.

use nalgebra::DVector;
use num_complex::Complex64;

use crate::basic::ecs::powerflow::systems::{PowerFlowMat, PowerFlowResult};
use crate::basic::newton_pf;
use crate::basic::newtonpf::{csc_matvec_and_scalc, NewtonCache};
use crate::basic::solver::{DefaultSolver, Solve};
use crate::basic::sparse::utils::permute_csr_to_csc_sort_free;

/// Native stateful Newton-Raphson power flow solver.
///
/// Holds the numerical matrix context [`PowerFlowMat`], linear solver factorization,
/// Newton cache, and calculation result [`PowerFlowResult`] directly in memory.
pub struct NewtonSolver {
    /// Power flow matrix context (Ybus CSC, Sbus, Vbus_init, npv, npq, permutations).
    pub mat: Option<PowerFlowMat>,
    /// Sparse linear solver instance (e.g. KLU or Faer LU).
    pub linear_solver: DefaultSolver,
    /// Newton solver cache holding Jacobian sparsity pattern, values, and buffers.
    pub cache: NewtonCache,
    /// Power flow calculation result (final voltage, iterations, convergence status).
    pub result: Option<PowerFlowResult>,
    /// Maximum power mismatch norm ||F||_inf after the last solve.
    pub last_residual: f64,
}

impl Default for NewtonSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 1. 状态管理与初始化配置 (State Management & Context Initialization)
// ============================================================================
impl NewtonSolver {
    /// Create a new `NewtonSolver` instance.
    pub fn new() -> Self {
        Self {
            mat: None,
            linear_solver: DefaultSolver::default(),
            cache: NewtonCache::default(),
            result: None,
            last_residual: 0.0,
        }
    }

    /// Immutable reference to the underlying [`PowerFlowMat`].
    #[inline(always)]
    pub fn mat(&self) -> Result<&PowerFlowMat, &'static str> {
        self.mat
            .as_ref()
            .ok_or("Solver context not initialized; call setup_context or setup_from_nodes first")
    }

    /// Mutable reference to the underlying [`PowerFlowMat`].
    #[inline(always)]
    pub fn mat_mut(&mut self) -> Result<&mut PowerFlowMat, &'static str> {
        self.mat
            .as_mut()
            .ok_or("Solver context not initialized; call setup_context or setup_from_nodes first")
    }

    /// Immutable reference to the underlying [`PowerFlowResult`].
    #[inline(always)]
    pub fn result(&self) -> Result<&PowerFlowResult, &'static str> {
        self.result
            .as_ref()
            .ok_or("Solve has not been run or did not converge; call solve first")
    }

    /// Internal setup worker.
    fn setup_internal(
        &mut self,
        n: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[Complex64],
        s_raw: &[Complex64],
        v_raw: &[Complex64],
        to_perm: Vec<usize>,
        from_perm: Vec<usize>,
        npv: usize,
        npq: usize,
    ) {
        // Use the ultra-fast O(NNZ) sort-free permutation utility
        // permute_csr_to_csc_sort_free expects p_vec (from_perm: new -> old) and p_inv (to_perm: old -> new)
        let y_perm_csc = permute_csr_to_csc_sort_free(
            n, indptr, indices, data, &from_perm, &to_perm,
        );

        let mut s_perm = DVector::from_element(n, Complex64::new(0.0, 0.0));
        let mut v_perm = DVector::from_element(n, Complex64::new(0.0, 0.0));
        for (new_idx, &old_idx) in from_perm.iter().enumerate() {
            s_perm[new_idx] = s_raw[old_idx];
            v_perm[new_idx] = v_raw[old_idx];
        }

        // Invalidate cached Jacobian pattern and reset linear solver factorization for new network
        self.cache.j_pattern = None;
        self.linear_solver.reset();
        self.result = None;
        self.last_residual = 0.0;

        self.mat = Some(PowerFlowMat {
            y_bus: y_perm_csc,
            s_bus: s_perm,
            v_bus_init: v_perm,
            npv,
            npq,
            to_perm,
            from_perm,
        });
    }

    /// Setup solver context by specifying node type partitions directly (`pq`, `pv`, `ref_bus`).
    ///
    /// Automatically constructs `from_perm = [pq..., pv..., ref_bus...]`, calculates `to_perm`,
    /// and derives `npv` and `npq`. Automatically invalidates stale caches.
    pub fn setup_from_nodes(
        &mut self,
        n: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[Complex64],
        s_bus: &[Complex64],
        v_init: &[Complex64],
        pq: &[usize],
        pv: &[usize],
        ref_bus: &[usize],
    ) -> Result<(), String> {
        let npq = pq.len();
        let npv = pv.len();
        let n_total = npq + npv + ref_bus.len();

        if n_total != n || v_init.len() != n || s_bus.len() != n {
            return Err(format!(
                "Dimension mismatch: n={n}, pq+pv+ref={n_total}, v_init={}, s_bus={}",
                v_init.len(),
                s_bus.len()
            ));
        }

        let mut from_perm = Vec::with_capacity(n_total);
        from_perm.extend_from_slice(pq);
        from_perm.extend_from_slice(pv);
        from_perm.extend_from_slice(ref_bus);

        let mut to_perm = vec![0usize; n_total];
        for (new_idx, &old_idx) in from_perm.iter().enumerate() {
            if old_idx >= n_total {
                return Err(format!("Bus index {old_idx} out of bounds for grid size {n_total}"));
            }
            to_perm[old_idx] = new_idx;
        }

        self.setup_internal(
            n,
            indptr,
            indices,
            data,
            s_bus,
            v_init,
            to_perm,
            from_perm,
            npv,
            npq,
        );

        Ok(())
    }

    /// Setup solver context using explicitly provided permutation vectors.
    pub fn setup_context(
        &mut self,
        n: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[Complex64],
        s_bus: &[Complex64],
        v_init: &[Complex64],
        to_perm: Vec<usize>,
        from_perm: Vec<usize>,
        npv: usize,
        npq: usize,
    ) -> Result<(), String> {
        if to_perm.len() != n || from_perm.len() != n || s_bus.len() != n || v_init.len() != n {
            return Err(format!(
                "Dimension mismatch: n={n}, to_perm={}, from_perm={}, s_bus={}, v_init={}",
                to_perm.len(),
                from_perm.len(),
                s_bus.len(),
                v_init.len()
            ));
        }
        self.setup_internal(
            n,
            indptr,
            indices,
            data,
            s_bus,
            v_init,
            to_perm,
            from_perm,
            npv,
            npq,
        );
        Ok(())
    }

    /// Update the initial voltage vector (in original bus order).
    pub fn set_v_init(&mut self, v_init: &[Complex64]) -> Result<(), &'static str> {
        let mat = self.mat_mut()?;
        if v_init.len() != mat.to_perm.len() {
            return Err("Length mismatch in v_init");
        }
        for (old_idx, &val) in v_init.iter().enumerate() {
            let perm_idx = mat.to_perm[old_idx];
            mat.v_bus_init[perm_idx] = val;
        }
        Ok(())
    }

    /// Update initial voltage for a single bus (in original bus index).
    pub fn set_v_init_at(&mut self, bus: usize, v: Complex64) -> Result<(), &'static str> {
        let mat = self.mat_mut()?;
        if bus >= mat.to_perm.len() {
            return Err("Bus index out of bounds");
        }
        let perm_idx = mat.to_perm[bus];
        mat.v_bus_init[perm_idx] = v;
        Ok(())
    }

    /// Batch update initial voltage for specified bus indices.
    pub fn update_v_init_batch(&mut self, buses: &[usize], values: &[Complex64]) -> Result<(), &'static str> {
        if buses.len() != values.len() {
            return Err("Length mismatch between buses and values");
        }
        let mat = self.mat_mut()?;
        let n = mat.to_perm.len();
        for (&bus, &v) in buses.iter().zip(values.iter()) {
            if bus >= n {
                return Err("Bus index out of bounds");
            }
            let perm_idx = mat.to_perm[bus];
            mat.v_bus_init[perm_idx] = v;
        }
        Ok(())
    }

    /// Get initial voltage vector in original bus order.
    pub fn v_init(&self) -> Result<Vec<Complex64>, &'static str> {
        let mat = self.mat()?;
        let n = mat.from_perm.len();
        let mut v_orig = vec![Complex64::new(0.0, 0.0); n];
        for (perm_idx, &old_idx) in mat.from_perm.iter().enumerate() {
            v_orig[old_idx] = mat.v_bus_init[perm_idx];
        }
        Ok(v_orig)
    }

    /// Update the entire bus power injection vector Sbus (in original bus order).
    pub fn set_s_bus(&mut self, s_bus: &[Complex64]) -> Result<(), &'static str> {
        let mat = self.mat_mut()?;
        if s_bus.len() != mat.to_perm.len() {
            return Err("Length mismatch in s_bus");
        }
        for (old_idx, &val) in s_bus.iter().enumerate() {
            let perm_idx = mat.to_perm[old_idx];
            mat.s_bus[perm_idx] = val;
        }
        Ok(())
    }

    /// Update Sbus injection for a single bus (in original bus index).
    pub fn set_s_bus_at(&mut self, bus: usize, s: Complex64) -> Result<(), &'static str> {
        let mat = self.mat_mut()?;
        if bus >= mat.to_perm.len() {
            return Err("Bus index out of bounds");
        }
        let perm_idx = mat.to_perm[bus];
        mat.s_bus[perm_idx] = s;
        Ok(())
    }

    /// Batch update Sbus injections for specified bus indices.
    pub fn update_s_bus_batch(&mut self, buses: &[usize], values: &[Complex64]) -> Result<(), &'static str> {
        if buses.len() != values.len() {
            return Err("Length mismatch between buses and values");
        }
        let mat = self.mat_mut()?;
        let n = mat.to_perm.len();
        for (&bus, &s) in buses.iter().zip(values.iter()) {
            if bus >= n {
                return Err("Bus index out of bounds");
            }
            let perm_idx = mat.to_perm[bus];
            mat.s_bus[perm_idx] = s;
        }
        Ok(())
    }

    /// Get bus power injection vector Sbus in original bus order.
    pub fn s_bus(&self) -> Result<Vec<Complex64>, &'static str> {
        let mat = self.mat()?;
        let n = mat.from_perm.len();
        let mut s_orig = vec![Complex64::new(0.0, 0.0); n];
        for (perm_idx, &old_idx) in mat.from_perm.iter().enumerate() {
            s_orig[old_idx] = mat.s_bus[perm_idx];
        }
        Ok(s_orig)
    }

    /// Explicitly clear cached Jacobian pattern, linear solver factorizations, and Newton buffers.
    pub fn clear_cache(&mut self) {
        self.cache.j_pattern = None;
        self.linear_solver.reset();
        self.last_residual = 0.0;
    }

    /// Alias for `clear_cache`.
    #[inline(always)]
    pub fn reset_cache(&mut self) {
        self.clear_cache();
    }

    /// Permutation mapping: original bus index -> permuted solver index (`p_inv`).
    #[inline(always)]
    pub fn to_perm(&self) -> Result<&[usize], &'static str> {
        Ok(&self.mat()?.to_perm)
    }

    /// Inverse permutation mapping: permuted solver index -> original bus index (`p_vec`).
    #[inline(always)]
    pub fn from_perm(&self) -> Result<&[usize], &'static str> {
        Ok(&self.mat()?.from_perm)
    }

    /// Number of PV buses configured in the solver.
    #[inline(always)]
    pub fn npv(&self) -> Result<usize, &'static str> {
        Ok(self.mat()?.npv)
    }

    /// Number of PQ buses configured in the solver.
    #[inline(always)]
    pub fn npq(&self) -> Result<usize, &'static str> {
        Ok(self.mat()?.npq)
    }

    /// Total number of buses configured in the solver.
    #[inline(always)]
    pub fn n_buses(&self) -> Result<usize, &'static str> {
        Ok(self.mat()?.to_perm.len())
    }
}

// ============================================================================
// 2. 核心数值求解 (Core Numerical Solve)
// ============================================================================
impl NewtonSolver {
    /// Run the Newton-Raphson power flow calculation.
    ///
    /// Direct mathematical execution using [`PowerFlowMat`].
    /// Returns `(converged, residual)`.
    pub fn solve(&mut self, max_iter: usize, tol: f64) -> Result<(bool, f64), &'static str> {
        let mat = self
            .mat
            .as_mut()
            .ok_or("Solver context not initialized; call setup_context first")?;

        let res = newton_pf(
            &mat.y_bus,
            &mat.s_bus,
            &mut mat.v_bus_init,
            mat.npv,
            mat.npq,
            Some(tol),
            Some(max_iter),
            &mut self.linear_solver,
            Some(&mut self.cache),
        );

        let (converged, its, v_final) = match res {
            Ok((v, i)) => (true, i, v),
            Err((_err, v, i)) => (false, i, v),
        };

        // Compute residual norm directly from cache.F
        let residual = if self.cache.F.len() > 0 {
            self.cache.F.as_slice().iter().fold(0.0f64, |acc, &x| acc.max(x.abs()))
        } else {
            0.0
        };

        self.last_residual = residual;
        if let Some(ref mut r) = self.result {
            r.v = v_final;
            r.iterations = its;
            r.converged = converged;
        } else {
            self.result = Some(PowerFlowResult {
                v: v_final,
                iterations: its,
                converged,
            });
        }

        Ok((converged, residual))
    }

    /// Get the maximum power mismatch norm ||F||_inf after the last solve.
    #[inline(always)]
    pub fn residual(&self) -> f64 {
        self.last_residual
    }

    /// Get the iteration count of the last solve.
    #[inline(always)]
    pub fn iterations(&self) -> Result<usize, &'static str> {
        Ok(self.result()?.iterations)
    }

    /// Get whether the last solve converged.
    #[inline(always)]
    pub fn converged(&self) -> Result<bool, &'static str> {
        Ok(self.result()?.converged)
    }
}

// ============================================================================
// 3. 后处理与电气数据读取 (Post-Processing & Data Extraction)
// ============================================================================
impl NewtonSolver {
    /// Get final complex bus voltages in original bus order.
    pub fn voltage(&self) -> Result<Vec<Complex64>, &'static str> {
        let mat = self.mat()?;
        let res = self.result()?;
        let n = res.v.len();
        let mut v_orig = vec![Complex64::new(0.0, 0.0); n];
        for (perm_idx, &old_idx) in mat.from_perm.iter().enumerate() {
            v_orig[old_idx] = res.v[perm_idx];
        }
        Ok(v_orig)
    }

    /// Get voltage magnitudes (p.u.) in original bus order.
    pub fn vm(&self) -> Result<Vec<f64>, &'static str> {
        let mat = self.mat()?;
        let res = self.result()?;
        let n = res.v.len();
        let mut vm = vec![0.0f64; n];
        for (perm_idx, &old_idx) in mat.from_perm.iter().enumerate() {
            vm[old_idx] = res.v[perm_idx].norm();
        }
        Ok(vm)
    }

    /// Get voltage angles in original bus order.
    pub fn va(&self, deg: bool) -> Result<Vec<f64>, &'static str> {
        let mat = self.mat()?;
        let res = self.result()?;
        let n = res.v.len();
        let mut va = vec![0.0f64; n];
        let factor = if deg { 180.0 / std::f64::consts::PI } else { 1.0 };
        for (perm_idx, &old_idx) in mat.from_perm.iter().enumerate() {
            va[old_idx] = res.v[perm_idx].im.atan2(res.v[perm_idx].re) * factor;
        }
        Ok(va)
    }

    /// Get calculated complex bus power injections S_calc in original bus order.
    pub fn scalc(&self) -> Result<Vec<Complex64>, &'static str> {
        let mat = self.mat()?;
        let res = self.result()?;
        let n = res.v.len();
        let mut s_orig = vec![Complex64::new(0.0, 0.0); n];
        if self.cache.s_calc.len() == n {
            for (perm_idx, &old_idx) in mat.from_perm.iter().enumerate() {
                s_orig[old_idx] = self.cache.s_calc[perm_idx];
            }
            return Ok(s_orig);
        }

        let mut ibus = vec![Complex64::new(0.0, 0.0); n];
        let mut scalc_perm = vec![Complex64::new(0.0, 0.0); n];
        csc_matvec_and_scalc(
            mat.y_bus.col_offsets(),
            mat.y_bus.row_indices(),
            mat.y_bus.values(),
            res.v.as_slice(),
            &mut ibus,
            &mut scalc_perm,
        );

        for (perm_idx, &old_idx) in mat.from_perm.iter().enumerate() {
            s_orig[old_idx] = scalc_perm[perm_idx];
        }
        Ok(s_orig)
    }

    /// Get calculated active power injections P_calc in original bus order.
    pub fn p_calc(&self) -> Result<Vec<f64>, &'static str> {
        let s = self.scalc()?;
        Ok(s.iter().map(|c| c.re).collect())
    }

    /// Get calculated reactive power injections Q_calc in original bus order.
    pub fn q_calc(&self) -> Result<Vec<f64>, &'static str> {
        let s = self.scalc()?;
        Ok(s.iter().map(|c| c.im).collect())
    }

    /// In-place result extraction directly into caller-provided slices.
    ///
    /// Zero heap allocations: directly scatters internal permuted vectors into output slices
    /// in original bus order. Any slice provided as `None` is skipped.
    ///
    /// # Arguments
    /// * `v` - Optional slice for complex bus voltages (length `n_buses`).
    /// * `vm` - Optional slice for voltage magnitudes (length `n_buses`).
    /// * `va` - Optional slice for voltage angles in radians (length `n_buses`).
    /// * `va_deg` - Optional slice for voltage angles in degrees (length `n_buses`).
    /// * `scalc` - Optional slice for complex bus power injections (length `n_buses`).
    /// * `p_calc` - Optional slice for active power injections (length `n_buses`).
    /// * `q_calc` - Optional slice for reactive power injections (length `n_buses`).
    pub fn extract_results(
        &self,
        mut v: Option<&mut [Complex64]>,
        mut vm: Option<&mut [f64]>,
        mut va: Option<&mut [f64]>,
        mut va_deg: Option<&mut [f64]>,
        mut scalc: Option<&mut [Complex64]>,
        mut p_calc: Option<&mut [f64]>,
        mut q_calc: Option<&mut [f64]>,
    ) -> Result<(), &'static str> {
        let mat = self.mat()?;
        let res = self.result()?;
        let n = res.v.len();

        if let Some(ref s) = v {
            if s.len() != n {
                return Err("v slice length mismatch with number of buses");
            }
        }
        if let Some(ref s) = vm {
            if s.len() != n {
                return Err("vm slice length mismatch with number of buses");
            }
        }
        if let Some(ref s) = va {
            if s.len() != n {
                return Err("va slice length mismatch with number of buses");
            }
        }
        if let Some(ref s) = va_deg {
            if s.len() != n {
                return Err("va_deg slice length mismatch with number of buses");
            }
        }
        if let Some(ref s) = scalc {
            if s.len() != n {
                return Err("scalc slice length mismatch with number of buses");
            }
        }
        if let Some(ref s) = p_calc {
            if s.len() != n {
                return Err("p_calc slice length mismatch with number of buses");
            }
        }
        if let Some(ref s) = q_calc {
            if s.len() != n {
                return Err("q_calc slice length mismatch with number of buses");
            }
        }

        // 1. Scatter voltage-related fields
        let has_v = v.is_some();
        let has_vm = vm.is_some();
        let has_va = va.is_some();
        let has_va_deg = va_deg.is_some();

        if has_v || has_vm || has_va || has_va_deg {
            let deg_factor = 180.0 / std::f64::consts::PI;
            for (perm_idx, &old_idx) in mat.from_perm.iter().enumerate() {
                let val = res.v[perm_idx];
                if let Some(ref mut s) = v {
                    s[old_idx] = val;
                }
                if let Some(ref mut s) = vm {
                    s[old_idx] = val.norm();
                }
                if let Some(ref mut s) = va {
                    s[old_idx] = val.im.atan2(val.re);
                }
                if let Some(ref mut s) = va_deg {
                    s[old_idx] = val.im.atan2(val.re) * deg_factor;
                }
            }
        }

        // 2. Scatter power injection-related fields
        let has_s = scalc.is_some();
        let has_p = p_calc.is_some();
        let has_q = q_calc.is_some();

        if has_s || has_p || has_q {
            if self.cache.s_calc.len() == n {
                for (perm_idx, &old_idx) in mat.from_perm.iter().enumerate() {
                    let val = self.cache.s_calc[perm_idx];
                    if let Some(ref mut s) = scalc {
                        s[old_idx] = val;
                    }
                    if let Some(ref mut s) = p_calc {
                        s[old_idx] = val.re;
                    }
                    if let Some(ref mut s) = q_calc {
                        s[old_idx] = val.im;
                    }
                }
                return Ok(());
            }

            // Fallback: If cache not populated, compute csc_matvec_and_scalc
            let mut ibus = vec![Complex64::new(0.0, 0.0); n];
            let mut scalc_perm = vec![Complex64::new(0.0, 0.0); n];
            csc_matvec_and_scalc(
                mat.y_bus.col_offsets(),
                mat.y_bus.row_indices(),
                mat.y_bus.values(),
                res.v.as_slice(),
                &mut ibus,
                &mut scalc_perm,
            );
            for (perm_idx, &old_idx) in mat.from_perm.iter().enumerate() {
                let val = scalc_perm[perm_idx];
                if let Some(ref mut s) = scalc {
                    s[old_idx] = val;
                }
                if let Some(ref mut s) = p_calc {
                    s[old_idx] = val.re;
                }
                if let Some(ref mut s) = q_calc {
                    s[old_idx] = val.im;
                }
            }
        }

        Ok(())
    }
}
