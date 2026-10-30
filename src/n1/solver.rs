//! Fixed-partition DCPF + Newton computation with parent-owned shared analysis.
//! No worker owns or clones a sparse pattern or KLU Symbolic object.
use super::numeric::fill_jacobian;
use crate::basic::{
    dcpf::DcpfModel,
    ecs::powerflow::systems::PowerFlowMat,
    jacobian_cache::JacobianCache,
    newtonpf::{csc_matvec_and_scalc, fill_f_from_scalc},
};
use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use rustpower_sol_klu::KLUSolver;
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

static NEXT_BASELINE: AtomicU64 = AtomicU64::new(1);

/// One immutable CSC input and its parent-owned KLU symbolic object.
struct SharedAnalysis {
    owner: KLUSolver,
    offsets: Vec<i64>,
    rows: Vec<i64>,
}
impl SharedAnalysis {
    fn new(offsets: &[usize], rows: &[usize]) -> Result<Self, String> {
        let offsets: Vec<i64> = offsets
            .iter()
            .map(|&x| i64::try_from(x))
            .collect::<Result<_, _>>()
            .map_err(|_| "CSC offsets exceed KLU integer range")?;
        let rows: Vec<i64> = rows
            .iter()
            .map(|&x| i64::try_from(x))
            .collect::<Result<_, _>>()
            .map_err(|_| "CSC rows exceed KLU integer range")?;
        let mut owner = KLUSolver::default();
        // klu.h specifies CSC inputs as unmodified. No mutable Rust references
        // to the shared buffers are created, including after publication.
        let status = unsafe {
            owner.solve_sym(
                offsets.as_ptr() as *mut i64,
                rows.as_ptr() as *mut i64,
                offsets.len() as i64 - 1,
            )
        };
        if status != 0 || owner.symbolic.is_null() {
            return Err(format!("KLU symbolic analysis failed: status {status}"));
        }
        Ok(Self {
            owner,
            offsets,
            rows,
        })
    }
    fn id(&self) -> usize {
        self.owner.symbolic as usize
    }
}

/// The main system owns this object until all worker calls have finished.
/// Read-only sharing is supported by KLU's Symbolic input contract. All mutable
/// KLU Common/Numeric state used for a solve belongs to its worker.
pub struct SharedBaseline {
    id: u64,
    pub(super) y_bus: CscMatrix<Complex64>,
    pub(super) s_bus: DVector<Complex64>,
    pub(super) v_init: DVector<Complex64>,
    ac_start: Option<Vec<Complex64>>,
    pub(super) npv: usize,
    pub(super) npq: usize,
    pub(super) to_perm: Vec<usize>,
    pub(super) from_perm: Vec<usize>,
    pub(super) j_pattern: JacobianCache,
    pub(super) dc_model: DcpfModel,
    ac_analysis: SharedAnalysis,
    dc_analysis: SharedAnalysis,
}
impl SharedBaseline {
    /// Prepare the immutable numerical baseline and both analyses once.
    /// p_shift_orig is the phase-shift injection vector in physical bus order,
    /// matching DcpfModel::from_ybus. Empty means all-zero phase injection.
    pub fn from_mat(mat: &PowerFlowMat, p_shift_orig: &[f64]) -> Result<Self, String> {
        let n = mat.y_bus.ncols();
        let active = mat
            .npq
            .checked_add(mat.npv)
            .ok_or("invalid bus partition")?;
        if n == 0
            || active == 0
            || active >= n
            || mat.y_bus.nrows() != n
            || mat.s_bus.len() != n
            || mat.v_bus_init.len() != n
            || mat.to_perm.len() != n
            || mat.from_perm.len() != n
        {
            return Err("fixed-partition baseline dimensions are invalid".into());
        }
        let mut seen = vec![false; n];
        for (k, &physical) in mat.from_perm.iter().enumerate() {
            if physical >= n || seen[physical] || mat.to_perm[physical] != k {
                return Err("baseline bus permutations are inconsistent".into());
            }
            seen[physical] = true;
        }
        // Structural diagonals are required by V4 assembly. This is local
        // matrix validation, with no graph reachability traversal.
        for k in 0..active {
            let rows = &mat.y_bus.row_indices()
                [mat.y_bus.col_offsets()[k]..mat.y_bus.col_offsets()[k + 1]];
            if rows.binary_search(&k).is_err() {
                return Err(format!(
                    "Ybus column {k} is missing its structural diagonal"
                ));
            }
        }
        if !p_shift_orig.is_empty() && p_shift_orig.len() != n {
            return Err("phase-shift injection length does not match the baseline".into());
        }
        let zero_shift;
        let shift = if p_shift_orig.is_empty() {
            zero_shift = vec![0.; n];
            &zero_shift[..]
        } else {
            p_shift_orig
        };
        let j_pattern = JacobianCache::build_from_permuted(
            mat.y_bus.col_offsets(),
            mat.y_bus.row_indices(),
            mat.npv,
            mat.npq,
        );
        let dc_model =
            DcpfModel::from_ybus(&mat.y_bus, &mat.v_bus_init, active, &mat.to_perm, shift);
        let ac_analysis = SharedAnalysis::new(&j_pattern.j_col_ptrs, &j_pattern.j_row_indices)?;
        let dc_analysis = SharedAnalysis::new(&dc_model.col_ptrs, &dc_model.row_indices)?;
        Ok(Self {
            id: NEXT_BASELINE.fetch_add(1, Ordering::Relaxed),
            y_bus: mat.y_bus.clone(),
            s_bus: mat.s_bus.clone(),
            v_init: mat.v_bus_init.clone(),
            ac_start: None,
            npv: mat.npv,
            npq: mat.npq,
            to_perm: mat.to_perm.clone(),
            from_perm: mat.from_perm.clone(),
            j_pattern,
            dc_model,
            ac_analysis,
            dc_analysis,
        })
    }
    /// Publish the main system's converged base AC voltage, in solver order.
    /// This changes the numerical initial guess only; patterns and symbolic
    /// identities remain unchanged. Call before sharing this baseline with workers.
    pub fn set_ac_start(&mut self, voltage: &[Complex64]) -> Result<(), String> {
        if voltage.len() != self.v_init.len()
            || voltage.iter().any(|z| {
                let magnitude = z.norm();
                !z.re.is_finite() || !z.im.is_finite() || !magnitude.is_finite() || magnitude == 0.0
            })
        {
            return Err(
                "base AC voltage must have the baseline length and finite nonzero entries".into(),
            );
        }
        let active = self.npq + self.npv;
        for k in self.npq..active {
            if (voltage[k].norm() - self.v_init[k].norm()).abs()
                > 1e-8 * self.v_init[k].norm().max(1.0)
            {
                return Err(format!(
                    "base AC voltage changes the PV magnitude at solver bus {k}"
                ));
            }
        }
        for k in active..voltage.len() {
            if (voltage[k] - self.v_init[k]).norm() > 1e-8 * self.v_init[k].norm().max(1.0) {
                return Err(format!(
                    "base AC voltage changes the Slack reference at solver bus {k}"
                ));
            }
        }
        match &mut self.ac_start {
            Some(existing) => existing.copy_from_slice(voltage),
            None => self.ac_start = Some(voltage.to_vec()),
        }
        Ok(())
    }
    /// Main-system AC initial voltage for independent N-1 Newton solves.
    pub fn ac_start(&self) -> Option<&[Complex64]> {
        self.ac_start.as_deref()
    }

    pub(super) fn generation(&self) -> u64 {
        self.id
    }

    /// Current known owned allocations, excluding the source ECS and allocator overhead.
    pub fn allocation_stats(&self) -> SharedAllocationStats {
        let j = &self.j_pattern;
        let pattern_bytes = (self.y_bus.col_offsets().len()
            + self.y_bus.row_indices().len()
            + self.to_perm.capacity()
            + self.from_perm.capacity()
            + j.j_col_ptrs.capacity()
            + j.j_row_indices.capacity()
            + j.pq_ends.capacity()
            + j.active_ends.capacity()
            + j.diag_ptrs.capacity()
            + self.dc_model.col_ptrs.capacity()
            + self.dc_model.row_indices.capacity())
            * std::mem::size_of::<usize>()
            + (self.ac_analysis.offsets.capacity()
                + self.ac_analysis.rows.capacity()
                + self.dc_analysis.offsets.capacity()
                + self.dc_analysis.rows.capacity())
                * std::mem::size_of::<i64>();
        let baseline_value_bytes = (self.y_bus.values().len()
            + self.s_bus.len()
            + self.v_init.len()
            + self.ac_start.as_ref().map_or(0, Vec::capacity))
            * std::mem::size_of::<Complex64>()
            + (self.dc_model.values.capacity() + self.dc_model.p_shift.len())
                * std::mem::size_of::<f64>();
        let symbolic_bytes = unsafe {
            (*self.ac_analysis.owner.common).memusage + (*self.dc_analysis.owner.common).memusage
        } as usize;
        SharedAllocationStats {
            pattern_bytes,
            baseline_value_bytes,
            symbolic_bytes,
        }
    }

    /// Read-only accessors keep prepared structure immutable to public callers.
    pub fn y_bus(&self) -> &CscMatrix<Complex64> {
        &self.y_bus
    }
    pub fn s_bus(&self) -> &[Complex64] {
        self.s_bus.as_slice()
    }
    pub fn v_init(&self) -> &[Complex64] {
        self.v_init.as_slice()
    }
    pub fn npv(&self) -> usize {
        self.npv
    }
    pub fn npq(&self) -> usize {
        self.npq
    }
    pub fn to_perm(&self) -> &[usize] {
        &self.to_perm
    }
    pub fn from_perm(&self) -> &[usize] {
        &self.from_perm
    }
    pub fn dc_model(&self) -> &DcpfModel {
        &self.dc_model
    }
    pub fn j_pattern(&self) -> &JacobianCache {
        &self.j_pattern
    }
    /// (DC, AC) object addresses, useful for proving analysis reuse.
    pub fn symbolic_ids(&self) -> (usize, usize) {
        (self.dc_analysis.id(), self.ac_analysis.id())
    }
    /// (Ybus columns, Ybus rows, DC columns, AC columns).
    pub fn pattern_ids(&self) -> (usize, usize, usize, usize) {
        (
            self.y_bus.col_offsets().as_ptr() as usize,
            self.y_bus.row_indices().as_ptr() as usize,
            self.dc_analysis.offsets.as_ptr() as usize,
            self.ac_analysis.offsets.as_ptr() as usize,
        )
    }
}

/// Per-case numerical work, including recovery after an unusable refactored solve.
#[derive(Default, Debug, Clone, Copy)]
pub struct LinearSolveStats {
    pub factor_calls: usize,
    pub refactor_calls: usize,
    pub refactor_solve_retries: usize,
}
impl LinearSolveStats {
    pub(super) fn add(self, other: Self) -> Self {
        Self {
            factor_calls: self.factor_calls + other.factor_calls,
            refactor_calls: self.refactor_calls + other.refactor_calls,
            refactor_solve_retries: self.refactor_solve_retries + other.refactor_solve_retries,
        }
    }
}

/// Worker-owned numbers. Shared Symbolic is attached only for the FFI calls.
struct NumericKlu {
    solver: KLUSolver,
    rhs_backup: Vec<f64>,
    stats: LinearSolveStats,
}
impl NumericKlu {
    fn new() -> Self {
        Self {
            solver: KLUSolver::default(),
            rhs_backup: Vec::new(),
            stats: LinearSolveStats::default(),
        }
    }
    unsafe fn factor(
        &mut self,
        p: *mut i64,
        i: *mut i64,
        values: &mut [f64],
    ) -> Result<(), String> {
        self.stats.factor_calls += 1;
        let status = unsafe { self.solver.factor(p, i, values.as_mut_ptr()) };
        if status != 0 || self.solver.numeric.is_null() {
            let common = unsafe { &*self.solver.common };
            return Err(format!(
                "KLU factor failed: status {status}, zero_pivot {}, singular_column {}",
                common.numerical_rank, common.singular_col
            ));
        }
        Ok(())
    }
    fn solve(
        &mut self,
        analysis: &SharedAnalysis,
        values: &mut [f64],
        rhs: &mut [f64],
    ) -> Result<(), String> {
        if values.len() != analysis.rows.len() || rhs.len() + 1 != analysis.offsets.len() {
            return Err("fixed-pattern numeric dimensions do not match analysis".into());
        }
        self.solver.symbolic = analysis.owner.symbolic;
        let result = (|| unsafe {
            if let Some((slot, value)) = values.iter().enumerate().find(|(_, x)| !x.is_finite()) {
                let col = analysis
                    .offsets
                    .partition_point(|&start| start <= slot as i64)
                    - 1;
                let row = analysis.rows[slot];
                return Err(format!(
                    "non-finite KLU matrix input at row {row}, col {col}: {value}"
                ));
            }
            if let Some((row, value)) = rhs.iter().enumerate().find(|(_, x)| !x.is_finite()) {
                return Err(format!("non-finite KLU RHS input at row {row}: {value}"));
            }
            let p = analysis.offsets.as_ptr() as *mut i64;
            let i = analysis.rows.as_ptr() as *mut i64;
            let mut used_refactor = false;
            if self.solver.numeric.is_null() {
                self.factor(p, i, values)?;
            } else {
                // KLU solve overwrites RHS; retain the original for a same-system retry.
                self.rhs_backup.resize(rhs.len(), 0.0);
                self.rhs_backup.copy_from_slice(rhs);
                self.stats.refactor_calls += 1;
                let status = self
                    .solver
                    .refactor(p, i, values.as_mut_ptr(), rhs.len() as i64);
                if status == 0 {
                    used_refactor = true;
                } else {
                    self.factor(p, i, values)
                        .map_err(|e| format!("refactor status {status}; {e}"))?;
                }
            }
            let mut status = self.solver.solve(rhs.as_mut_ptr(), rhs.len() as i64, 1);
            let mut finite = rhs.iter().all(|x| x.is_finite());
            if used_refactor && (status != 0 || !finite) {
                self.stats.refactor_solve_retries += 1;
                let reason = format!("refactored solve status {status}, finite={finite}");
                self.factor(p, i, values)
                    .map_err(|e| format!("{reason}; retry {e}"))?;
                rhs.copy_from_slice(&self.rhs_backup);
                status = self.solver.solve(rhs.as_mut_ptr(), rhs.len() as i64, 1);
                finite = rhs.iter().all(|x| x.is_finite());
                if status != 0 || !finite {
                    return Err(format!(
                        "{reason}; full-factor solve status {status}, finite={finite}"
                    ));
                }
            }
            if status != 0 {
                return Err(format!("KLU solve failed after factor: status {status}"));
            }
            if !finite {
                return Err("linear solve returned non-finite values after full factor".into());
            }
            Ok(())
        })();
        self.solver.symbolic = std::ptr::null_mut();
        if result.is_err() {
            self.solver.reset();
        }
        result
    }
}
impl Drop for NumericKlu {
    fn drop(&mut self) {
        self.solver.symbolic = std::ptr::null_mut();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStage {
    Converged,
    DcFailed,
    NewtonFailed,
}
/// Numerical timing scopes for one solve. Shared preparation and scenario
/// patching happen outside these clocks; copying the returned voltage is excluded.
#[derive(Default, Clone, Copy, Debug)]
pub struct SolveTimings {
    /// KLU DC factor/refactor/solve only; RHS preparation is excluded.
    pub dc: Duration,
    /// Initial-voltage construction/copy through the AC terminal decision.
    /// AC-only solves begin here and keep dc equal to zero.
    pub ac: Duration,
    /// Sum of each attempted NR step: J fill, linear solve, voltage and mismatch update.
    pub ac_iterations: Duration,
    /// Includes a step whose linear solve failed before a voltage update.
    pub attempted_iterations: usize,
}

#[derive(Debug)]
pub struct CaseSolution {
    pub converged: bool,
    pub stage: SolveStage,
    pub iterations: usize,
    pub residual: f64,
    /// Voltage in the shared solver's [PQ | PV | Slack] order.
    pub voltage: Vec<Complex64>,
    pub error: Option<String>,
    pub timings: SolveTimings,
    pub linear: LinearSolveStats,
}

/// Solver metadata for worker-local postprocessing; voltage stays in N1Worker.
#[derive(Debug)]
pub(super) struct CaseStatus {
    pub converged: bool,
    pub stage: SolveStage,
    pub iterations: usize,
    pub residual: f64,
    pub error: Option<String>,
    pub timings: SolveTimings,
    pub linear: LinearSolveStats,
}
impl CaseStatus {
    fn with_voltage(self, voltage: &[Complex64]) -> CaseSolution {
        CaseSolution {
            converged: self.converged,
            stage: self.stage,
            iterations: self.iterations,
            residual: self.residual,
            voltage: voltage.to_vec(),
            error: self.error,
            timings: self.timings,
            linear: self.linear,
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct KluAllocationStats {
    pub current_bytes: usize,
    pub peak_bytes: usize,
    pub l_nonzeros: usize,
    pub u_nonzeros: usize,
    pub factor_block_bytes: usize,
    pub permanent_workspace_bytes: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct WorkerAllocationStats {
    pub vector_bytes: usize,
    pub dc: KluAllocationStats,
    pub ac: KluAllocationStats,
}
#[derive(Clone, Copy, Debug)]
pub struct SharedAllocationStats {
    pub pattern_bytes: usize,
    pub baseline_value_bytes: usize,
    pub symbolic_bytes: usize,
}
impl NumericKlu {
    fn allocation_stats(&self) -> KluAllocationStats {
        unsafe {
            let common = &*self.solver.common;
            let mut stats = KluAllocationStats {
                current_bytes: common.memusage as usize,
                peak_bytes: common.mempeak as usize,
                l_nonzeros: 0,
                u_nonzeros: 0,
                factor_block_bytes: 0,
                permanent_workspace_bytes: 0,
            };
            if let Some(numeric) = self.solver.numeric.as_ref() {
                stats.l_nonzeros = numeric.lnz as usize;
                stats.u_nonzeros = numeric.unz as usize;
                stats.permanent_workspace_bytes = numeric.worksize as usize;
                if !numeric.LUsize.is_null() {
                    stats.factor_block_bytes =
                        std::slice::from_raw_parts(numeric.LUsize, numeric.nblocks as usize)
                            .iter()
                            .map(|&n| n as usize * std::mem::size_of::<f64>())
                            .sum();
                }
            }
            stats
        }
    }
}

/// Per-worker numbers and KLU factors, with shared structure supplied by borrow.
/// No Entity/World pointers and no borrowed lifetime are stored in this object.
pub struct N1Worker {
    baseline_id: u64,
    shared_symbolics: (usize, usize),
    shared_patterns: (usize, usize, usize, usize),
    pub y_values: Vec<Complex64>,
    pub dc_values: Vec<f64>,
    pub dc_shift: Vec<f64>,
    dc_rhs: Vec<f64>,
    j_values: Vec<f64>,
    v: Vec<Complex64>,
    v_norm: Vec<Complex64>,
    vm: Vec<f64>,
    va: Vec<f64>,
    ibus: Vec<Complex64>,
    s_calc: Vec<Complex64>,
    mismatch: Vec<f64>,
    dc_numeric: Option<NumericKlu>,
    ac_numeric: NumericKlu,
    timings: SolveTimings,
    ac_started: Option<Instant>,
}
impl N1Worker {
    /// Reserved numeric buffers and KLU-reported live/peak allocations for this worker.
    pub fn allocation_stats(&self) -> WorkerAllocationStats {
        let vector_bytes = (self.y_values.capacity()
            + self.v.capacity()
            + self.v_norm.capacity()
            + self.ibus.capacity()
            + self.s_calc.capacity())
            * std::mem::size_of::<Complex64>()
            + (self.dc_values.capacity()
                + self.dc_shift.capacity()
                + self.dc_rhs.capacity()
                + self.j_values.capacity()
                + self.vm.capacity()
                + self.va.capacity()
                + self.mismatch.capacity()
                + self.ac_numeric.rhs_backup.capacity()
                + self
                    .dc_numeric
                    .as_ref()
                    .map_or(0, |s| s.rhs_backup.capacity()))
                * std::mem::size_of::<f64>();
        WorkerAllocationStats {
            vector_bytes,
            dc: self
                .dc_numeric
                .as_ref()
                .map_or_else(KluAllocationStats::default, NumericKlu::allocation_stats),
            ac: self.ac_numeric.allocation_stats(),
        }
    }
    pub(super) fn baseline_generation(&self) -> u64 {
        self.baseline_id
    }
    /// Legacy foundation worker with DCPF followed by Newton.
    pub fn new(base: &SharedBaseline) -> Self {
        Self::new_mode(base, true)
    }
    /// N-1 worker: one private AC Numeric/Common and no DC numerical buffers.
    pub fn new_ac(base: &SharedBaseline) -> Self {
        Self::new_mode(base, false)
    }
    pub(super) fn uses_dc(&self) -> bool {
        self.dc_numeric.is_some()
    }
    fn new_mode(base: &SharedBaseline, with_dc: bool) -> Self {
        let n = base.v_init.len();
        let active = base.npv + base.npq;
        Self {
            baseline_id: base.id,
            shared_symbolics: base.symbolic_ids(),
            shared_patterns: base.pattern_ids(),
            y_values: base.y_bus.values().to_vec(),
            dc_values: if with_dc {
                base.dc_model.values.clone()
            } else {
                Vec::new()
            },
            dc_shift: if with_dc {
                base.dc_model.p_shift.as_slice().to_vec()
            } else {
                Vec::new()
            },
            dc_rhs: if with_dc {
                vec![0.; active]
            } else {
                Vec::new()
            },
            j_values: vec![0.; base.j_pattern.j_row_indices.len()],
            v: vec![Complex64::new(0., 0.); n],
            v_norm: vec![Complex64::new(0., 0.); n],
            vm: vec![0.; n],
            va: vec![0.; n],
            ibus: vec![Complex64::new(0., 0.); n],
            s_calc: vec![Complex64::new(0., 0.); n],
            mismatch: vec![0.; active + base.npq],
            dc_numeric: with_dc.then(NumericKlu::new),
            ac_numeric: NumericKlu::new(),
            timings: SolveTimings::default(),
            ac_started: None,
        }
    }
    pub fn symbolic_ids(&self) -> (usize, usize) {
        self.shared_symbolics
    }
    pub fn pattern_ids(&self) -> (usize, usize, usize, usize) {
        self.shared_patterns
    }
    /// Solve and return an owned voltage snapshot for callers retaining the result.
    pub fn solve_base(&mut self, base: &SharedBaseline, max_iter: usize, tol: f64) -> CaseSolution {
        self.solve_base_status(base, max_iter, tol)
            .with_voltage(&self.v)
    }
    /// Solve patched AC/DC values and return an owned voltage snapshot.
    pub fn solve_current(
        &mut self,
        base: &SharedBaseline,
        max_iter: usize,
        tol: f64,
    ) -> CaseSolution {
        self.solve_current_status(base, max_iter, tol)
            .with_voltage(&self.v)
    }
    /// Solve from shared base AC and return an owned voltage snapshot.
    pub fn solve_ac_current(
        &mut self,
        base: &SharedBaseline,
        max_iter: usize,
        tol: f64,
    ) -> CaseSolution {
        self.solve_ac_current_status(base, max_iter, tol)
            .with_voltage(&self.v)
    }
    /// Current solver-order voltage, valid until the next mutable worker operation.
    pub(super) fn voltage(&self) -> &[Complex64] {
        &self.v
    }
    /// Copy base numerical values into existing buffers, then solve the base.
    fn solve_base_status(
        &mut self,
        base: &SharedBaseline,
        max_iter: usize,
        tol: f64,
    ) -> CaseStatus {
        self.reset_timings();
        if base.id != self.baseline_id {
            return self.failed(
                SolveStage::NewtonFailed,
                0,
                f64::INFINITY,
                "worker belongs to a different baseline".into(),
            );
        }
        if !self.uses_dc() {
            return self.failed(
                SolveStage::DcFailed,
                0,
                f64::INFINITY,
                "AC-only worker uses solve_ac_current; DCPF buffers are absent".into(),
            );
        }
        self.y_values
            .resize(base.y_bus.values().len(), Complex64::new(0.0, 0.0));
        self.dc_values.resize(base.dc_model.values.len(), 0.0);
        self.dc_shift.resize(base.npv + base.npq, 0.0);
        self.y_values.copy_from_slice(base.y_bus.values());
        self.dc_values.copy_from_slice(&base.dc_model.values);
        self.dc_shift
            .copy_from_slice(base.dc_model.p_shift.as_slice());
        self.solve_current_status(base, max_iter, tol)
    }
    /// Solve the current private numerical values with unchanged shared patterns.
    /// Caller updates Ybus and DC numerical data consistently before invoking it.
    pub(super) fn solve_current_status(
        &mut self,
        base: &SharedBaseline,
        max_iter: usize,
        tol: f64,
    ) -> CaseStatus {
        self.reset_timings();
        if base.id != self.baseline_id {
            return self.failed(
                SolveStage::NewtonFailed,
                0,
                f64::INFINITY,
                "worker belongs to a different baseline".into(),
            );
        }
        if self.y_values.len() != base.y_bus.values().len() {
            return self.failed(
                SolveStage::NewtonFailed,
                0,
                f64::INFINITY,
                "Ybus numerical values do not match the shared pattern".into(),
            );
        }
        if !self.uses_dc() {
            return self.failed(
                SolveStage::DcFailed,
                0,
                f64::INFINITY,
                "AC-only worker uses solve_ac_current; DCPF buffers are absent".into(),
            );
        }
        if self.dc_values.len() != base.dc_model.values.len()
            || self.dc_shift.len() != base.npv + base.npq
        {
            return self.failed(
                SolveStage::DcFailed,
                0,
                f64::INFINITY,
                "DC numerical values or shift vector do not match the shared pattern".into(),
            );
        }
        self.v.copy_from_slice(base.v_init.as_slice());
        if !tol.is_finite() || tol <= 0. {
            return self.failed(
                SolveStage::NewtonFailed,
                0,
                f64::INFINITY,
                "tolerance must be positive and finite".into(),
            );
        }
        let active = base.npv + base.npq;
        for k in 0..active {
            self.dc_rhs[k] = base.s_bus[k].re - self.dc_shift[k];
        }
        let dc_started = Instant::now();
        let dc_result = self
            .dc_numeric
            .as_mut()
            .expect("DC mode was checked")
            .solve(&base.dc_analysis, &mut self.dc_values, &mut self.dc_rhs);
        self.timings.dc = dc_started.elapsed();
        if let Err(e) = dc_result {
            return self.failed(SolveStage::DcFailed, 0, f64::INFINITY, e);
        }
        self.ac_started = Some(Instant::now());
        for k in 0..active {
            self.v[k] = Complex64::from_polar(base.v_init[k].norm(), self.dc_rhs[k]);
        }
        self.solve_newton(base, max_iter, tol)
    }
    /// AC-only N-1 solve from the main system's base AC voltage. Current private
    /// Ybus values must represent this scenario; no DCPF arrays or solve are used.
    pub(super) fn solve_ac_current_status(
        &mut self,
        base: &SharedBaseline,
        max_iter: usize,
        tol: f64,
    ) -> CaseStatus {
        self.reset_timings();
        if base.id != self.baseline_id {
            return self.failed(
                SolveStage::NewtonFailed,
                0,
                f64::INFINITY,
                "worker belongs to a different baseline".into(),
            );
        }
        if self.y_values.len() != base.y_bus.values().len() {
            return self.failed(
                SolveStage::NewtonFailed,
                0,
                f64::INFINITY,
                "Ybus numerical values do not match the shared pattern".into(),
            );
        }
        if !tol.is_finite() || tol <= 0.0 {
            return self.failed(
                SolveStage::NewtonFailed,
                0,
                f64::INFINITY,
                "tolerance must be positive and finite".into(),
            );
        }
        let Some(start) = base.ac_start() else {
            return self.failed(
                SolveStage::NewtonFailed,
                0,
                f64::INFINITY,
                "prepare the shared base AC voltage before running N-1".into(),
            );
        };
        self.ac_started = Some(Instant::now());
        self.v.copy_from_slice(start);
        self.solve_newton(base, max_iter, tol)
    }
    fn solve_newton(&mut self, base: &SharedBaseline, max_iter: usize, tol: f64) -> CaseStatus {
        let active = base.npv + base.npq;
        for k in 0..self.v.len() {
            self.vm[k] = self.v[k].norm();
            self.va[k] = self.v[k].arg();
            self.v_norm[k] = Complex64::from_polar(1., self.va[k]);
        }
        let mut residual = self.evaluate(base);
        if !residual.is_finite() {
            return self.failed(
                SolveStage::NewtonFailed,
                0,
                residual,
                "non-finite initial AC mismatch".into(),
            );
        }
        if residual < tol {
            return self.success(0, residual);
        }
        let j = &base.j_pattern;
        for it in 0..max_iter {
            let iteration_started = Instant::now();
            self.timings.attempted_iterations += 1;
            fill_jacobian::<false>(
                base.y_bus.col_offsets(),
                base.y_bus.row_indices(),
                &self.y_values,
                &self.v,
                &self.v_norm,
                &self.s_calc,
                &j.j_col_ptrs,
                &j.pq_ends,
                &j.active_ends,
                &j.diag_ptrs,
                base.npv,
                base.npq,
                &mut self.j_values,
            );
            let linear =
                self.ac_numeric
                    .solve(&base.ac_analysis, &mut self.j_values, &mut self.mismatch);
            if let Err(e) = linear {
                self.timings.ac_iterations += iteration_started.elapsed();
                let (min_bus, min_v) = self
                    .v
                    .iter()
                    .enumerate()
                    .map(|(bus, v)| (base.from_perm[bus], v.norm()))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap_or((0, 0.0));
                let max_v = self.v.iter().map(|v| v.norm()).fold(0.0, f64::max);
                return self.failed(
                    SolveStage::NewtonFailed,
                    it,
                    residual,
                    format!("{e}; AC |V| min={min_v} at bus {min_bus}, max={max_v}"),
                );
            }
            for k in 0..active {
                self.va[k] -= self.mismatch[k];
            }
            for k in 0..base.npq {
                self.vm[k] -= self.mismatch[active + k];
            }
            for k in 0..self.v.len() {
                self.v_norm[k] = Complex64::from_polar(1., self.va[k]);
                self.v[k] = self.v_norm[k] * self.vm[k];
            }
            residual = self.evaluate(base);
            self.timings.ac_iterations += iteration_started.elapsed();
            if !residual.is_finite() {
                return self.failed(
                    SolveStage::NewtonFailed,
                    it + 1,
                    residual,
                    "non-finite AC mismatch".into(),
                );
            }
            if residual < tol {
                return self.success(it + 1, residual);
            }
        }
        self.failed(
            SolveStage::NewtonFailed,
            max_iter,
            residual,
            "Newton iteration limit reached".into(),
        )
    }
    fn evaluate(&mut self, base: &SharedBaseline) -> f64 {
        csc_matvec_and_scalc(
            base.y_bus.col_offsets(),
            base.y_bus.row_indices(),
            &self.y_values,
            &self.v,
            &mut self.ibus,
            &mut self.s_calc,
        );
        if self
            .s_calc
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return f64::INFINITY;
        }
        fill_f_from_scalc::<false>(
            &self.s_calc,
            base.s_bus.as_slice(),
            base.npq,
            base.npv + base.npq,
            &mut self.mismatch,
        )
    }
    fn reset_timings(&mut self) {
        self.ac_numeric.stats = LinearSolveStats::default();
        if let Some(dc) = &mut self.dc_numeric {
            dc.stats = LinearSolveStats::default();
        }
        self.timings = SolveTimings::default();
        self.ac_started = None;
    }
    fn finish_ac_timing(&mut self) {
        if let Some(started) = self.ac_started.take() {
            self.timings.ac = started.elapsed();
        }
    }
    fn linear_stats(&self) -> LinearSolveStats {
        self.ac_numeric.stats.add(
            self.dc_numeric
                .as_ref()
                .map_or_else(LinearSolveStats::default, |s| s.stats),
        )
    }
    fn success(&mut self, iterations: usize, residual: f64) -> CaseStatus {
        self.finish_ac_timing();
        CaseStatus {
            converged: true,
            stage: SolveStage::Converged,
            iterations,
            residual,
            error: None,
            timings: self.timings,
            linear: self.linear_stats(),
        }
    }
    fn failed(
        &mut self,
        stage: SolveStage,
        iterations: usize,
        residual: f64,
        error: String,
    ) -> CaseStatus {
        self.finish_ac_timing();
        CaseStatus {
            converged: false,
            stage,
            iterations,
            residual,
            error: Some(error),
            timings: self.timings,
            linear: self.linear_stats(),
        }
    }
}

#[cfg(test)]
mod timing_tests {
    use super::*;
    use crate::basic::ecs::{
        elements::PPNetwork,
        network::{DataOps, PowerFlow, PowerGrid},
    };

    #[test]
    fn nonfinite_refactored_solve_retries_factor_with_original_rhs() {
        let analysis = SharedAnalysis::new(&[0, 2, 4], &[0, 1, 0, 1]).unwrap();
        let mut numeric = NumericKlu::new();
        numeric
            .solve(&analysis, &mut [2., 1., 1., 2.], &mut [1., 1.])
            .unwrap();
        // Make the old first pivot tiny while another row has a healthy pivot.
        // Refactor retains the old pivot and overflows; factor can choose again.
        let (row, col) = unsafe {
            (
                (*(*numeric.solver.numeric).Pnum) as usize,
                (*(*analysis.owner.symbolic).Q) as usize,
            )
        };
        let mut values = [1.; 4];
        values[2 * col + row] = 1e-310;
        let original_rhs = [values[0] + values[2], values[1] + values[3]];
        let mut rhs = original_rhs;
        numeric.solve(&analysis, &mut values, &mut rhs).unwrap();
        assert!(rhs.iter().all(|&x| (x - 1.).abs() < 1e-12), "{rhs:?}");
        assert_eq!(numeric.rhs_backup, original_rhs);
        assert_eq!(numeric.stats.refactor_calls, 1);
        assert_eq!(numeric.stats.factor_calls, 2);
        assert_eq!(numeric.stats.refactor_solve_retries, 1);
        assert!(numeric.solver.symbolic.is_null());
        assert!(!analysis.owner.symbolic.is_null());
    }

    #[test]
    fn nonfinite_full_factor_result_stops_and_private_solver_recovers() {
        let analysis = SharedAnalysis::new(&[0, 1], &[0]).unwrap();
        let mut numeric = NumericKlu::new();
        let error = numeric
            .solve(&analysis, &mut [1e-310], &mut [1.])
            .unwrap_err();
        assert!(error.contains("non-finite"), "{error}");
        assert_eq!(numeric.stats.factor_calls, 1);
        assert_eq!(numeric.stats.refactor_solve_retries, 0);
        assert!(numeric.solver.numeric.is_null());
        assert!(!analysis.owner.symbolic.is_null());
        assert!(
            numeric
                .solve(&analysis, &mut [f64::NAN], &mut [1.])
                .is_err()
        );
        assert_eq!(numeric.stats.factor_calls, 1);
        let mut rhs = [2.];
        numeric.solve(&analysis, &mut [1.], &mut rhs).unwrap();
        assert_eq!(rhs, [2.]);
        assert_eq!(numeric.stats.factor_calls, 2);
    }

    #[test]
    fn ac_workers_use_shared_solved_voltage_without_dc_allocations() {
        let net = serde_json::from_str::<crate::io::pandapower::Network>(
            crate::testcases::case_ieee39::IEEE_39,
        )
        .unwrap();
        let mut grid = PowerGrid::default();
        grid.world_mut().insert_resource(PPNetwork(net));
        grid.init_pf_net();
        let mut base =
            SharedBaseline::from_mat(grid.world().resource::<PowerFlowMat>(), &[]).unwrap();
        let mut preparation = N1Worker::new(&base);
        let solved = preparation.solve_base(&base, 50, 1e-10);
        assert!(solved.converged);
        let symbolic_ids = base.symbolic_ids();
        let pattern_ids = base.pattern_ids();
        base.set_ac_start(&solved.voltage).unwrap();
        assert_eq!(base.symbolic_ids(), symbolic_ids);
        assert_eq!(base.pattern_ids(), pattern_ids);
        let mut worker = N1Worker::new_ac(&base);
        assert!(!worker.uses_dc());
        assert_eq!(worker.dc_values.capacity(), 0);
        assert_eq!(worker.dc_shift.capacity(), 0);
        assert_eq!(worker.dc_rhs.capacity(), 0);
        assert_eq!(worker.allocation_stats().dc.current_bytes, 0);
        assert_eq!(worker.allocation_stats().dc.peak_bytes, 0);
        let base_again = worker.solve_ac_current(&base, 0, 1e-10);
        assert!(base_again.converged);
        assert_eq!(base_again.iterations, 0);
        assert_eq!(base_again.timings.dc, Duration::ZERO);
        assert_eq!(base_again.timings.attempted_iterations, 0);
        assert_eq!(base_again.voltage, solved.voltage);
        // The execution path borrows this same working allocation on every case.
        // A changed Ybus forces real Newton work, including the failure return path.
        let voltage_ptr = worker.voltage().as_ptr();
        let shared_start = base.ac_start().unwrap().to_vec();
        worker.y_values[0] *= 1.001;
        let exhausted = worker.solve_ac_current_status(&base, 0, 1e-10);
        assert!(!exhausted.converged);
        assert_eq!(exhausted.stage, SolveStage::NewtonFailed);
        assert_eq!(worker.voltage().as_ptr(), voltage_ptr);
        let in_place = worker.solve_ac_current_status(&base, 50, 1e-10);
        assert!(in_place.converged, "{:?}", in_place.error);
        assert!(in_place.iterations > 0);
        assert_eq!(worker.voltage().as_ptr(), voltage_ptr);
        let expected_voltage = worker.voltage().to_vec();
        assert_eq!(base.ac_start().unwrap(), shared_start.as_slice());
        let changed = worker.solve_ac_current(&base, 50, 1e-10);
        assert_eq!(changed.voltage, expected_voltage);
        assert_eq!(changed.iterations, in_place.iterations);
        assert_eq!(changed.residual, in_place.residual);
        assert_ne!(changed.voltage.as_ptr(), worker.voltage().as_ptr());
        assert!(changed.converged, "{:?}", changed.error);
        assert!(changed.timings.attempted_iterations > 0);
        assert_eq!(changed.timings.dc, Duration::ZERO);
        assert_eq!(worker.symbolic_ids(), symbolic_ids);
        assert_eq!(worker.pattern_ids(), pattern_ids);
        let unsupported = worker.solve_base(&base, 50, 1e-10);
        assert_eq!(unsupported.stage, SolveStage::DcFailed);
        assert_eq!(worker.dc_values.capacity(), 0);
        assert_eq!(worker.dc_shift.capacity(), 0);
        assert_eq!(worker.dc_rhs.capacity(), 0);
        assert_eq!(worker.allocation_stats().dc.current_bytes, 0);

        let before = base.ac_start().unwrap().to_vec();
        assert!(base.set_ac_start(&before[..before.len() - 1]).is_err());
        let mut invalid = before.clone();
        invalid[0] = Complex64::new(f64::NAN, 0.0);
        assert!(base.set_ac_start(&invalid).is_err());
        invalid.copy_from_slice(&before);
        invalid[base.npq + base.npv] *= 1.01;
        assert!(base.set_ac_start(&invalid).is_err());
        assert_eq!(base.ac_start().unwrap(), before);
    }

    #[test]
    fn timings_reset_and_count_failed_linear_attempts() {
        let net = serde_json::from_str::<crate::io::pandapower::Network>(
            crate::testcases::case_ieee39::IEEE_39,
        )
        .unwrap();
        let mut grid = PowerGrid::default();
        grid.world_mut().insert_resource(PPNetwork(net));
        grid.init_pf_net();
        let base = SharedBaseline::from_mat(grid.world().resource::<PowerFlowMat>(), &[]).unwrap();
        let mut worker = N1Worker::new(&base);
        let solved = worker.solve_base(&base, 50, 1e-10);
        assert!(solved.converged);
        assert_eq!(solved.timings.attempted_iterations, solved.iterations);
        assert!(solved.timings.dc > Duration::ZERO);
        assert!(solved.timings.ac >= solved.timings.ac_iterations);

        let no_steps = worker.solve_base(&base, 0, 1e-10);
        assert_eq!(no_steps.stage, SolveStage::NewtonFailed);
        assert_eq!(no_steps.timings.attempted_iterations, 0);
        assert_eq!(no_steps.timings.ac_iterations, Duration::ZERO);
        assert!(no_steps.timings.dc > Duration::ZERO);
        assert!(no_steps.timings.ac > Duration::ZERO);

        let invalid = worker.solve_current(&base, 50, -1.0);
        assert_eq!(invalid.timings.dc, Duration::ZERO);
        assert_eq!(invalid.timings.ac, Duration::ZERO);
        assert_eq!(invalid.timings.ac_iterations, Duration::ZERO);
        assert_eq!(invalid.timings.attempted_iterations, 0);

        worker.dc_values.fill(0.0);
        let dc_failed = worker.solve_current(&base, 50, 1e-10);
        assert_eq!(dc_failed.stage, SolveStage::DcFailed);
        assert!(dc_failed.timings.dc > Duration::ZERO);
        assert_eq!(dc_failed.timings.ac, Duration::ZERO);
        assert_eq!(dc_failed.timings.attempted_iterations, 0);

        assert!(worker.solve_base(&base, 50, 1e-10).converged);
        worker.y_values.fill(Complex64::new(0.0, 0.0));
        let ac_failed = worker.solve_current(&base, 50, 1e-10);
        assert_eq!(ac_failed.stage, SolveStage::NewtonFailed);
        assert_eq!(ac_failed.iterations, 0);
        assert_eq!(ac_failed.timings.attempted_iterations, 1);
        assert!(ac_failed.timings.ac_iterations > Duration::ZERO);
        assert!(ac_failed.timings.ac >= ac_failed.timings.ac_iterations);
        assert!(worker.solve_base(&base, 50, 1e-10).converged);
    }
}
