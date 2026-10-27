use rustpower_klu_sys::*;
use rustpower_sol_klu::KLUSolver;
use std::time::{Duration, Instant};

#[cfg(feature = "c_amd")]
extern "C" {
    fn rustpower_ybus_supported() -> i32;
    fn rustpower_ybus_analyze(
        p: *const usize,
        i: *const usize,
        active: i64,
        npq: i64,
        common: *mut klu_l_common,
    ) -> *mut klu_l_symbolic;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PatternKey {
    offsets: Vec<usize>,
    rows: Vec<usize>,
    npv: usize,
    npq: usize,
    btf: i32,
    ordering: i32,
}
impl PatternKey {
    fn matches(
        &self,
        p: &[usize],
        i: &[usize],
        npv: usize,
        npq: usize,
        btf: i32,
        ordering: i32,
    ) -> bool {
        self.npv == npv
            && self.npq == npq
            && self.btf == btf
            && self.ordering == ordering
            && self.offsets == p
            && self.rows == i
    }
}
/// Ordering implementation used when Ybus specialization is enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YbusOrderingBackend {
    C,
    Rust,
}
impl Default for YbusOrderingBackend {
    fn default() -> Self {
        Self::Rust
    }
}
impl YbusOrderingBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::C => "c",
            Self::Rust => "rust",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct YbusOrderingStats {
    pub optimized: bool,
    pub preparations: u64,
    pub last_preparation: Duration,
    pub pattern_key_bytes: usize,
}

#[cfg(feature = "c_amd")]
use crate::powerflow::eligible;

/// Optional preparation state owned by the experimental solver/plugin.
/// The legacy KLUSolver layout and methods are unchanged.
pub struct KluOrdering {
    owner: Option<usize>,
    ybus_enabled: bool,
    ybus_backend: YbusOrderingBackend,
    pattern_key: Option<PatternKey>,
    ybus_stats: YbusOrderingStats,
}
impl Default for KluOrdering {
    fn default() -> Self {
        Self::new(YbusOrderingBackend::Rust)
    }
}
impl KluOrdering {
    pub fn new(backend: YbusOrderingBackend) -> Self {
        Self {
            owner: None,
            ybus_enabled: true,
            ybus_backend: backend,
            pattern_key: None,
            ybus_stats: YbusOrderingStats::default(),
        }
    }
    fn clear_analysis(&mut self, solver: &mut KLUSolver) {
        unsafe {
            klu_l_free_numeric(&mut solver.numeric, solver.common);
            klu_l_free_symbolic(&mut solver.symbolic, solver.common);
        }
    }

    /// Switching implementations invalidates both Symbolic and Numeric.
    pub fn set_ybus_ordering_backend(
        &mut self,
        solver: &mut KLUSolver,
        backend: YbusOrderingBackend,
    ) {
        if self.ybus_backend != backend {
            self.clear_analysis(solver);
            self.pattern_key = None;
            self.ybus_backend = backend;
            self.ybus_stats.optimized = false;
            self.ybus_stats.pattern_key_bytes = 0;
            self.ybus_stats.last_preparation = Duration::ZERO;
        }
    }
    pub fn ybus_ordering_backend(&self) -> YbusOrderingBackend {
        self.ybus_backend
    }
    pub fn ybus_ordering_enabled(&self) -> bool {
        self.ybus_enabled
    }
    pub fn set_ybus_ordering(&mut self, solver: &mut KLUSolver, enabled: bool) {
        if self.ybus_enabled != enabled {
            self.clear_analysis(solver);
            self.pattern_key = None;
            self.ybus_enabled = enabled;
            self.ybus_stats.optimized = false;
            self.ybus_stats.pattern_key_bytes = 0;
            self.ybus_stats.last_preparation = Duration::ZERO;
        }
    }
    pub fn ybus_ordering_stats(&self) -> YbusOrderingStats {
        self.ybus_stats
    }
    /// Prepare from exact, already-permuted Ybus structure. Returns true when the context was rebuilt.
    /// The caller must invalidate its Jacobian cache when true. Numeric Ybus values are irrelevant here.
    pub fn prepare_ybus(
        &mut self,
        solver: &mut KLUSolver,
        p: &[usize],
        i: &[usize],
        npv: usize,
        npq: usize,
    ) -> bool {
        if !self.ybus_enabled {
            return false;
        }
        let (btf, ordering) = unsafe { ((*solver.common).btf, (*solver.common).ordering) };
        if self.owner == Some(solver.common as usize)
            && !solver.symbolic.is_null()
            && self
                .pattern_key
                .as_ref()
                .is_some_and(|k| k.matches(p, i, npv, npq, btf, ordering))
        {
            return false;
        }
        self.clear_analysis(solver);
        self.ybus_stats.optimized = false;
        let started = Instant::now();
        let supported = cfg!(target_pointer_width = "64") && KLU_MAIN_VERSION == 2;
        if ordering == 0 && supported {
            let symbolic = match self.ybus_backend {
                #[cfg(feature = "c_amd")]
                YbusOrderingBackend::C
                    if eligible(p, i, npv, npq) && unsafe { rustpower_ybus_supported() != 0 } =>
                unsafe {
                    rustpower_ybus_analyze(
                        p.as_ptr(),
                        i.as_ptr(),
                        (npv + npq) as i64,
                        npq as i64,
                        solver.common,
                    )
                },
                YbusOrderingBackend::Rust => unsafe {
                    crate::klu_rust::analyze(p, i, npv, npq, solver.common)
                },
                _ => std::ptr::null_mut(),
            };
            if !symbolic.is_null() {
                solver.symbolic = symbolic;
                self.ybus_stats.optimized = true;
            } else {
                unsafe {
                    (*solver.common).status = 0;
                }
            } // Leave the original KLU analysis available on failure.
        }
        self.owner = Some(solver.common as usize);
        self.pattern_key = Some(PatternKey {
            offsets: p.to_vec(),
            rows: i.to_vec(),
            npv,
            npq,
            btf,
            ordering,
        });
        self.ybus_stats.pattern_key_bytes = (p.len() + i.len()) * std::mem::size_of::<usize>();
        self.ybus_stats.preparations += 1;
        self.ybus_stats.last_preparation = started.elapsed();
        true
    }
}
