//! Shared numerical inputs for all implementations.
//!
//! `OPFData` defines base MVA, CSC admittances, branch endpoints, generator
//! incidence, per-unit loads and bounds, and quadratic costs in MW units.
//! Its range methods define `[theta (rad), Vm (p.u.), Pg (p.u.), Qg (p.u.)]`.
//! Bounds and initial points must use these same model indices. Conversion from
//! source tables stays in `io::pandapower`; ECS integration lives in `adapters`.
//!
//! `NewOPFData` adds the existing mapped cache, shared without changing the
//! reference model type. Each assembly/evaluation strategy owns additional caches.
use crate::new_opf::assembly::mapped::symbolic::SymbolicCache;
pub use crate::opf::problem::OPFData;

/// New OPF Data structure that integrates the Symbolic Cache for high performance.
pub struct NewOPFData {
    pub base: OPFData,
    pub cache: SymbolicCache,
}

impl NewOPFData {
    pub fn new(base: OPFData) -> Self {
        let cache = SymbolicCache::analyze(&base);
        Self { base, cache }
    }
}

impl std::ops::Deref for NewOPFData {
    type Target = OPFData;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl std::ops::DerefMut for NewOPFData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}
