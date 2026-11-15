//! Levenberg-Marquardt power flow on a fixed augmented sparse system.
//!
//! The implementation shares one Ybus-derived analysis across iterations and
//! updates only numerical values. The augmented system is factored directly;
//! normal equations are not formed explicitly.

pub mod block;
pub mod cache;
pub mod exact;
pub mod flat;
pub mod kernels;
pub mod pattern;
pub mod residual;
pub mod step_control;
pub mod study;

pub use block::BlockDesc;
pub use cache::YbusAnalysisCache;
pub use exact::driver::{LmDriver, LmResult};
pub use flat::{FlatLayout, fill_kkt, fill_kkt_flat};
pub use kernels::{apply_mu_delta, fill_h, fill_jt};
pub use pattern::KktPattern;
pub use step_control::{DampingMetric, LmOptions, TrustRegionOptions};
pub use study::{LmpfBackend, LmpfReport, LmpfRunOptions, LmpfStatus, PreparedLmpf};
