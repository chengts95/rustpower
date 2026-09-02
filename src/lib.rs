pub mod basic;

/// Performance probe timer: `timeit!(path::to::COUNTER, { ... })` runs the
/// block and, when the `probe` cargo feature is enabled, adds its wall-clock
/// nanoseconds to the given `AtomicU64` counter.
///
/// Probes exist to measure RELEASE performance, so the gate is the feature,
/// not the profile: `cargo test --release --features probe` gives fully
/// optimized code WITH instrumentation. With the feature off (the default)
/// the macro expands to just the block — the counter path is never
/// name-resolved, so `*_probe` modules do not need to exist.
#[macro_export]
macro_rules! timeit {
    ($counter:expr, $body:block) => {{
        #[cfg(feature = "probe")]
        let __timeit_start = std::time::Instant::now();
        let __timeit_out = $body;
        #[cfg(feature = "probe")]
        ($counter).fetch_add(
            __timeit_start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        __timeit_out
    }};
}

/// Probe event counter companion of [`timeit!`]: `probe_count!(COUNTER)`
/// increments by one when the `probe` feature is on; vanishes otherwise.
#[macro_export]
macro_rules! probe_count {
    ($counter:expr) => {
        #[cfg(feature = "probe")]
        ($counter).fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    };
}

#[allow(non_snake_case)]
pub mod lm;
pub mod new_opf;
pub mod new_pf;
#[allow(non_snake_case)]
pub mod opf;

pub mod bevy_cmdbuffer;
pub mod io;
pub mod testcases;
pub mod timeseries;

#[cfg(feature = "python")]
pub mod python;

pub mod prelude {
    pub use crate::basic::ecs::elements::PPNetwork;
    pub use crate::basic::ecs::gn_plugin::GnPlugin;
    pub use crate::basic::ecs::lm_plugin::LmPlugin;
    pub use crate::basic::ecs::network::{DataOps, PowerFlow, PowerGrid};
    pub use crate::basic::ecs::plugin::{ActiveSolver, IwamotoPlugin, default_app};
    pub use crate::basic::ecs::post_processing::PostProcessing;
    pub use crate::basic::ecs::powerflow::prelude::PowerFlowResult;
    pub use crate::basic::*;
    pub use crate::io::pandapower;
}
