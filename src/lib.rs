mod basic;

#[macro_export]
macro_rules! timeit {
    ($counter:expr, $body:block) => {{
        #[cfg(feature = "probe")]
        let __timeit_start = std::time::Instant::now();
        #[cfg(feature = "probe")]
        let __timeit_out = $body;
        #[cfg(feature = "probe")]
        ($counter).fetch_add(
            __timeit_start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        #[cfg(not(feature = "probe"))]
        $body
    }};
}

#[macro_export]
macro_rules! probe_count {
    ($counter:expr) => {
        #[cfg(feature = "probe")]
        $counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    };
}

pub mod bevy_cmdbuffer;
pub mod cpf;
pub mod io;
pub mod lm;
pub mod native;
#[cfg(feature = "python")]
pub mod python;
pub mod testcases;
pub mod timeseries;

pub mod prelude {
    pub use crate::basic::ecs::elements::PPNetwork;
    pub use crate::basic::ecs::network::{DataOps, PowerFlow, PowerGrid};
    pub use crate::basic::ecs::plugin::{CustomSolverActive, IwamotoPlugin, default_app};
    pub use crate::basic::ecs::post_processing::PostProcessing;
    pub use crate::basic::ecs::powerflow::prelude::PowerFlowResult;
    pub use crate::basic::*;
    pub use crate::cpf::{CpfNose, CpfOptions, CpfPoint, CpfReport, CpfStats, CpfTermination};
    pub use crate::io::pandapower;
    pub use crate::lm::{LmpfBackend, LmpfReport, LmpfRunOptions, LmpfStatus, PreparedLmpf};
    pub use crate::native::solver::NewtonSolver;
}

#[cfg(feature = "experimental_n1")]
pub mod n1;
