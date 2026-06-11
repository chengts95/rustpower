//! Python bindings for the RustPower simulation framework.
#![cfg(feature = "python")]
#![allow(dead_code)]
pub mod handles;
pub mod grid;
pub mod solver;
pub mod network;

use pyo3::prelude::*;

/// Get the version of the rustpower package.
#[pyfunction]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get the list of enabled features in this build.
#[pyfunction]
pub fn features() -> Vec<&'static str> {
    let mut f = Vec::new();
    if cfg!(feature = "klu") { f.push("klu"); }
    if cfg!(feature = "faer") { f.push("faer"); }
    if cfg!(feature = "rsparse") { f.push("rsparse"); }
    if cfg!(feature = "archive") { f.push("archive"); }
    if cfg!(feature = "arrow") { f.push("arrow"); }
    if cfg!(feature = "python") { f.push("python"); }
    f
}

/// Load a pandapower network from a CSV-ZIP file.
#[pyfunction]
pub fn load_csv_zip(path: String) -> PyResult<crate::io::pandapower::Network> {
    crate::io::pandapower::load_csv_zip(&path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))
}

#[pymodule]
pub fn rustpower(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    
    // High-level API classes in root module
    m.add_class::<grid::PowerGrid>()?;
    m.add_class::<grid::GridEditor>()?;
    m.add_class::<grid::SolveReport>()?;
    
    // IO classes
    m.add_class::<crate::io::pandapower::Network>()?;
    m.add_class::<crate::io::pandapower::Bus>()?;
    m.add_class::<crate::io::pandapower::Line>()?;
    m.add_class::<crate::io::pandapower::Transformer>()?;
    m.add_class::<crate::io::pandapower::Load>()?;
    m.add_class::<crate::io::pandapower::Gen>()?;
    m.add_class::<crate::io::pandapower::ExtGrid>()?;
    m.add_class::<crate::io::pandapower::Shunt>()?;
    m.add_class::<crate::io::pandapower::SGen>()?;
    m.add_class::<crate::io::pandapower::Switch>()?;
    
    // Elemental Handles in root module
    m.add_class::<handles::BusHandle>()?;
    m.add_class::<handles::LineHandle>()?;
    m.add_class::<handles::TrafoHandle>()?;
    m.add_class::<handles::LoadHandle>()?;
    m.add_class::<handles::GenHandle>()?;
    m.add_class::<handles::ExtGridHandle>()?;
    m.add_class::<handles::ShuntHandle>()?;
    m.add_class::<handles::SGenHandle>()?;
    m.add_class::<handles::SwitchHandle>()?;
    
    // Register Solver as a submodule
    let solver_module = PyModule::new(py, "rustpower.solver")?;
    solver_module.add_class::<solver::NewtonSolver>()?;
    m.add_submodule(&solver_module)?;
    
    // Attempt robust sys.modules registration so 'import rustpower.solver' works
    let sys = py.import("sys")?;
    let modules: Bound<'_, pyo3::types::PyDict> = sys.getattr("modules")?.downcast_into()?;
    modules.set_item("rustpower.solver", &solver_module)?;
    
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(features, m)?)?;
    m.add_function(wrap_pyfunction!(load_csv_zip, m)?)?;
    Ok(())
}
