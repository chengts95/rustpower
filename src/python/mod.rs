#[cfg(feature = "python")]
pub mod handles;
#[cfg(feature = "python")]
pub mod grid;
#[cfg(feature = "python")]
pub mod solver;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyfunction]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(feature = "python")]
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

#[cfg(feature = "python")]
#[pymodule]
pub fn rustpower(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    
    // High-level API classes in root module
    m.add_class::<grid::PowerGrid>()?;
    m.add_class::<grid::GridBuilder>()?;
    
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
    Ok(())
}
