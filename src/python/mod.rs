pub mod handles;
pub mod grid;
pub mod solver;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(feature = "python")]
#[pyfunction]
fn features() -> Vec<&'static str> {
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
    m.add_class::<grid::PowerGrid>()?;
    m.add_class::<solver::NewtonSolver>()?;
    
    m.add_class::<handles::BusHandle>()?;
    m.add_class::<handles::LineHandle>()?;
    m.add_class::<handles::TrafoHandle>()?;
    m.add_class::<handles::LoadHandle>()?;
    m.add_class::<handles::GenHandle>()?;
    m.add_class::<handles::ExtGridHandle>()?;
    m.add_class::<handles::ShuntHandle>()?;
    m.add_class::<handles::SGenHandle>()?;
    m.add_class::<handles::SwitchHandle>()?;
    
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(features, m)?)?;
    Ok(())
}
