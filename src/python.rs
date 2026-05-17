#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::{PyArray1, IntoPyArray};
#[cfg(feature = "python")]
use bevy_app::App;
#[cfg(feature = "python")]
use crate::prelude::*;
#[cfg(feature = "python")]
use crate::basic::ecs::{elements::*, powerflow::systems::*};
#[cfg(feature = "python")]
use crate::io::pandapower::load_csv_zip;
#[cfg(feature = "python")]
use bevy_ecs::prelude::*;

#[cfg(feature = "python")]
#[pyclass(unsendable)]
pub struct PowerGrid {
    app: App,
}

#[cfg(feature = "python")]
#[pymethods]
impl PowerGrid {
    #[new]
    #[pyo3(signature = (case_path=None, qlim=false, **kwargs))]
    fn new(case_path: Option<String>, qlim: bool, kwargs: Option<Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut app = default_app();
        
        // Handle QLim
        if qlim {
            // app.add_plugins(crate::basic::ecs::powerflow::qlim::QLimPlugin);
        }

        // Handle additional plugins from kwargs
        if let Some(args) = kwargs {
            if let Some(branch_analysis) = args.get_item("branch_analysis")? {
                if branch_analysis.extract::<bool>()? {
                    app.add_plugins(crate::basic::ecs::powerflow::branch_data::BranchAnalysisPlugin);
                }
            }
        }

        if let Some(path) = case_path {
            let net = load_csv_zip(&path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            app.world_mut().insert_resource(PPNetwork(net));
        }

        Ok(Self { app })
    }

    /// Run a single power flow calculation.
    fn run_pf(&mut self) {
        self.app.update();
    }

    /// Get the voltage magnitudes (Vm) of all buses as a Numpy array.
    fn get_bus_vm<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let results = self.app.world().get_resource::<PowerFlowResult>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow has not been run yet"))?;
        
        let vm: Vec<f64> = results.v.iter().map(|c| c.norm()).collect();
        Ok(vm.into_pyarray(py))
    }

    /// Get the voltage angles (Va) in degrees of all buses as a Numpy array.
    fn get_bus_va<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let results = self.app.world().get_resource::<PowerFlowResult>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow has not been run yet"))?;
        
        let va: Vec<f64> = results.v.iter().map(|c| {
            let (_, arg) = c.to_polar();
            arg.to_degrees()
        }).collect();
        Ok(va.into_pyarray(py))
    }

    /// Returns the number of iterations for the last power flow run.
    fn iterations(&self) -> PyResult<usize> {
        let results = self.app.world().get_resource::<PowerFlowResult>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow has not been run yet"))?;
        Ok(results.iterations)
    }

    /// Get incidence matrix if BranchAnalysisPlugin was added.
    fn get_incidence<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.app.world();
        let res = world.get_resource::<crate::basic::ecs::powerflow::branch_data::BranchAnalysisRes>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("BranchAnalysisPlugin not added"))?;
        
        let dict = pyo3::types::PyDict::new(py);
        let (offsets, indices, values) = res.incidence.csr_data();
        dict.set_item("indptr", offsets.to_vec().into_pyarray(py))?;
        dict.set_item("indices", indices.to_vec().into_pyarray(py))?;
        dict.set_item("data", values.to_vec().into_pyarray(py))?;
        dict.set_item("shape", (res.incidence.nrows(), res.incidence.ncols()))?;
        Ok(dict)
    }
}

#[cfg(feature = "python")]
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(feature = "python")]
#[pymodule]
fn rustpower(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PowerGrid>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
