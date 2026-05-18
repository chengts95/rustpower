#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::IntoPyArray;
#[cfg(feature = "python")]
use crate::prelude::*;
#[cfg(feature = "python")]
use crate::basic::ecs::elements::BusID;
#[cfg(feature = "python")]
use crate::basic::ecs::post_processing::{VBusResult, SBusResult};
#[cfg(feature = "python")]
use crate::io::pandapower::load_csv_zip;
#[cfg(feature = "python")]
use num_complex::ComplexFloat;
#[cfg(feature = "python")]
use pyo3::types::PyDictMethods;

#[cfg(feature = "python")]
#[pyclass(unsendable)]
pub struct PowerGrid {
    inner: crate::prelude::PowerGrid,
}

#[cfg(feature = "python")]
#[pymethods]
impl PowerGrid {
    #[new]
    #[pyo3(signature = (case_path=None, _qlim=false, **kwargs))]
    fn new(case_path: Option<String>, _qlim: bool, kwargs: Option<Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut inner = crate::prelude::PowerGrid::default();
        
        // Handle additional plugins from kwargs BEFORE init_pf_net
        if let Some(args) = kwargs {
            if let Some(branch_analysis) = args.get_item("branch_analysis")? {
                if branch_analysis.extract::<bool>()? {
                    inner.app_mut().add_plugins(crate::basic::ecs::powerflow::branch_data::BranchAnalysisPlugin);
                }
            }
        }

        if let Some(path) = case_path {
            let net = load_csv_zip(&path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            inner.world_mut().insert_resource(PPNetwork(net));
        }

        Ok(Self { inner })
    }

    /// Initialize the power flow network (runs Bevy Startup systems).
    fn init_pf(&mut self) {
        self.inner.init_pf_net();
    }

    /// Run a single power flow calculation.
    fn run_pf(&mut self) {
        self.inner.run_pf();
    }

    /// Run post-processing to extract results.
    fn post_process(&mut self) {
        self.inner.post_process();
    }

    /// Returns bus results (Vm, Va, P, Q) as a dictionary of Numpy arrays.
    fn get_bus_results<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world_mut();
        
        let mut bus_ids = Vec::new();
        let mut vms = Vec::new();
        let mut vas = Vec::new();
        let mut ps = Vec::new();
        let mut qs = Vec::new();

        let mut query = world.query::<(&BusID, &VBusResult, &SBusResult)>();
        for (id, v, s) in query.iter(world) {
            bus_ids.push(id.0 as i32);
            vms.push(v.0.norm());
            vas.push(v.0.arg().to_degrees());
            ps.push(s.0.re());
            qs.push(s.0.im());
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("bus_id", bus_ids.into_pyarray(py))?;
        dict.set_item("vm_pu", vms.into_pyarray(py))?;
        dict.set_item("va_degree", vas.into_pyarray(py))?;
        dict.set_item("p_mw", ps.into_pyarray(py))?;
        dict.set_item("q_mvar", qs.into_pyarray(py))?;
        Ok(dict)
    }

    /// Returns line results as a dictionary of Numpy arrays.
    fn get_line_results<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world_mut();
        
        let mut p_f = Vec::new();
        let mut q_f = Vec::new();
        let mut p_t = Vec::new();
        let mut q_t = Vec::new();
        let mut loading = Vec::new();

        let mut query = world.query::<&crate::basic::ecs::post_processing::LineResultData>();
        for data in query.iter(world) {
            p_f.push(data.p_from_mw);
            q_f.push(data.q_from_mvar);
            p_t.push(data.p_to_mw);
            q_t.push(data.q_to_mvar);
            loading.push(data.loading_percent);
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("p_from_mw", p_f.into_pyarray(py))?;
        dict.set_item("q_from_mvar", q_f.into_pyarray(py))?;
        dict.set_item("p_to_mw", p_t.into_pyarray(py))?;
        dict.set_item("q_to_mvar", q_t.into_pyarray(py))?;
        dict.set_item("loading_percent", loading.into_pyarray(py))?;
        Ok(dict)
    }

    /// Returns the number of iterations for the last power flow run.
    fn iterations(&self) -> PyResult<usize> {
        let results = self.inner.world().get_resource::<PowerFlowResult>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow has not been run yet"))?;
        Ok(results.iterations)
    }

    /// Get incidence matrix if BranchAnalysisPlugin was added.
    fn get_incidence<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world();
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
