use csv::ReaderBuilder;
use serde::Deserializer;
use serde::{Deserialize, Serialize};
use std::{fs, fs::File};
use std::{io::Read, option::Option};

use serde_json;
use serde_json::{Map, Value};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// This module is used to parse pandapower network parameters
/// Deserializes a number from JSON format.
fn from_number<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: Deserializer<'de>,
{
    let val: serde_json::Value = Deserialize::deserialize(deserializer)?;
    if let serde_json::Value::Number(n) = val {
        let res = n.as_f64().unwrap();
        return Ok(Some(res as i64));
    }
    Ok(None)
}

/// Deserializes a string from JSON format.
fn from_str<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let val: serde_json::Value = Deserialize::deserialize(deserializer)?;
    if let serde_json::Value::Number(n) = val {
        return Ok(Some(n.to_string()));
    }
    if let serde_json::Value::String(s) = val {
        return Ok(Some(s));
    }
    Ok(None)
}

/// Represents a bus in the network.
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct Bus {
    pub index: i64,
    pub in_service: bool,
    pub max_vm_pu: Option<f64>,
    pub min_vm_pu: Option<f64>,
    #[serde(deserialize_with = "from_str")]
    pub name: Option<String>,
    pub r#type: Option<String>, // Added underscore to avoid conflict with Rust keyword
    pub vn_kv: f64,
    #[serde(deserialize_with = "from_number")]
    pub zone: Option<i64>,
}

#[cfg(feature = "python")]
#[pymethods]
impl Bus {
    #[new]
    #[pyo3(signature = (index=0, in_service=true, max_vm_pu=None, min_vm_pu=None, name=None, r#type=None, vn_kv=110.0, zone=None))]
    fn new(index: i64, in_service: bool, max_vm_pu: Option<f64>, min_vm_pu: Option<f64>, name: Option<String>, r#type: Option<String>, vn_kv: f64, zone: Option<i64>) -> Self {
        Self { index, in_service, max_vm_pu, min_vm_pu, name, r#type, vn_kv, zone }
    }
}

/// Represents a generator in the network.
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct Gen {
    pub bus: i64,
    pub controllable: Option<bool>,
    pub in_service: bool,
    pub name: Option<String>,
    pub p_mw: f64,
    pub scaling: f64,
    pub sn_mva: Option<f64>,
    #[serde(rename = "type")]
    pub type_: Option<String>, // Added underscore to avoid conflict with Rust keyword
    pub vm_pu: f64,
    pub slack: bool,
    pub max_p_mw: f64,
    pub min_p_mw: f64,
    pub max_q_mvar: f64,
    pub min_q_mvar: f64,
    pub slack_weight: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl Gen {
    #[new]
    #[pyo3(signature = (bus=0, controllable=None, in_service=true, name=None, p_mw=0.0, scaling=1.0, sn_mva=None, type_=None, vm_pu=1.0, slack=false, max_p_mw=0.0, min_p_mw=0.0, max_q_mvar=0.0, min_q_mvar=0.0, slack_weight=0.0))]
    pub fn new(bus: i64, controllable: Option<bool>, in_service: bool, name: Option<String>, p_mw: f64, scaling: f64, sn_mva: Option<f64>, type_: Option<String>, vm_pu: f64, slack: bool, max_p_mw: f64, min_p_mw: f64, max_q_mvar: f64, min_q_mvar: f64, slack_weight: f64) -> Self {
        Self { bus, controllable, in_service, name, p_mw, scaling, sn_mva, type_: type_, vm_pu, slack, max_p_mw, min_p_mw, max_q_mvar, min_q_mvar, slack_weight }
    }
}

/// Represents a load in the network.
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct Load {
    pub bus: i64,
    pub const_i_percent: f64,
    pub const_z_percent: f64,
    pub controllable: Option<bool>,
    pub in_service: bool,
    pub name: Option<String>,
    pub p_mw: f64,
    pub q_mvar: f64,
    pub scaling: f64,
    pub sn_mva: Option<f64>,
    #[serde(rename = "type")]
    pub type_: Option<String>, // Added underscore to avoid conflict with Rust keyword
}

#[cfg(feature = "python")]
#[pymethods]
impl Load {
    #[new]
    #[pyo3(signature = (bus=0, const_i_percent=0.0, const_z_percent=0.0, controllable=None, in_service=true, name=None, p_mw=0.0, q_mvar=0.0, scaling=1.0, sn_mva=None, type_=None))]
    pub fn new(bus: i64, const_i_percent: f64, const_z_percent: f64, controllable: Option<bool>, in_service: bool, name: Option<String>, p_mw: f64, q_mvar: f64, scaling: f64, sn_mva: Option<f64>, type_: Option<String>) -> Self {
        Self { bus, const_i_percent, const_z_percent, controllable, in_service, name, p_mw, q_mvar, scaling, sn_mva, type_: type_ }
    }
}

/// Represents a line in the network.
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct Line {
    pub c_nf_per_km: f64,
    pub df: f64,
    pub from_bus: i64,
    pub to_bus: i64,
    pub g_us_per_km: f64,
    pub in_service: bool,
    pub length_km: f64,
    pub max_i_ka: f64,
    pub max_loading_percent: Option<f64>,
    pub parallel: i32,
    pub r_ohm_per_km: f64,
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub x_ohm_per_km: f64,
    pub name: Option<String>,
    pub std_type: Option<String>,
}

#[cfg(feature = "python")]
#[pymethods]
impl Line {
    #[new]
    #[pyo3(signature = (from_bus=0, to_bus=0, length_km=1.0, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, g_us_per_km=0.0, in_service=true, parallel=1, max_i_ka=0.0, max_loading_percent=None, type_=None, name=None, std_type=None))]
    fn new(from_bus: i64, to_bus: i64, length_km: f64, r_ohm_per_km: f64, x_ohm_per_km: f64, c_nf_per_km: f64, g_us_per_km: f64, in_service: bool, parallel: i32, max_i_ka: f64, max_loading_percent: Option<f64>, type_: Option<String>, name: Option<String>, std_type: Option<String>) -> Self {
        Self { from_bus, to_bus, length_km, r_ohm_per_km, x_ohm_per_km, c_nf_per_km, g_us_per_km, in_service, parallel, max_i_ka, max_loading_percent, type_: type_, name, std_type, df: 1.0 }
    }
}

/// Represents a transformer in the network.
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct Transformer {
    pub df: f64,
    pub hv_bus: i32,
    pub i0_percent: f64,
    pub in_service: bool,
    pub lv_bus: i32,
    pub max_loading_percent: Option<f64>,
    pub parallel: i32,
    pub pfe_kw: f64,
    pub shift_degree: f64,
    pub sn_mva: f64,
    pub tap_phase_shifter: bool,
    pub vn_hv_kv: f64,
    pub vn_lv_kv: f64,
    pub vk_percent: f64,
    pub vkr_percent: f64,
    pub name: Option<String>,
    pub std_type: Option<String>,
    pub tap_side: Option<String>,
    pub tap_neutral: Option<f64>,
    pub tap_max: Option<f64>,
    pub tap_pos: Option<f64>,
    pub tap_min: Option<f64>,
    pub tap_step_degree: Option<f64>,
    pub tap_step_percent: Option<f64>,
}

#[cfg(feature = "python")]
#[pymethods]
impl Transformer {
    #[new]
    #[pyo3(signature = (hv_bus=0, lv_bus=0, sn_mva=1.0, vn_hv_kv=110.0, vn_lv_kv=10.0, vk_percent=10.0, vkr_percent=0.1, pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0, in_service=true, parallel=1, tap_side=None, tap_pos=None, tap_neutral=None, tap_max=None, tap_min=None, tap_step_percent=None, tap_step_degree=None, tap_phase_shifter=false, name=None, std_type=None))]
    fn new(hv_bus: i32, lv_bus: i32, sn_mva: f64, vn_hv_kv: f64, vn_lv_kv: f64, vk_percent: f64, vkr_percent: f64, pfe_kw: f64, i0_percent: f64, shift_degree: f64, in_service: bool, parallel: i32, tap_side: Option<String>, tap_pos: Option<f64>, tap_neutral: Option<f64>, tap_max: Option<f64>, tap_min: Option<f64>, tap_step_percent: Option<f64>, tap_step_degree: Option<f64>, tap_phase_shifter: bool, name: Option<String>, std_type: Option<String>) -> Self {
        Self { hv_bus, lv_bus, sn_mva, vn_hv_kv, vn_lv_kv, vk_percent, vkr_percent, pfe_kw, i0_percent, shift_degree, in_service, parallel, tap_side, tap_pos, tap_neutral, tap_max, tap_min, tap_step_percent, tap_step_degree, tap_phase_shifter, name, std_type, df: 1.0, max_loading_percent: None }
    }
}

/// Represents an external grid in the network.
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct ExtGrid {
    pub bus: i64,
    pub in_service: bool,
    pub va_degree: f64,
    pub vm_pu: f64,
    pub max_p_mw: Option<f64>,
    pub min_p_mw: Option<f64>,
    pub max_q_mvar: Option<f64>,
    pub min_q_mvar: Option<f64>,
    pub slack_weight: f64,
    pub name: Option<String>,
}

#[cfg(feature = "python")]
#[pymethods]
impl ExtGrid {
    #[new]
    #[pyo3(signature = (bus=0, vm_pu=1.0, va_degree=0.0, in_service=true, slack_weight=1.0, name=None))]
    fn new(bus: i64, vm_pu: f64, va_degree: f64, in_service: bool, slack_weight: f64, name: Option<String>) -> Self {
        Self { bus, vm_pu, va_degree, in_service, slack_weight, name, max_p_mw: None, min_p_mw: None, max_q_mvar: None, min_q_mvar: None }
    }
}

/// Represents the data from the sgen.
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct SGen {
    pub name: Option<String>,
    pub bus: i64,
    pub p_mw: f64,
    pub q_mvar: f64,
    pub sn_mva: Option<f64>,
    pub scaling: f64,
    pub in_service: bool,
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub current_source: bool,
    pub controllable: Option<bool>,
}

#[cfg(feature = "python")]
#[pymethods]
impl SGen {
    #[new]
    #[pyo3(signature = (bus=0, p_mw=0.0, q_mvar=0.0, in_service=true, scaling=1.0, name=None, type_=None))]
    fn new(bus: i64, p_mw: f64, q_mvar: f64, in_service: bool, scaling: f64, name: Option<String>, type_: Option<String>) -> Self {
        Self { bus, p_mw, q_mvar, in_service, scaling, name, type_: type_, sn_mva: None, current_source: false, controllable: None }
    }
}

/// Represents a shunt in the network.
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct Shunt {
    pub bus: i64,
    pub q_mvar: f64,
    pub p_mw: f64,
    pub vn_kv: f64,
    pub step: i32,
    pub max_step: i32,
    pub in_service: bool,
    pub name: Option<String>,
}

#[cfg(feature = "python")]
#[pymethods]
impl Shunt {
    #[new]
    #[pyo3(signature = (bus=0, p_mw=0.0, q_mvar=0.0, vn_kv=110.0, in_service=true, name=None))]
    fn new(bus: i64, p_mw: f64, q_mvar: f64, vn_kv: f64, in_service: bool, name: Option<String>) -> Self {
        Self { bus, p_mw, q_mvar, vn_kv, in_service, name, step: 1, max_step: 1 }
    }
}

#[derive(Debug, Default, PartialEq, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
pub enum SwitchType {
    #[serde(rename = "l")]
    SwitchBusLine,
    #[serde(rename = "t")]
    SwitchBusTransformer,
    #[serde(rename = "t3")]
    SwitchBusTransformer3w,
    #[serde(rename = "b")]
    #[default]
    SwitchTwoBuses,
    Unknown,
}

/// Represents a switch in the network.
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct Switch {
    pub bus: i64,
    pub element: i64,
    pub et: SwitchType,
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub closed: bool,
    pub name: Option<String>,
    pub z_ohm: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl Switch {
    #[new]
    #[pyo3(signature = (bus=0, element=0, et=SwitchType::SwitchTwoBuses, closed=true, name=None))]
    fn new(bus: i64, element: i64, et: SwitchType, closed: bool, name: Option<String>) -> Self {
        Self { bus, element, et, closed, name, type_: None, z_ohm: 0.0 }
    }
}

impl From<&str> for SwitchType {
    fn from(s: &str) -> SwitchType {
        match s {
            "l" => SwitchType::SwitchBusLine,
            "t" => SwitchType::SwitchBusTransformer,
            "t3" => SwitchType::SwitchBusTransformer3w,
            "b" => SwitchType::SwitchTwoBuses,
            _ => SwitchType::Unknown,
        }
    }
}

/// Represents a network.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct Network {
    pub r#gen: Option<Vec<Gen>>,
    pub bus: Vec<Bus>,
    pub load: Option<Vec<Load>>,
    pub line: Option<Vec<Line>>,
    pub trafo: Option<Vec<Transformer>>,
    pub shunt: Option<Vec<Shunt>>,
    pub ext_grid: Option<Vec<ExtGrid>>,
    pub sgen: Option<Vec<SGen>>,
    pub switch: Option<Vec<Switch>>,
    pub f_hz: f64,
    pub sn_mva: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl Network {
    #[new]
    #[pyo3(signature = (f_hz=50.0, sn_mva=100.0))]
    fn py_new(f_hz: f64, sn_mva: f64) -> Self {
        Self {
            f_hz,
            sn_mva,
            ..Default::default()
        }
    }

    /// Load the network from a pandapower net object.
    pub fn from_pp_net(&mut self, net: Bound<'_, PyAny>) -> PyResult<()> {
        let py = net.py();
        let bus_df = net.getattr("bus")?;
        self.bus = self.extract_buses(py, bus_df)?;

        if let Ok(df) = net.getattr("line") { self.line = Some(self.extract_lines(py, df)?); }
        if let Ok(df) = net.getattr("trafo") { self.trafo = Some(self.extract_trafos(py, df)?); }
        if let Ok(df) = net.getattr("load") { self.load = Some(self.extract_loads(py, df)?); }
        if let Ok(df) = net.getattr("gen") { self.r#gen = Some(self.extract_gens(py, df)?); }
        if let Ok(df) = net.getattr("ext_grid") { self.ext_grid = Some(self.extract_ext_grids(py, df)?); }
        if let Ok(df) = net.getattr("shunt") { self.shunt = Some(self.extract_shunts(py, df)?); }
        if let Ok(df) = net.getattr("sgen") { self.sgen = Some(self.extract_sgens(py, df)?); }
        if let Ok(df) = net.getattr("switch") { self.switch = Some(self.extract_switches(py, df)?); }

        if let Ok(f_hz) = net.getattr("f_hz") { self.f_hz = f_hz.extract()?; }
        if let Ok(sn_mva) = net.getattr("sn_mva") { self.sn_mva = sn_mva.extract()?; }

        Ok(())
    }
}

#[cfg(feature = "python")]
impl Network {
    fn get_int_vec(df: &Bound<'_, PyAny>, col: &str) -> PyResult<Vec<i64>> {
        df.getattr(col)?.call_method1("fillna", (0,))?.call_method1("astype", ("int64",))?.call_method0("tolist")?.extract()
    }
    
    fn get_opt_int_vec(py: Python<'_>, df: &Bound<'_, PyAny>, col: &str) -> PyResult<Vec<Option<i64>>> {
        let numpy = py.import("numpy")?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item(numpy.getattr("nan")?, py.None())?;
        df.getattr(col)?.call_method1("replace", (dict,))?.call_method0("tolist")?.extract()
    }

    fn get_int32_vec(df: &Bound<'_, PyAny>, col: &str) -> PyResult<Vec<i32>> {
        df.getattr(col)?.call_method1("fillna", (0,))?.call_method1("astype", ("int32",))?.call_method0("tolist")?.extract()
    }

    fn get_float_vec(df: &Bound<'_, PyAny>, col: &str) -> PyResult<Vec<f64>> {
        df.getattr(col)?.call_method1("fillna", (0.0,))?.call_method1("astype", ("float64",))?.call_method0("tolist")?.extract()
    }
    
    fn get_opt_float_vec(py: Python<'_>, df: &Bound<'_, PyAny>, col: &str) -> PyResult<Vec<Option<f64>>> {
        let numpy = py.import("numpy")?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item(numpy.getattr("nan")?, py.None())?;
        df.getattr(col)?.call_method1("replace", (dict,))?.call_method0("tolist")?.extract()
    }

    fn get_bool_vec(df: &Bound<'_, PyAny>, col: &str) -> PyResult<Vec<bool>> {
        df.getattr(col)?.call_method1("fillna", (false,))?.call_method1("astype", ("bool",))?.call_method0("tolist")?.extract()
    }
    
    fn get_opt_str_vec(py: Python<'_>, df: &Bound<'_, PyAny>, col: &str) -> PyResult<Vec<Option<String>>> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("nan", py.None())?;
        dict.set_item("None", py.None())?;
        df.getattr(col)?.call_method1("astype", ("str",))?.call_method1("replace", (dict,))?.call_method0("tolist")?.extract()
    }

    fn extract_buses(&self, py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<Vec<Bus>> {
        let index = Self::get_int_vec(&df, "index")?;
        let vn_kv = Self::get_float_vec(&df, "vn_kv")?;
        let in_service = Self::get_bool_vec(&df, "in_service")?;
        let names = if df.hasattr("name")? { Self::get_opt_str_vec(py, &df, "name")? } else { vec![None; index.len()] };
        let types = if df.hasattr("type")? { Self::get_opt_str_vec(py, &df, "type")? } else { vec![None; index.len()] };
        
        let zones_f = if df.hasattr("zone")? { Self::get_opt_float_vec(py, &df, "zone")? } else { vec![None; index.len()] };
        let zones: Vec<Option<i64>> = zones_f.into_iter().map(|z| z.map(|v| v as i64)).collect();
        
        let min_vm = if df.hasattr("min_vm_pu")? { Self::get_opt_float_vec(py, &df, "min_vm_pu")? } else { vec![None; index.len()] };
        let max_vm = if df.hasattr("max_vm_pu")? { Self::get_opt_float_vec(py, &df, "max_vm_pu")? } else { vec![None; index.len()] };

        Ok((0..index.len()).map(|i| Bus {
            index: index[i], vn_kv: vn_kv[i], in_service: in_service[i], name: names[i].clone(), r#type: types[i].clone(), zone: zones[i], min_vm_pu: min_vm[i], max_vm_pu: max_vm[i],
        }).collect())
    }

    fn extract_lines(&self, py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<Vec<Line>> {
        let from_bus = Self::get_int_vec(&df, "from_bus")?;
        let to_bus = Self::get_int_vec(&df, "to_bus")?;
        let length_km = Self::get_float_vec(&df, "length_km")?;
        let r_ohm = Self::get_float_vec(&df, "r_ohm_per_km")?;
        let x_ohm = Self::get_float_vec(&df, "x_ohm_per_km")?;
        let c_nf = Self::get_float_vec(&df, "c_nf_per_km")?;
        let g_us = Self::get_float_vec(&df, "g_us_per_km")?;
        let in_service = Self::get_bool_vec(&df, "in_service")?;
        let parallel = Self::get_int32_vec(&df, "parallel").unwrap_or_else(|_| vec![1; from_bus.len()]);
        let names = if df.hasattr("name")? { Self::get_opt_str_vec(py, &df, "name")? } else { vec![None; from_bus.len()] };

        Ok((0..from_bus.len()).map(|i| Line {
            from_bus: from_bus[i], to_bus: to_bus[i], length_km: length_km[i], r_ohm_per_km: r_ohm[i], x_ohm_per_km: x_ohm[i], c_nf_per_km: c_nf[i], g_us_per_km: g_us[i], in_service: in_service[i], parallel: parallel[i], name: names[i].clone(), df: 1.0, max_i_ka: 0.0, max_loading_percent: None, type_: None, std_type: None,
        }).collect())
    }

    fn extract_trafos(&self, py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<Vec<Transformer>> {
        let hv_bus = Self::get_int32_vec(&df, "hv_bus")?;
        let lv_bus = Self::get_int32_vec(&df, "lv_bus")?;
        let sn_mva = Self::get_float_vec(&df, "sn_mva")?;
        let vn_hv = Self::get_float_vec(&df, "vn_hv_kv")?;
        let vn_lv = Self::get_float_vec(&df, "vn_lv_kv")?;
        let vk = Self::get_float_vec(&df, "vk_percent")?;
        let vkr = Self::get_float_vec(&df, "vkr_percent")?;
        let pfe = Self::get_float_vec(&df, "pfe_kw")?;
        let i0 = Self::get_float_vec(&df, "i0_percent")?;
        let shift = Self::get_float_vec(&df, "shift_degree")?;
        let in_service = Self::get_bool_vec(&df, "in_service")?;
        
        let tap_pos = if df.hasattr("tap_pos")? { Self::get_opt_float_vec(py, &df, "tap_pos")? } else { vec![None; hv_bus.len()] };
        let tap_side = if df.hasattr("tap_side")? { Self::get_opt_str_vec(py, &df, "tap_side")? } else { vec![None; hv_bus.len()] };
        let tap_neutral = if df.hasattr("tap_neutral")? { Self::get_opt_float_vec(py, &df, "tap_neutral")? } else { vec![None; hv_bus.len()] };
        let tap_step_percent = if df.hasattr("tap_step_percent")? { Self::get_opt_float_vec(py, &df, "tap_step_percent")? } else { vec![None; hv_bus.len()] };

        Ok((0..hv_bus.len()).map(|i| Transformer {
            hv_bus: hv_bus[i], lv_bus: lv_bus[i], sn_mva: sn_mva[i], vn_hv_kv: vn_hv[i], vn_lv_kv: vn_lv[i], vk_percent: vk[i], vkr_percent: vkr[i], pfe_kw: pfe[i], i0_percent: i0[i], shift_degree: shift[i], in_service: in_service[i], tap_pos: tap_pos[i], tap_side: tap_side[i].clone(), tap_neutral: tap_neutral[i], tap_step_percent: tap_step_percent[i], parallel: 1, df: 1.0, tap_phase_shifter: false, name: None, std_type: None, max_loading_percent: None, tap_max: None, tap_min: None, tap_step_degree: None,
        }).collect())
    }

    fn extract_loads(&self, py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<Vec<Load>> {
        let bus = Self::get_int_vec(&df, "bus")?;
        let p_mw = Self::get_float_vec(&df, "p_mw")?;
        let q_mvar = Self::get_float_vec(&df, "q_mvar")?;
        let in_service = Self::get_bool_vec(&df, "in_service")?;
        let names = if df.hasattr("name")? { Self::get_opt_str_vec(py, &df, "name")? } else { vec![None; bus.len()] };

        Ok((0..bus.len()).map(|i| Load {
            bus: bus[i], p_mw: p_mw[i], q_mvar: q_mvar[i], in_service: in_service[i], name: names[i].clone(), scaling: 1.0, const_i_percent: 0.0, const_z_percent: 0.0, controllable: None, sn_mva: None, type_: None,
        }).collect())
    }

    fn extract_gens(&self, py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<Vec<Gen>> {
        let bus = Self::get_int_vec(&df, "bus")?;
        let p_mw = Self::get_float_vec(&df, "p_mw")?;
        let vm_pu = Self::get_float_vec(&df, "vm_pu")?;
        let in_service = Self::get_bool_vec(&df, "in_service")?;
        let slack = if df.hasattr("slack")? { Self::get_bool_vec(&df, "slack")? } else { vec![false; bus.len()] };
        let names = if df.hasattr("name")? { Self::get_opt_str_vec(py, &df, "name")? } else { vec![None; bus.len()] };

        Ok((0..bus.len()).map(|i| Gen {
            bus: bus[i], p_mw: p_mw[i], vm_pu: vm_pu[i], in_service: in_service[i], slack: slack[i], scaling: 1.0, max_p_mw: 0.0, min_p_mw: 0.0, max_q_mvar: 0.0, min_q_mvar: 0.0, slack_weight: 0.0, controllable: None, name: names[i].clone(), sn_mva: None, type_: None,
        }).collect())
    }

    fn extract_ext_grids(&self, _py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<Vec<ExtGrid>> {
        let bus = Self::get_int_vec(&df, "bus")?;
        let vm_pu = Self::get_float_vec(&df, "vm_pu")?;
        let va_degree = Self::get_float_vec(&df, "va_degree")?;
        let in_service = Self::get_bool_vec(&df, "in_service")?;

        Ok((0..bus.len()).map(|i| ExtGrid {
            bus: bus[i], vm_pu: vm_pu[i], va_degree: va_degree[i], in_service: in_service[i], slack_weight: 1.0, name: None, max_p_mw: None, min_p_mw: None, max_q_mvar: None, min_q_mvar: None,
        }).collect())
    }

    fn extract_shunts(&self, _py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<Vec<Shunt>> {
        let bus = Self::get_int_vec(&df, "bus")?;
        let p_mw = Self::get_float_vec(&df, "p_mw")?;
        let q_mvar = Self::get_float_vec(&df, "q_mvar")?;
        let in_service = Self::get_bool_vec(&df, "in_service")?;
        let vn_kv = if df.hasattr("vn_kv")? { Self::get_float_vec(&df, "vn_kv")? } else { vec![110.0; bus.len()] };

        Ok((0..bus.len()).map(|i| Shunt {
            bus: bus[i], p_mw: p_mw[i], q_mvar: q_mvar[i], in_service: in_service[i], vn_kv: vn_kv[i], step: 1, max_step: 1, name: None,
        }).collect())
    }

    fn extract_sgens(&self, _py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<Vec<SGen>> {
        let bus = Self::get_int_vec(&df, "bus")?;
        let p_mw = Self::get_float_vec(&df, "p_mw")?;
        let q_mvar = Self::get_float_vec(&df, "q_mvar")?;
        let in_service = Self::get_bool_vec(&df, "in_service")?;

        Ok((0..bus.len()).map(|i| SGen {
            bus: bus[i], p_mw: p_mw[i], q_mvar: q_mvar[i], in_service: in_service[i], scaling: 1.0, name: None, type_: None, sn_mva: None, current_source: false, controllable: None,
        }).collect())
    }

    fn extract_switches(&self, py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<Vec<Switch>> {
        let bus = Self::get_int_vec(&df, "bus")?;
        let element = Self::get_int_vec(&df, "element")?;
        let closed = Self::get_bool_vec(&df, "closed")?;
        let et = if df.hasattr("et")? { Self::get_opt_str_vec(py, &df, "et")? } else { vec![None; bus.len()] };

        Ok((0..bus.len()).map(|i| Switch {
            bus: bus[i], element: element[i], closed: closed[i], et: SwitchType::from(et[i].as_deref().unwrap_or("b")), name: None, type_: None, z_ohm: 0.0,
        }).collect())
    }
}

pub trait ToCSV {
    fn save_csv(&self) -> Result<(), &'static str>;
}

impl ToCSV for Network {
    fn save_csv(&self) -> Result<(), &'static str> {
        todo!()
    }
}

impl Default for Network {
    fn default() -> Self {
        Self {
            r#gen: None,
            bus: Vec::new(),
            load: None,
            line: None,
            trafo: None,
            shunt: None,
            ext_grid: None,
            sgen: None,
            switch: None,
            f_hz: 60.0,
            sn_mva: 100.0,
        }
    }
}

/// Loads a pandapower CSV file into a vector of the specified type.
fn load_pandapower_csv<T: for<'de> Deserialize<'de>>(name: &str) -> Option<Vec<T>> {
    let file = read_csv(&name);
    if file.is_err() {
        return None;
    }
    let file = file.unwrap();
    let mut rdr = ReaderBuilder::new().from_reader(file.as_bytes());
    let mut records: Vec<T> = Vec::new();
    let headers = rdr.headers().unwrap().to_owned();
    for (_idx, i) in rdr.records().enumerate() {
        let record = i.unwrap();
        records.push(record.deserialize(Some(&headers)).unwrap());
    }
    Some(records)
}

/// Reads a CSV file and replaces "True"/"False" with "true"/"false".
fn read_csv(name: &str) -> Result<String, std::io::Error> {
    let mut file = File::open(name)?;
    let mut buffer = String::new();
    file.read_to_string(&mut buffer)?;
    let file = buffer.replace("True", "true").replace("False", "false");
    Ok(file)
}

/// Reads a CSV file from the given map and deserializes it into a vector of the specified type.
fn csv_from_map<T: for<'de> Deserialize<'de>>(
    map: &std::collections::HashMap<String, String>,
    key: &str,
) -> Option<Vec<T>> {
    if !map.contains_key(key) {
        return None;
    }
    let s = map
        .get(key)
        .unwrap()
        .replace("True", "true")
        .replace("False", "false");
    let mut rdr = ReaderBuilder::new().from_reader(s.as_bytes());
    let mut records: Vec<T> = Vec::new();
    let headers = rdr.headers().unwrap().to_owned();
    for (_idx, i) in rdr.records().enumerate() {
        let record = i.unwrap();
        records.push(record.deserialize(Some(&headers)).unwrap());
    }
    if records.is_empty() {
        return None;
    }
    Some(records)
}

/// Macro to read network data from a CSV file.
macro_rules! read_csv_network {
    ($net:ident, $map:ident, { $($field:ident: $file:expr),* $(,)? }) => {
        $(
            $net.$field = csv_from_map(&$map, $file);
        )*
    };
}

/// Macro to read network data from a CSV file.
macro_rules! read_csv_network_folder {
    ($net:ident,  { $($field:ident: $file:expr),* $(,)? }) => {
        $(
            $net.$field = load_pandapower_csv($file);
        )*
    };
}

/// Macro to read network data from a json key.
macro_rules! read_json_network {
    ($net:ident, $map:ident, { $($field:ident: $file:expr),* $(,)? }) => {
        $(
            $net.$field = load_pandapower_element_json(&$map, $file);
        )*
    };
}

/// Loads a CSV folder into a Network structure.
pub fn load_csv_folder(folder: &str) -> Network {
    let bus = folder.to_owned() + "/bus.csv";
    let r#gen = folder.to_owned() + "/gen.csv";
    let line = folder.to_owned() + "/line.csv";
    let shunt = folder.to_owned() + "/shunt.csv";
    let trafo = folder.to_owned() + "/trafo.csv";
    let extgrid = folder.to_owned() + "/ext_grid.csv";
    let load = folder.to_owned() + "/load.csv";
    let sgen = folder.to_owned() + "/sgen.csv";
    let switch = folder.to_owned() + "/switch.csv";
    let mut net = Network::default();
    net.bus = load_pandapower_csv(&bus).unwrap();
    read_csv_network_folder!(net,  {
        r#gen: &r#gen,
        line: &line,
        shunt: &shunt,
        trafo: &trafo,
        ext_grid: &extgrid,
        load: &load,
        sgen:&sgen,
        switch: &switch
    });
    net
}

/// Loads a network from a ZIP file containing CSV files.
pub fn load_csv_zip(name: &str) -> Result<Network, std::io::Error> {
    let f = File::open(name)?;
    let mut zip = zip::ZipArchive::new(f)?;
    let mut map = std::collections::HashMap::new();
    for i in 0..zip.len() {
        let mut file = zip.by_index(i).unwrap();

        if file.is_file() {
            let mut s = String::with_capacity(file.size() as usize);
            file.read_to_string(&mut s).unwrap();
            map.insert(file.name().to_owned(), s);
        }
    }

    let mut net = Network::default();
    net.bus = csv_from_map(&map, "bus.csv").unwrap();
    read_csv_network!(net, map, {
        r#gen: "gen.csv",
        line: "line.csv",
        shunt: "shunt.csv",
        trafo: "trafo.csv",
        ext_grid: "ext_grid.csv",
        load: "load.csv",
        sgen:"sgen.csv",
        switch:"switch.csv"
    });
    Ok(net)
}

fn load_json_from_str(file_content: &str) -> Result<Map<String, Value>, std::io::Error> {
    let parsed: Value = serde_json::from_str(&file_content)?;
    let obj: Map<String, Value> = parsed.as_object().unwrap().clone();
    Ok(obj)
}

fn load_json(file_path: &str) -> Result<Map<String, Value>, std::io::Error> {
    let file_content =
        fs::read_to_string(file_path).expect(format!("Error reading file network file").as_str());
    let obj = load_json_from_str(&file_content);
    obj
}

fn load_pandapower_element_json<T: serde::de::DeserializeOwned>(
    object: &Map<String, Value>,
    key: &str,
) -> Option<Vec<T>> {
    let element = object
        .get(key)
        .and_then(|v| v.as_object())
        .and_then(|v| v.get("_object"));
    if element.is_none() {
        return None;
    }
    let mut elements = Vec::new();
    let element = element.unwrap();
    let map = load_json_from_str(element.as_str().unwrap()).unwrap();

    let headers = map
        .get("columns")
        .and_then(|v| v.as_array())
        .unwrap()
        .to_owned();

    let rows = map.get("data").and_then(|v| v.as_array()).unwrap();

    for (index, row) in rows.iter().enumerate() {
        let obj: Map<String, Value> = Map::new();
        let mut obj: Map<String, Value> =
            headers
                .iter()
                .zip(row.as_array().unwrap().iter())
                .fold(obj, |mut acc, (k, v)| {
                    let key = k.as_str().unwrap();
                    let value = v.to_owned();
                    acc.insert(key.to_string(), value);
                    acc
                });

        obj.insert(
            "index".to_string(),
            Value::Number(serde_json::Number::from(index as i64)),
        );

        let elem: T = serde_json::from_value(obj.clone().into()).unwrap();
        elements.push(elem);
    }

    return Some(elements);
}

pub fn load_pandapower_json(file_path: &str) -> Network {
    let map: Map<String, Value> = load_json(file_path).unwrap();
    load_pandapower_json_obj(&map)
}

pub fn load_pandapower_json_obj(json: &Map<String, Value>) -> Network {
    let object: &Map<String, Value> = json.get("_object").and_then(|v| v.as_object()).unwrap();

    let mut net = Network::default();
    net.bus = load_pandapower_element_json(object, "bus").unwrap();
    read_json_network!(net, object, {
        r#gen: "gen",
        line: "line",
        shunt: "shunt",
        trafo: "trafo",
        ext_grid: "ext_grid",
        load: "load",
        sgen:"sgen",
        switch:"switch"
    });

    return net;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_load_json() -> () {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases", dir);
        let filepath: String = folder.to_owned() + "/networks.json";
        let net = load_pandapower_json(&filepath);
        net.r#gen.unwrap();
    }

    #[test]
    fn test_load_csv() -> () {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/bus.csv";
        let file = read_csv(&name).unwrap();
        let mut rdr = ReaderBuilder::new().from_reader(file.as_bytes());
        for result in rdr.deserialize() {
            let record: Bus = result.unwrap();
            println!("{:?}", record);
        }
    }

    #[test]
    fn load_csv_all() -> () {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let mut net = load_csv_folder(&folder);
        net.f_hz = 60.0;
        net.sn_mva = 100.0;
    }
    #[test]
    fn test_load_csv_zip() -> () {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        let _net = load_csv_zip(&name).unwrap();
    }
}
