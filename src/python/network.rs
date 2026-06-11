use pyo3::prelude::*;
use crate::io::pandapower::{Network, Bus, Gen, Load, Line, Transformer, ExtGrid, Shunt, SGen, Switch, SwitchType};

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
            from_bus: from_bus[i], to_bus: to_bus[i], length_km: length_km[i], r_ohm_per_km: r_ohm[i], x_ohm_per_km: x_ohm[i], c_nf_per_km: c_nf[i], g_us_per_km: g_us[i], in_service: in_service[i], parallel: parallel[i], name: names[i].clone(), df: 1.0, max_i_ka: Some(0.0), max_loading_percent: None, type_: None, std_type: None,
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
        let opt_limit = |col: &str, unlimited: f64| -> PyResult<Vec<f64>> {
            if df.hasattr(col)? {
                Ok(Self::get_opt_float_vec(py, &df, col)?
                    .into_iter()
                    .map(|v| v.unwrap_or(unlimited))
                    .collect())
            } else {
                Ok(vec![unlimited; bus.len()])
            }
        };
        let max_q = opt_limit("max_q_mvar", 1e9)?;
        let min_q = opt_limit("min_q_mvar", -1e9)?;
        let max_p = opt_limit("max_p_mw", 1e9)?;
        let min_p = opt_limit("min_p_mw", -1e9)?;

        Ok((0..bus.len()).map(|i| Gen {
            bus: bus[i], p_mw: p_mw[i], vm_pu: vm_pu[i], in_service: in_service[i], slack: slack[i], scaling: 1.0, max_p_mw: max_p[i], min_p_mw: min_p[i], max_q_mvar: max_q[i], min_q_mvar: min_q[i], slack_weight: 0.0, controllable: None, name: names[i].clone(), sn_mva: None, type_: None,
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
