use serde::Deserializer;
use serde::{Deserialize, Serialize};
use std::option::Option;

//This module is used to parse pandapower network parameters

fn from_number<'de, D>(deserializer: D) -> Result<i64, D::Error>
where
    D: Deserializer<'de>,
{
    let val: serde_json::Value = Deserialize::deserialize(deserializer)?;
    if let serde_json::Value::Number(n) = val {
        let res = n.as_f64().unwrap();
        return Ok(res as i64);
    }
    Err(serde::de::Error::custom("invalid number format"))
}

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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Bus {
    pub index: i64,
    pub in_service: bool,
    pub max_vm_pu: f64,
    pub min_vm_pu: f64,
    #[serde(deserialize_with = "from_str")]
    pub name: Option<String>,
    #[serde(rename = "type")]
    pub type_: Option<String>, // Added underscore to avoid conflict with Rust keyword
    pub vn_kv: f64,
    #[serde(deserialize_with = "from_number")]
    pub zone: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Gen {
    pub bus: i64,
    pub controllable: bool,
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

#[derive(Debug, Serialize, Deserialize)]
pub struct Load {
    pub bus: i64,
    pub const_i_percent: f64,
    pub const_z_percent: f64,
    pub controllable: bool,
    pub in_service: bool,
    pub name: Option<String>,
    pub p_mw: f64,
    pub q_mvar: f64,
    pub scaling: f64,
    pub sn_mva: Option<f64>,
    #[serde(rename = "type")]
    pub type_: Option<String>, // Added underscore to avoid conflict with Rust keyword
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Line {
    pub c_nf_per_km: f64,
    pub df: f64,
    pub from_bus: i64,
    pub to_bus: i64,
    pub g_us_per_km: f64,
    pub in_service: bool,
    pub length_km: f64,
    pub max_i_ka: f64,
    pub max_loading_percent: f64,
    pub parallel: i32,
    pub r_ohm_per_km: f64,
    #[serde(rename = "type")]
    pub type_: String,
    pub x_ohm_per_km: f64,
    pub name: Option<String>,
    pub std_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Transformer {
    pub df: f64,
    pub hv_bus: i32,
    pub i0_percent: f64,
    pub in_service: bool,
    pub lv_bus: i32,
    pub max_loading_percent: f64,
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

#[derive(Debug, Serialize, Deserialize)]
pub struct ExtGrid {
    pub bus: i64,
    pub in_service: bool,
    pub va_degree: f64,
    pub vm_pu: f64,
    pub max_p_mw: f64,
    pub min_p_mw: f64,
    pub max_q_mvar: f64,
    pub min_q_mvar: f64,
    pub slack_weight: f64,
    pub name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
pub struct Network {
    pub gen: Option<Vec<Gen>>,
    pub bus: Vec<Bus>,
    pub load: Option<Vec<Load>>,
    pub line: Option<Vec<Line>>,
    pub trafo: Option<Vec<Transformer>>,
    pub shunt: Option<Vec<Shunt>>,
    pub ext_grid: Option<Vec<ExtGrid>>,
    pub f_hz: f64,
    pub sn_mva: f64,
}
