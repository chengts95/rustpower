use csv::{DeserializeRecordsIntoIter, ReaderBuilder};
use serde::Deserializer;
use serde::{Deserialize, Serialize};
use soa_rs::Soars;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::option::Option;
use std::path::Path;
use std::str::FromStr;

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

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
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

#[derive(Default, Debug, Serialize, Deserialize)]
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

#[derive(Default, Debug, Serialize, Deserialize)]
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

#[derive(Default, Debug, Serialize, Deserialize, Soars)]
#[soa_derive(include(Ref), Serialize)]
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

#[derive(Default, Debug, Serialize, Deserialize)]
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

#[derive(Default, Debug, Serialize, Deserialize)]
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

#[derive(Default, Debug, Serialize, Deserialize)]
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

#[derive(Default, Debug, Serialize, Deserialize)]
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
pub trait ToCSV {
    fn save_csv(&self) -> Result<(), &'static str>;
}

impl ToCSV for Network {
    fn save_csv(&self) -> Result<(), &'static str> {
        todo!()
    }
}

#[test]
fn test_load_csv() -> () {
    use std::io::{BufRead, BufReader};
    // let file_path = "data.zip";
    let folder = "D:/projects/rust/rustpower/out";
    let name = folder.to_owned() + "/bus.csv";
    let file = read_csv(&name).unwrap();
    let mut rdr = ReaderBuilder::new().from_reader(file.as_bytes());
    for result in rdr.deserialize() {
        let record: Bus = result.unwrap();
        println!("{:?}", record);
    }
}

pub fn load_pandapower_csv<T:for<'de> Deserialize<'de>>(name: String) -> Vec<T> {
    let file = read_csv(&name).unwrap();
    let rdr = ReaderBuilder::new().from_reader(file.as_bytes());
     rdr.into_deserialize::<T>().map(|x| x.unwrap()).collect()
}

fn read_csv(name: &str) -> Result<String,std::io::Error> {
    let mut file = File::open(name)?;
    let mut buffer = String::new();
    file.read_to_string(&mut buffer)?;
    let file = buffer.replace("True", "true").replace("False", "false");
    Ok(file)
}

#[test]
fn load_csv_all() -> () {
    use std::io::{BufRead, BufReader};
    // let file_path = "data.zip";
    let folder = "D:/projects/rust/rustpower/out";
    let bus = folder.to_owned() + "/bus.csv";
    let gen = folder.to_owned() + "/gen.csv";
    let line = folder.to_owned() + "/line.csv";
    let shunt = folder.to_owned() + "/shunt.csv";
    let trafo = folder.to_owned() + "/trafo.csv";
    let extgrid = folder.to_owned() + "/extgrid.csv";
    let load = folder.to_owned() + "/load.csv";
    let mut net = Network::default();
    net.bus = load_pandapower_csv(bus);
    net.gen = Some(load_pandapower_csv(gen));
    net.line = Some(load_pandapower_csv(line));
    net.shunt = Some(load_pandapower_csv(shunt));
    net.trafo = Some(load_pandapower_csv(trafo));
    net.ext_grid = Some(load_pandapower_csv(extgrid));
    net.load = Some(load_pandapower_csv(load));
    net.f_hz = 60.0;
    net.sn_mva = 100.0;
    println!("{:?}",net);
}
