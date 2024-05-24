use csv::ReaderBuilder;
use nalgebra::{vector, Complex};
use serde::Deserializer;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fs::File;
use std::{io::Read, option::Option};

use crate::basic::system::*;
use crate::prelude::admittance::*;

/// This module is used to parse pandapower network parameters

/// Deserializes a number from JSON format.
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

/// Represents a generator in the network.
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

/// Represents a load in the network.
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

/// Represents a line in the network.
#[derive(Default, Debug, Serialize, Deserialize)]
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

/// Represents a transformer in the network.
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

/// Represents an external grid in the network.
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

/// Represents the data from the sgen.
#[derive(Default, Debug, Serialize, Deserialize)]
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
    pub controllable: bool,
}

/// Represents a shunt in the network.
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

/// Represents a network.
#[derive(Debug, Serialize, Deserialize)]
pub struct Network {
    pub gen: Option<Vec<Gen>>,
    pub bus: Vec<Bus>,
    pub load: Option<Vec<Load>>,
    pub line: Option<Vec<Line>>,
    pub trafo: Option<Vec<Transformer>>,
    pub shunt: Option<Vec<Shunt>>,
    pub ext_grid: Option<Vec<ExtGrid>>,
    pub sgen: Option<Vec<SGen>>,
    pub f_hz: f64,
    pub sn_mva: f64,
}

/// Trait for saving a network to CSV files.
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
            gen: None,
            bus: Vec::new(),
            load: None,
            line: None,
            trafo: None,
            shunt: None,
            ext_grid: None,
            sgen: None,
            f_hz: 60.0,
            sn_mva: 100.0,
        }
    }
}

/// Loads a pandapower CSV file into a vector of the specified type.
fn load_pandapower_csv<T: for<'de> Deserialize<'de>>(name: String) -> Vec<T> {
    let file = read_csv(&name).unwrap();
    let mut rdr = ReaderBuilder::new().from_reader(file.as_bytes());
    let mut records: Vec<T> = Vec::new();
    let headers = rdr.headers().unwrap().to_owned();
    for (_idx, i) in rdr.records().enumerate() {
        let record = i.unwrap();
        records.push(record.deserialize(Some(&headers)).unwrap());
    }
    records
}

/// Reads a CSV file and replaces "True"/"False" with "true"/"false".
fn read_csv(name: &str) -> Result<String, std::io::Error> {
    let mut file = File::open(name)?;
    let mut buffer = String::new();
    file.read_to_string(&mut buffer)?;
    let file = buffer.replace("True", "true").replace("False", "false");
    Ok(file)
}

/// Loads a CSV folder into a Network structure.
pub fn load_csv_folder(folder: String) -> Network {
    let bus = folder.to_owned() + "/bus.csv";
    let gen = folder.to_owned() + "/gen.csv";
    let line = folder.to_owned() + "/line.csv";
    let shunt = folder.to_owned() + "/shunt.csv";
    let trafo = folder.to_owned() + "/trafo.csv";
    let extgrid = folder.to_owned() + "/ext_grid.csv";
    let load = folder.to_owned() + "/load.csv";
    let sgen = folder.to_owned() + "/sgen.csv";
    let mut net = Network::default();
    net.bus = load_pandapower_csv(bus);
    net.gen = Some(load_pandapower_csv(gen));
    net.line = Some(load_pandapower_csv(line));
    net.shunt = Some(load_pandapower_csv(shunt));
    net.trafo = Some(load_pandapower_csv(trafo));
    net.ext_grid = Some(load_pandapower_csv(extgrid));
    net.load = Some(load_pandapower_csv(load));
    net.sgen = Some(load_pandapower_csv(sgen));
    net
}

/// Converts a line to its equivalent admittance branches.
fn line_to_admit(wbase: f64, bus: &[Bus], line: &Line) -> Vec<AdmittanceBranch> {
    let mut out = Vec::new();
    let (mut shunt_f, mut shunt_t) = (AdmittanceBranch::default(), AdmittanceBranch::default());
    let b = wbase * 1e-9 * line.c_nf_per_km * line.length_km * (line.parallel as f64);
    let g = line.g_us_per_km * line.length_km * 1e-6 * (line.parallel as f64);
    let v_base = bus[line.from_bus as usize].vn_kv;
    let a = Admittance(0.5 * Complex { re: g, im: b });
    if line.g_us_per_km != 0.0 || line.c_nf_per_km != 0.0 {
        shunt_f.y = a.clone();
        shunt_f.v_base = v_base;
        shunt_t.y = a;
        shunt_t.v_base = v_base;
        shunt_f.port = Port2(vector![line.from_bus as i32, GND]);
        shunt_t.port = Port2(vector![line.to_bus as i32, GND]);
        out.push(shunt_f);
        out.push(shunt_t);
    }

    let rl = line.r_ohm_per_km * line.length_km * (line.parallel as f64);
    let xl = line.x_ohm_per_km * line.length_km * (line.parallel as f64);
    let l = AdmittanceBranch {
        y: Admittance(1.0 / Complex { re: rl, im: xl }),
        port: Port2(vector![line.from_bus as i32, line.to_bus as i32]),
        v_base,
    };
    out.push(l);
    out
}

/// Converts a load to its equivalent PQ nodes.
fn load_to_pqnode(item: &Load) -> [PQNode; 1] {
    let s = Complex::new(item.p_mw, item.q_mvar);
    let bus = item.bus;
    [PQNode { s, bus }]
}

/// Converts a generator to its equivalent PV nodes.
fn gen_to_pvnode(item: &Gen) -> [PVNode; 1] {
    let p = item.p_mw;
    let v = item.vm_pu;
    let bus = item.bus;
    [PVNode { p, v, bus }]
}

/// Converts an external grid to its equivalent external grid node.
fn extgrid_to_extnode(item: &ExtGrid) -> [ExtGridNode; 1] {
    let bus = item.bus;
    let v = item.vm_pu;
    let phase = item.va_degree.to_radians();

    [ExtGridNode { v, phase, bus }]
}

/// Converts a shunt to its equivalent PQ nodes.
fn shunt_to_pqnode(item: &Shunt) -> [PQNode; 1] {
    let s = Complex::new(item.p_mw, item.q_mvar);
    let bus = item.bus;
    [PQNode { s, bus }]
}

/// Converts a shunt to its equivalent PQ nodes.
fn sgen_to_pqnode(item: &SGen) -> [PQNode; 1] {
    let s = Complex::new(-item.p_mw, -item.q_mvar);
    let bus = item.bus;
    [PQNode { s, bus }]
}

/// Converts a transformer to its equivalent admittance branches.
fn trafo_to_admit(item: &Transformer) -> Vec<AdmittanceBranch> {
    let v_base = item.vn_lv_kv;
    let vkr = item.vkr_percent * 0.01;
    let vk = item.vk_percent * 0.01;

    let tap_m = 1.0
        + (item.tap_pos.unwrap_or(0.0) - item.tap_neutral.unwrap_or(0.0))
            * 0.01
            * item.tap_step_percent.unwrap_or(0.0);
    let zbase = v_base * v_base / item.sn_mva;
    let z = zbase * vk;
    let parallel = item.parallel;

    let re = zbase * vkr;
    let im = (z.powi(2) - re.powi(2)).sqrt();
    let port = Port2(vector![item.hv_bus, item.lv_bus]);
    let y = 1.0 / (Complex { re, im } * parallel as f64);
    let sc = AdmittanceBranch {
        y: Admittance(y / tap_m),
        port,
        v_base,
    };
    let mut v = Vec::new();
    v.push(sc);
    v.push(AdmittanceBranch {
        y: Admittance((1.0 - tap_m) * y / tap_m.powi(2)),
        port: Port2(vector![item.hv_bus, GND]),
        v_base,
    });
    v.push(AdmittanceBranch {
        y: Admittance((1.0 - 1.0 / tap_m) * y),
        port: Port2(vector![item.lv_bus, GND]),
        v_base,
    });
    let re = zbase * (0.001 * item.pfe_kw) / item.sn_mva;
    let im = zbase / (0.01 * item.i0_percent);
    let c = parallel as f64 / Complex { re, im };

    if c.is_nan() {
        return v;
    }
    let port = Port2(vector![item.hv_bus, GND]);
    let y = Admittance(0.5 * c / tap_m.powi(2));
    let shunt = AdmittanceBranch { y, port, v_base };
    v.push(shunt);
    let port = Port2(vector![item.lv_bus, GND]);
    let y = Admittance(0.5 * c);
    let shunt = AdmittanceBranch { y, port, v_base };
    v.push(shunt);
    v
}

/// Collects PQ nodes from the given items using the provided converter function.
#[inline(always)]
fn collect_pq_nodes<T>(items: Option<Vec<T>>, converter: fn(&T) -> [PQNode; 1]) -> Vec<PQNode> {
    items
        .unwrap_or_else(Vec::new)
        .iter()
        .flat_map(converter)
        .collect()
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
    for (idx, i) in rdr.records().enumerate() {
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

/// Loads a network from a ZIP file containing CSV files.
pub fn load_csv_zip(name: String) -> Result<Network, std::io::Error> {
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
        gen: "gen.csv",
        line: "line.csv",
        shunt: "shunt.csv",
        trafo: "trafo.csv",
        ext_grid: "ext_grid.csv",
        load: "load.csv",
        sgen:"sgen.csv"
    });
    Ok(net)
}

impl From<Network> for PFNetwork {
    fn from(value: Network) -> Self {
        let v_base = value.bus[value.ext_grid.as_ref().unwrap()[0].bus as usize].vn_kv;
        let s_base = value.sn_mva;
        let wbase = value.f_hz * 2.0 * PI;
        let binding = value.line.unwrap_or(Vec::new());
        let bus = &value.bus;
        let a = binding
            .iter()
            .flat_map(|x| line_to_admit(wbase, bus, x).into_iter());

        let binding = value.trafo.unwrap_or(Vec::new());
        let b = binding.iter().flat_map(|x| trafo_to_admit(x).into_iter());
        let y_br = a.chain(b).collect();

        let ext = extgrid_to_extnode(&value.ext_grid.unwrap_or(Vec::new())[0])[0];
        let pq_loads = collect_pq_nodes(value.load, load_to_pqnode)
            .into_iter()
            .chain(collect_pq_nodes(value.shunt, shunt_to_pqnode))
            .chain(collect_pq_nodes(value.sgen, sgen_to_pqnode))
            .collect();

        let pv_nodes = value
            .gen
            .unwrap_or(Vec::new())
            .iter()
            .map(|x| gen_to_pvnode(x).into_iter())
            .flatten()
            .collect();
        Self {
            v_base,
            s_base,
            pq_loads,
            pv_nodes,
            ext,
            y_br,
            buses: value.bus,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
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
        let mut net = load_csv_folder(folder);
        net.f_hz = 60.0;
        net.sn_mva = 100.0;
    }
    #[test]
    fn test_load_csv_zip() -> () {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        load_csv_zip(name).unwrap();
    }
}
