use std::f64::consts::PI;

use crate::basic::system::*;
use crate::prelude::pandapower::*;
use nalgebra::vector;
use nalgebra::Complex;
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

/// Converts a shunt to its equivalent admittance.
fn shunt_to_admit(item: &Shunt) -> [AdmittanceBranch; 1] {
    let s = Complex::new(-item.p_mw, -item.q_mvar) * Complex::new(item.step as f64, 0.0);
    let y = s / (item.vn_kv * item.vn_kv);
    [AdmittanceBranch {
        y: Admittance(y),
        port: Port2(vector![item.bus as i32, GND.into()]),
        v_base: item.vn_kv,
    }]
}
/// Converts a static generator to its equivalent PQ nodes.
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
    let y = Admittance(c / tap_m.powi(2));
    let shunt = AdmittanceBranch { y, port, v_base };
    v.push(shunt);
    // let port = Port2(vector![item.lv_bus, GND]);
    // let y = Admittance(0.5 * c);
    // let shunt = AdmittanceBranch { y, port, v_base };
    // v.push(shunt);
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
        let binding = value.shunt.unwrap_or(Vec::new());
        let shunts = binding.iter().flat_map(|x| shunt_to_admit(x).into_iter());
        let y_br = a.chain(b).chain(shunts).collect();

        let ext = extgrid_to_extnode(&value.ext_grid.unwrap_or(Vec::new())[0])[0];
        let pq_loads = collect_pq_nodes(value.load, load_to_pqnode)
            .into_iter()
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
