use bevy_app::App;
use bevy_ecs::{prelude::*, system::RunSystemOnce};

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use tabled::{Table, settings::Style};

mod res_display;
use res_display::*;

use super::{elements::*, network::*, powerflow::prelude::*};

/// Component storing the result of SBus power flow calculation.
#[derive(Debug, Component, Clone, Serialize, Deserialize)]
pub struct SBusResult(pub Complex64);

/// Component storing the result of VBus power flow calculation.
#[derive(Debug, Component, Clone, Serialize, Deserialize)]
pub struct VBusResult(pub Complex64);

/// Data structure for storing results of power flow calculations for a line.
#[repr(C)]
#[derive(Component, Debug, Default, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub struct LineResultData {
    pub p_from_mw: f64,       // Active power from the 'from' bus (MW)
    pub q_from_mvar: f64,     // Reactive power from the 'from' bus (MVAr)
    pub p_to_mw: f64,         // Active power to the 'to' bus (MW)
    pub q_to_mvar: f64,       // Reactive power to the 'to' bus (MVAr)
    pub pl_mw: f64,           // Line active power loss (MW)
    pub ql_mvar: f64,         // Line reactive power loss (MVAr)
    pub i_from_ka: f64,       // Current from the 'from' bus (kA)
    pub i_to_ka: f64,         // Current to the 'to' bus (kA)
    pub i_ka: f64,            // Line current (kA)
    pub vm_from_pu: f64,      // Voltage magnitude at the 'from' bus (p.u.)
    pub va_from_degree: f64,  // Voltage angle at the 'from' bus (degrees)
    pub vm_to_pu: f64,        // Voltage magnitude at the 'to' bus (p.u.)
    pub va_to_degree: f64,    // Voltage angle at the 'to' bus (degrees)
    pub loading_percent: f64, // Line loading percentage (%)
}

impl LineResultData {
    #[inline(always)]
    pub fn as_slice(&self) -> &[f64; 14] {
        unsafe { &*(self as *const Self as *const [f64; 14]) }
    }
}

impl From<&LineResultData> for LineResTable {
    fn from(val: &LineResultData) -> Self {
        LineResTable {
            from: 0,
            to: 0,
            p_from_mw: FloatWrapper::new(val.p_from_mw, 3),
            q_from_mvar: FloatWrapper::new(val.q_from_mvar, 3),
            p_to_mw: FloatWrapper::new(val.p_to_mw, 3),
            q_to_mvar: FloatWrapper::new(val.q_to_mvar, 3),
            pl_mw: FloatWrapper::new(val.pl_mw, 3),
            ql_mvar: FloatWrapper::new(val.ql_mvar, 3),
            i_from_ka: FloatWrapper::new(val.i_from_ka, 3),
            i_to_ka: FloatWrapper::new(val.i_to_ka, 3),
            i_ka: FloatWrapper::new(val.i_ka, 3),
            vm_from_pu: FloatWrapper::new(val.vm_from_pu, 2),
            va_from_degree: FloatWrapper::new(val.va_from_degree, 2),
            vm_to_pu: FloatWrapper::new(val.vm_to_pu, 2),
            va_to_degree: FloatWrapper::new(val.va_to_degree, 2),
            loading_percent: FloatWrapper::new(val.loading_percent, 1),
        }
    }
}

/// Data structure for storing results of power flow calculations for a transformer.
#[repr(C)]
#[derive(Component, Debug, Default, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub struct TrafoResultData {
    pub p_hv_mw: f64,
    pub q_hv_mvar: f64,
    pub p_lv_mw: f64,
    pub q_lv_mvar: f64,
    pub pl_mw: f64,
    pub ql_mvar: f64,
    pub i_hv_ka: f64,
    pub i_lv_ka: f64,
    pub vm_hv_pu: f64,
    pub va_hv_degree: f64,
    pub vm_lv_pu: f64,
    pub va_lv_degree: f64,
    pub loading_percent: f64,
}

impl TrafoResultData {
    #[inline(always)]
    pub fn as_slice(&self) -> &[f64; 13] {
        unsafe { &*(self as *const Self as *const [f64; 13]) }
    }
}

/// Extracts bus results after power flow calculation.
fn extract_res_bus(
    mut cmd: Commands,
    nodes: Res<NodeLookup>,
    mat: Res<PowerFlowMat>,
    res: Res<PowerFlowResult>,
    common: Res<PFCommonData>,
    cache: Option<Res<crate::basic::newtonpf::NewtonCache>>,
) {
    let n = res.v.len();
    let s_base = common.sbase;
    let mut s_orig = vec![Complex64::new(0.0, 0.0); n];

    if let Some(c) = cache.as_ref().filter(|c| c.s_calc.len() == n) {
        for (new_idx, &orig_idx) in mat.from_perm.iter().enumerate() {
            s_orig[orig_idx] = -c.s_calc[new_idx] * s_base;
        }
    } else {
        // Fallback when NewtonCache is empty or not enabled:
        // mat.y_bus is in permuted order (new_idx).
        // Since res.v is in natural order (orig_idx), map to permuted order:
        let mut v_perm = vec![Complex64::new(0.0, 0.0); n];
        for (new_idx, &orig_idx) in mat.from_perm.iter().enumerate() {
            v_perm[new_idx] = res.v[orig_idx];
        }
        let mut ibus = vec![Complex64::new(0.0, 0.0); n];
        let mut scalc_perm = vec![Complex64::new(0.0, 0.0); n];
        crate::basic::newtonpf::csc_matvec_and_scalc(
            mat.y_bus.col_offsets(),
            mat.y_bus.row_indices(),
            mat.y_bus.values(),
            &v_perm,
            &mut ibus,
            &mut scalc_perm,
        );
        for (new_idx, &orig_idx) in mat.from_perm.iter().enumerate() {
            s_orig[orig_idx] = -scalc_perm[new_idx] * s_base;
        }
    }

    for (idx, entity) in nodes.iter() {
        let i = idx as usize;
        cmd.entity(entity).insert((
            SBusResult(s_orig[i]),
            VBusResult(res.v[i]),
        ));
    }
}

/// Extracts line and transformer results using contiguous_iter_mut + bypass_change_detection.
/// Completely skips OutOfService components with zero overhead.
fn extract_res_branches(
    mut lines_q: Query<(
        &Port4MatPatch,
        &FromBus,
        &ToBus,
        &LineParams,
        &mut LineResultData,
    ), Without<OutOfService>>,
    mut trafos_q: Query<(
        &Port4MatPatch,
        &FromBus,
        &ToBus,
        &TransformerDevice,
        &mut TrafoResultData,
    ), Without<OutOfService>>,
    buses: Query<&VNominal>,
    lut: Res<NodeLookup>,
    results: Res<PowerFlowResult>,
    common: Res<PFCommonData>,
) {
    let s_base = common.sbase;
    let nodes = lut.len();

    // Cache bus nominal voltage array indexed by bus id (3NF compliant: single source of truth)
    let mut bus_vn = vec![1.0; nodes];
    for (bus_idx, entity) in lut.iter() {
        if let Ok(vnom) = buses.get(entity) {
            bus_vn[bus_idx as usize] = vnom.0.0;
        }
    }

    // Direct voltage slice without reorder transposition
    let v_slice = results.v.as_slice();

    const SQRT3: f64 = 1.7320508075688772;

    // 1. Process In-service Lines using contiguous slices with bypass_change_detection
    if let Ok(chunks) = lines_q.contiguous_iter_mut() {
        for (patch_slice, from_slice, to_slice, params_slice, mut res_slice) in chunks {
            let res = res_slice.bypass_change_detection();
            let len = res.len();

            for i in 0..len {
                let f_idx = from_slice[i].0 as usize;
                let t_idx = to_slice[i].0 as usize;
                let vf = v_slice[f_idx];
                let vt = v_slice[t_idx];

                let vn_kv = bus_vn[f_idx];
                let scale = (vn_kv * vn_kv) / s_base;

                // 2x2 physical admittance matrix * [vf, vt]
                let g = patch_slice[i].0;
                let if_pu = (g[(0, 0)] * vf + g[(0, 1)] * vt) * scale;
                let it_pu = (g[(1, 0)] * vf + g[(1, 1)] * vt) * scale;

                let sf = vf * if_pu.conj() * s_base;
                let st = vt * it_pu.conj() * s_base;

                let base_i_ka = s_base / (SQRT3 * vn_kv);
                let i_from_ka = if_pu.norm() * base_i_ka;
                let i_to_ka = it_pu.norm() * base_i_ka;
                let i_ka = i_from_ka.max(i_to_ka);

                let max_i = params_slice[i].max_i_ka;
                let loading_percent = if max_i > 0.0 {
                    (i_ka / max_i) * 100.0
                } else {
                    0.0
                };

                let r = &mut res[i];
                r.p_from_mw = sf.re;
                r.q_from_mvar = sf.im;
                r.p_to_mw = st.re;
                r.q_to_mvar = st.im;
                r.pl_mw = sf.re + st.re;
                r.ql_mvar = sf.im + st.im;
                r.i_from_ka = i_from_ka;
                r.i_to_ka = i_to_ka;
                r.i_ka = i_ka;
                r.vm_from_pu = vf.norm();
                r.va_from_degree = vf.arg().to_degrees();
                r.vm_to_pu = vt.norm();
                r.va_to_degree = vt.arg().to_degrees();
                r.loading_percent = loading_percent;
            }
        }
    } else {
        // Fallback for non-contiguous iteration if any
        for (patch, from, to, params, mut res) in lines_q.iter_mut() {
            let r = res.bypass_change_detection();
            let f_idx = from.0 as usize;
            let t_idx = to.0 as usize;
            let vf = v_slice[f_idx];
            let vt = v_slice[t_idx];
            let vn_kv = bus_vn[f_idx];
            let scale = (vn_kv * vn_kv) / s_base;

            let g = patch.0;
            let if_pu = (g[(0, 0)] * vf + g[(0, 1)] * vt) * scale;
            let it_pu = (g[(1, 0)] * vf + g[(1, 1)] * vt) * scale;

            let sf = vf * if_pu.conj() * s_base;
            let st = vt * it_pu.conj() * s_base;

            let base_i_ka = s_base / (SQRT3 * vn_kv);
            let i_from_ka = if_pu.norm() * base_i_ka;
            let i_to_ka = it_pu.norm() * base_i_ka;
            let i_ka = i_from_ka.max(i_to_ka);

            let max_i = params.max_i_ka;
            let loading_percent = if max_i > 0.0 {
                (i_ka / max_i) * 100.0
            } else {
                0.0
            };

            r.p_from_mw = sf.re;
            r.q_from_mvar = sf.im;
            r.p_to_mw = st.re;
            r.q_to_mvar = st.im;
            r.pl_mw = sf.re + st.re;
            r.ql_mvar = sf.im + st.im;
            r.i_from_ka = i_from_ka;
            r.i_to_ka = i_to_ka;
            r.i_ka = i_ka;
            r.vm_from_pu = vf.norm();
            r.va_from_degree = vf.arg().to_degrees();
            r.vm_to_pu = vt.norm();
            r.va_to_degree = vt.arg().to_degrees();
            r.loading_percent = loading_percent;
        }
    }

    // 2. Process In-service Transformers using contiguous slices with bypass_change_detection
    if let Ok(chunks) = trafos_q.contiguous_iter_mut() {
        for (patch_slice, from_slice, to_slice, dev_slice, mut res_slice) in chunks {
            let res = res_slice.bypass_change_detection();
            let len = res.len();

            for i in 0..len {
                let hv_idx = from_slice[i].0 as usize;
                let lv_idx = to_slice[i].0 as usize;
                let vh = v_slice[hv_idx];
                let vl = v_slice[lv_idx];

                let dev = &dev_slice[i];
                let vn_lv = dev.vn_lv_kv;
                let vn_hv = dev.vn_hv_kv;
                let scale = (vn_lv * vn_lv) / s_base;

                // 2x2 physical admittance matrix * [vh, vl]
                let g = patch_slice[i].0;
                let ih_pu = (g[(0, 0)] * vh + g[(0, 1)] * vl) * scale;
                let il_pu = (g[(1, 0)] * vh + g[(1, 1)] * vl) * scale;

                let sh = vh * ih_pu.conj() * s_base;
                let sl = vl * il_pu.conj() * s_base;

                let base_i_hv = s_base / (SQRT3 * vn_hv);
                let base_i_lv = s_base / (SQRT3 * vn_lv);
                let i_hv_ka = ih_pu.norm() * base_i_hv;
                let i_lv_ka = il_pu.norm() * base_i_lv;

                let sn_rated = dev.sn_mva * (dev.parallel as f64);
                let i_rated_hv = sn_rated / (SQRT3 * vn_hv);
                let i_rated_lv = sn_rated / (SQRT3 * vn_lv);
                let loading_percent = if sn_rated > 0.0 && i_rated_hv > 0.0 && i_rated_lv > 0.0 {
                    (i_hv_ka / i_rated_hv).max(i_lv_ka / i_rated_lv) * 100.0
                } else {
                    0.0
                };

                let r = &mut res[i];
                r.p_hv_mw = sh.re;
                r.q_hv_mvar = sh.im;
                r.p_lv_mw = sl.re;
                r.q_lv_mvar = sl.im;
                r.pl_mw = sh.re + sl.re;
                r.ql_mvar = sh.im + sl.im;
                r.i_hv_ka = i_hv_ka;
                r.i_lv_ka = i_lv_ka;
                r.vm_hv_pu = vh.norm();
                r.va_hv_degree = vh.arg().to_degrees();
                r.vm_lv_pu = vl.norm();
                r.va_lv_degree = vl.arg().to_degrees();
                r.loading_percent = loading_percent;
            }
        }
    } else {
        // Fallback for non-contiguous iteration if any
        for (patch, from, to, dev, mut res) in trafos_q.iter_mut() {
            let r = res.bypass_change_detection();
            let hv_idx = from.0 as usize;
            let lv_idx = to.0 as usize;
            let vh = v_slice[hv_idx];
            let vl = v_slice[lv_idx];

            let vn_lv = dev.vn_lv_kv;
            let vn_hv = dev.vn_hv_kv;
            let scale = (vn_lv * vn_lv) / s_base;

            let g = patch.0;
            let ih_pu = (g[(0, 0)] * vh + g[(0, 1)] * vl) * scale;
            let il_pu = (g[(1, 0)] * vh + g[(1, 1)] * vl) * scale;

            let sh = vh * ih_pu.conj() * s_base;
            let sl = vl * il_pu.conj() * s_base;

            let base_i_hv = s_base / (SQRT3 * vn_hv);
            let base_i_lv = s_base / (SQRT3 * vn_lv);
            let i_hv_ka = ih_pu.norm() * base_i_hv;
            let i_lv_ka = il_pu.norm() * base_i_lv;

            let sn_rated = dev.sn_mva * (dev.parallel as f64);
            let i_rated_hv = sn_rated / (SQRT3 * vn_hv);
            let i_rated_lv = sn_rated / (SQRT3 * vn_lv);
            let loading_percent = if sn_rated > 0.0 && i_rated_hv > 0.0 && i_rated_lv > 0.0 {
                (i_hv_ka / i_rated_hv).max(i_lv_ka / i_rated_lv) * 100.0
            } else {
                0.0
            };

            r.p_hv_mw = sh.re;
            r.q_hv_mvar = sh.im;
            r.p_lv_mw = sl.re;
            r.q_lv_mvar = sl.im;
            r.pl_mw = sh.re + sl.re;
            r.ql_mvar = sh.im + sl.im;
            r.i_hv_ka = i_hv_ka;
            r.i_lv_ka = i_lv_ka;
            r.vm_hv_pu = vh.norm();
            r.va_hv_degree = vh.arg().to_degrees();
            r.vm_lv_pu = vl.norm();
            r.va_lv_degree = vl.arg().to_degrees();
            r.loading_percent = loading_percent;
        }
    }
}

/// Prints the results of the power flow for each bus.
fn print_res_bus(q: Query<(&BusID, &VBusResult, &SBusResult)>) {
    let bus_res_table = q
        .iter()
        .sort_by::<&BusID>(|value_1, value_2| value_1.cmp(value_2))
        .map(|(node, v, s)| {
            let vm = v.0.norm();
            let angle = v.0.arg().to_degrees();
            let p = s.0.re;
            let q = s.0.im;
            BusResTable {
                Bus: node.0 as i32,
                Vm: FloatWrapper::new(vm, 5),
                Va: FloatWrapper::new(angle, 5),
                P_mw: FloatWrapper::new(p, 5),
                Q_mvar: FloatWrapper::new(q, 5),
            }
        });
    let table = Table::new(bus_res_table)
        .with(Style::markdown())
        .to_string();
    println!("{table}");
}

/// Prints the results of the power flow for each line.
fn print_res_line(q: Query<(&FromBus, &ToBus, &LineResultData)>) {
    let table = q.iter().map(|(from, to, record)| {
        let mut row_display: LineResTable = record.into();
        row_display.from = from.0;
        row_display.to = to.0;
        row_display
    });

    let table = Table::new(table).with(Style::markdown()).to_string();
    println!("{table}");
}

/// Trait for post-processing after a power flow simulation.
pub trait PostProcessing {
    fn post_process(&mut self);
    fn print_res_bus(&mut self);
    fn print_res_line(&mut self);
}

impl PostProcessing for PowerGrid {
    fn print_res_bus(&mut self) {
        self.world_mut().run_system_once(print_res_bus).unwrap();
    }

    fn print_res_line(&mut self) {
        self.world_mut().run_system_once(print_res_line).unwrap();
    }

    fn post_process(&mut self) {
        self.world_mut().run_system_once(extract_res_bus).unwrap();
        self.world_mut().run_system_once(extract_res_branches).unwrap();
    }
}

impl PostProcessing for App {
    fn print_res_bus(&mut self) {
        self.world_mut().run_system_once(print_res_bus).unwrap();
    }

    fn print_res_line(&mut self) {
        self.world_mut().run_system_once(print_res_line).unwrap();
    }

    fn post_process(&mut self) {
        self.world_mut().run_system_once(extract_res_bus).unwrap();
        self.world_mut().run_system_once(extract_res_branches).unwrap();
    }
}
