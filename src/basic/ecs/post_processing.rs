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
/// Uses in-place mutation via Query when components exist (zero command allocations in hot loop),
/// falling back to Command entity insertion only on initial setup.
fn extract_res_bus(
    mut cmd: Commands,
    mut bus_q: Query<(&BusID, &mut SBusResult, &mut VBusResult)>,
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

    if bus_q.is_empty() {
        for (idx, entity) in nodes.iter() {
            let i = idx as usize;
            if i < n {
                cmd.entity(entity)
                    .insert((SBusResult(s_orig[i]), VBusResult(res.v[i])));
            }
        }
    } else if let Ok(chunks) = bus_q.contiguous_iter_mut() {
        for (id_slice, mut s_slice, mut v_slice) in chunks {
            let s = s_slice.bypass_change_detection();
            let v = v_slice.bypass_change_detection();
            let len = id_slice.len();
            for i in 0..len {
                let idx = id_slice[i].0 as usize;
                if idx < n {
                    s[i].0 = s_orig[idx];
                    v[i].0 = res.v[idx];
                }
            }
        }
    }
}

/// Bus precalculated quantities for post-processing.
/// Compact 48-byte struct (cache line friendly) holding bus voltage components,
/// magnitude, angle in degrees, and power/current base scaling factors.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BusPrecalc {
    pub vr: f64,
    pub vi: f64,
    pub vm: f64,
    pub va_deg: f64,
    pub p_scale: f64, // vn_kv * vn_kv (MW scale)
    pub i_scale: f64, // vn_kv / sqrt(3) (kA scale)
}

/// Two-port branch unified output flow.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TwoPortFlow {
    pub p0_mw: f64,
    pub q0_mvar: f64,
    pub p1_mw: f64,
    pub q1_mvar: f64,
    pub pl_mw: f64,
    pub ql_mvar: f64,
    pub i0_norm: f64, // sqrt(ir0^2 + ii0^2) before i_scale
    pub i1_norm: f64, // sqrt(ir1^2 + ii1^2) before i_scale
}

/// Generic portable two-port branch calculation.
/// Pure mathematical kernel with zero branches and contiguous arithmetic,
/// highly receptive to LLVM SLP auto-vectorization and unrolling.
#[inline(always)]
pub fn compute_two_port_flow_portable(
    g: &[Complex64; 4],
    f: &BusPrecalc,
    t: &BusPrecalc,
    p_scale: f64,
) -> TwoPortFlow {
    let g00 = g[0];
    let g10 = g[1];
    let g01 = g[2];
    let g11 = g[3];

    let i0_r = (g00.re * f.vr - g00.im * f.vi) + (g01.re * t.vr - g01.im * t.vi);
    let i0_i = (g00.re * f.vi + g00.im * f.vr) + (g01.re * t.vi + g01.im * t.vr);
    let i1_r = (g10.re * f.vr - g10.im * f.vi) + (g11.re * t.vr - g11.im * t.vi);
    let i1_i = (g10.re * f.vi + g10.im * f.vr) + (g11.re * t.vi + g11.im * t.vr);

    let p0_mw = (f.vr * i0_r + f.vi * i0_i) * p_scale;
    let q0_mvar = (f.vi * i0_r - f.vr * i0_i) * p_scale;
    let p1_mw = (t.vr * i1_r + t.vi * i1_i) * p_scale;
    let q1_mvar = (t.vi * i1_r - t.vr * i1_i) * p_scale;

    let i0_norm = (i0_r * i0_r + i0_i * i0_i).sqrt();
    let i1_norm = (i1_r * i1_r + i1_i * i1_i).sqrt();

    TwoPortFlow {
        p0_mw,
        q0_mvar,
        p1_mw,
        q1_mvar,
        pl_mw: p0_mw + p1_mw,
        ql_mvar: q0_mvar + q1_mvar,
        i0_norm,
        i1_norm,
    }
}

/// AVX2 + FMA specialized two-port kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn compute_two_port_flow_avx2(
    g_ptr: *const f64,
    f: &BusPrecalc,
    t: &BusPrecalc,
    p_scale: f64,
) -> TwoPortFlow {
    use std::arch::x86_64::*;
    unsafe {
        let neg_mask = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);

        let col0 = _mm256_loadu_pd(g_ptr);
        let col1 = _mm256_loadu_pd(g_ptr.add(4));

        let col0_rot = _mm256_xor_pd(_mm256_shuffle_pd(col0, col0, 0b0101), neg_mask);
        let col1_rot = _mm256_xor_pd(_mm256_shuffle_pd(col1, col1, 0b0101), neg_mask);

        let v_fr = _mm256_set1_pd(f.vr);
        let v_fi = _mm256_set1_pd(f.vi);
        let v_tr = _mm256_set1_pd(t.vr);
        let v_ti = _mm256_set1_pd(t.vi);

        let mut i_vec = _mm256_mul_pd(col0, v_fr);
        i_vec = _mm256_fmadd_pd(col0_rot, v_fi, i_vec);
        i_vec = _mm256_fmadd_pd(col1, v_tr, i_vec);
        i_vec = _mm256_fmadd_pd(col1_rot, v_ti, i_vec);

        let v_re = _mm256_set_pd(-t.vr, t.vr, -f.vr, f.vr);
        let v_im = _mm256_set_pd(t.vi, t.vi, f.vi, f.vi);
        let i_swap = _mm256_shuffle_pd(i_vec, i_vec, 0b0101);

        let mut pq_vec = _mm256_mul_pd(v_re, i_vec);
        pq_vec = _mm256_fmadd_pd(v_im, i_swap, pq_vec);
        pq_vec = _mm256_mul_pd(pq_vec, _mm256_set1_pd(p_scale));

        let mut curr = [0.0f64; 4];
        let mut pq = [0.0f64; 4];
        _mm256_storeu_pd(curr.as_mut_ptr(), i_vec);
        _mm256_storeu_pd(pq.as_mut_ptr(), pq_vec);

        let i0_norm = (curr[0] * curr[0] + curr[1] * curr[1]).sqrt();
        let i1_norm = (curr[2] * curr[2] + curr[3] * curr[3]).sqrt();

        TwoPortFlow {
            p0_mw: pq[0],
            q0_mvar: pq[1],
            p1_mw: pq[2],
            q1_mvar: pq[3],
            pl_mw: pq[0] + pq[2],
            ql_mvar: pq[1] + pq[3],
            i0_norm,
            i1_norm,
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn process_lines_avx2(
    patch_slice: &[Port4MatPatch],
    from_slice: &[FromBus],
    to_slice: &[ToBus],
    params_slice: &[LineParams],
    res: &mut [LineResultData],
    bus_calc: &[BusPrecalc],
) {
    let len = res.len();
    unsafe {
        for i in 0..len {
            let f_idx = from_slice[i].0 as usize;
            let t_idx = to_slice[i].0 as usize;
            let f = &bus_calc[f_idx];
            let t = &bus_calc[t_idx];

            let flow = compute_two_port_flow_avx2(
                patch_slice[i].0.as_slice().as_ptr() as *const f64,
                f,
                t,
                f.p_scale,
            );

            let i_from_ka = flow.i0_norm * f.i_scale;
            let i_to_ka = flow.i1_norm * f.i_scale;
            let i_ka = i_from_ka.max(i_to_ka);

            let max_i = params_slice[i].max_i_ka;
            let loading_percent = if max_i > 0.0 {
                (i_ka / max_i) * 100.0
            } else {
                0.0
            };

            res[i] = LineResultData {
                p_from_mw: flow.p0_mw,
                q_from_mvar: flow.q0_mvar,
                p_to_mw: flow.p1_mw,
                q_to_mvar: flow.q1_mvar,
                pl_mw: flow.pl_mw,
                ql_mvar: flow.ql_mvar,
                i_from_ka,
                i_to_ka,
                i_ka,
                vm_from_pu: f.vm,
                va_from_degree: f.va_deg,
                vm_to_pu: t.vm,
                va_to_degree: t.va_deg,
                loading_percent,
            };
        }
    }
}

#[inline(always)]
fn process_lines_portable(
    patch_slice: &[Port4MatPatch],
    from_slice: &[FromBus],
    to_slice: &[ToBus],
    params_slice: &[LineParams],
    res: &mut [LineResultData],
    bus_calc: &[BusPrecalc],
) {
    let len = res.len();
    for i in 0..len {
        let f_idx = from_slice[i].0 as usize;
        let t_idx = to_slice[i].0 as usize;
        let f = &bus_calc[f_idx];
        let t = &bus_calc[t_idx];

        let flow = compute_two_port_flow_portable(
            unsafe { &*(patch_slice[i].0.as_slice().as_ptr() as *const [Complex64; 4]) },
            f,
            t,
            f.p_scale,
        );

        let i_from_ka = flow.i0_norm * f.i_scale;
        let i_to_ka = flow.i1_norm * f.i_scale;
        let i_ka = i_from_ka.max(i_to_ka);

        let max_i = params_slice[i].max_i_ka;
        let loading_percent = if max_i > 0.0 {
            (i_ka / max_i) * 100.0
        } else {
            0.0
        };

        res[i] = LineResultData {
            p_from_mw: flow.p0_mw,
            q_from_mvar: flow.q0_mvar,
            p_to_mw: flow.p1_mw,
            q_to_mvar: flow.q1_mvar,
            pl_mw: flow.pl_mw,
            ql_mvar: flow.ql_mvar,
            i_from_ka,
            i_to_ka,
            i_ka,
            vm_from_pu: f.vm,
            va_from_degree: f.va_deg,
            vm_to_pu: t.vm,
            va_to_degree: t.va_deg,
            loading_percent,
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn process_trafos_avx2(
    patch_slice: &[Port4MatPatch],
    from_slice: &[FromBus],
    to_slice: &[ToBus],
    dev_slice: &[TransformerDevice],
    res: &mut [TrafoResultData],
    bus_calc: &[BusPrecalc],
) {
    const SQRT3: f64 = 1.7320508075688772;
    const INV_SQRT3: f64 = 0.5773502691896257645;
    let len = res.len();
    unsafe {
        for i in 0..len {
            let hv_idx = from_slice[i].0 as usize;
            let lv_idx = to_slice[i].0 as usize;
            let f = &bus_calc[hv_idx];
            let t = &bus_calc[lv_idx];
            let dev = &dev_slice[i];

            let vn_lv = dev.vn_lv_kv;
            let vn_hv = dev.vn_hv_kv;
            let v_scale = vn_lv * vn_lv;

            let flow = compute_two_port_flow_avx2(
                patch_slice[i].0.as_slice().as_ptr() as *const f64,
                f,
                t,
                v_scale,
            );

            let base_i_hv_scale = v_scale / (SQRT3 * vn_hv);
            let base_i_lv_scale = vn_lv * INV_SQRT3;
            let i_hv_ka = flow.i0_norm * base_i_hv_scale;
            let i_lv_ka = flow.i1_norm * base_i_lv_scale;

            let sn_rated = dev.sn_mva * (dev.parallel as f64);
            let inv_sn_factor = if sn_rated > 0.0 {
                (v_scale * 100.0) / sn_rated
            } else {
                0.0
            };
            let loading_percent = flow.i0_norm.max(flow.i1_norm) * inv_sn_factor;

            res[i] = TrafoResultData {
                p_hv_mw: flow.p0_mw,
                q_hv_mvar: flow.q0_mvar,
                p_lv_mw: flow.p1_mw,
                q_lv_mvar: flow.q1_mvar,
                pl_mw: flow.pl_mw,
                ql_mvar: flow.ql_mvar,
                i_hv_ka,
                i_lv_ka,
                vm_hv_pu: f.vm,
                va_hv_degree: f.va_deg,
                vm_lv_pu: t.vm,
                va_lv_degree: t.va_deg,
                loading_percent,
            };
        }
    }
}

#[inline(always)]
fn process_trafos_portable(
    patch_slice: &[Port4MatPatch],
    from_slice: &[FromBus],
    to_slice: &[ToBus],
    dev_slice: &[TransformerDevice],
    res: &mut [TrafoResultData],
    bus_calc: &[BusPrecalc],
) {
    const SQRT3: f64 = 1.7320508075688772;
    const INV_SQRT3: f64 = 0.5773502691896257645;
    let len = res.len();
    for i in 0..len {
        let hv_idx = from_slice[i].0 as usize;
        let lv_idx = to_slice[i].0 as usize;
        let f = &bus_calc[hv_idx];
        let t = &bus_calc[lv_idx];
        let dev = &dev_slice[i];

        let vn_lv = dev.vn_lv_kv;
        let vn_hv = dev.vn_hv_kv;
        let v_scale = vn_lv * vn_lv;

        let flow = compute_two_port_flow_portable(
            unsafe { &*(patch_slice[i].0.as_slice().as_ptr() as *const [Complex64; 4]) },
            f,
            t,
            v_scale,
        );

        let base_i_hv_scale = v_scale / (SQRT3 * vn_hv);
        let base_i_lv_scale = vn_lv * INV_SQRT3;
        let i_hv_ka = flow.i0_norm * base_i_hv_scale;
        let i_lv_ka = flow.i1_norm * base_i_lv_scale;

        let sn_rated = dev.sn_mva * (dev.parallel as f64);
        let inv_sn_factor = if sn_rated > 0.0 {
            (v_scale * 100.0) / sn_rated
        } else {
            0.0
        };
        let loading_percent = flow.i0_norm.max(flow.i1_norm) * inv_sn_factor;

        res[i] = TrafoResultData {
            p_hv_mw: flow.p0_mw,
            q_hv_mvar: flow.q0_mvar,
            p_lv_mw: flow.p1_mw,
            q_lv_mvar: flow.q1_mvar,
            pl_mw: flow.pl_mw,
            ql_mvar: flow.ql_mvar,
            i_hv_ka,
            i_lv_ka,
            vm_hv_pu: f.vm,
            va_hv_degree: f.va_deg,
            vm_lv_pu: t.vm,
            va_lv_degree: t.va_deg,
            loading_percent,
        };
    }
}

/// Extracts line and transformer results using SIMD vectorized chunks + bypass_change_detection.
/// Completely skips OutOfService components with zero overhead.
/// Zero per-iteration heap allocations: bus_calc memory is retained in Local<Vec<BusPrecalc>>!
fn extract_res_branches(
    mut lines_q: Query<
        (
            &Port4MatPatch,
            &FromBus,
            &ToBus,
            &LineParams,
            &mut LineResultData,
        ),
        Without<OutOfService>,
    >,
    mut trafos_q: Query<
        (
            &Port4MatPatch,
            &FromBus,
            &ToBus,
            &TransformerDevice,
            &mut TrafoResultData,
        ),
        Without<OutOfService>,
    >,
    buses: Query<(&BusID, &VNominal)>,
    lut: Res<NodeLookup>,
    mat: Res<PowerFlowMat>,
    results: Res<PowerFlowResult>,
    cache: Option<Res<crate::basic::newtonpf::NewtonCache>>,
    mut bus_calc: Local<Vec<BusPrecalc>>,
    mut init_done: Local<bool>,
) {
    let nodes = lut.len();
    let v_slice = results.v.as_slice();

    const INV_SQRT3: f64 = 0.5773502691896257645;
    const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;

    // Single static setup of nominal voltage scales. Retained across runs with 0 malloc overhead!
    if !*init_done || bus_calc.len() != nodes {
        bus_calc.resize(
            nodes,
            BusPrecalc {
                vr: 1.0,
                vi: 0.0,
                vm: 1.0,
                va_deg: 0.0,
                p_scale: 1.0,
                i_scale: INV_SQRT3,
            },
        );
        if let Ok(chunks) = buses.contiguous_iter() {
            for (id_slice, vnom_slice) in chunks {
                let len = id_slice.len();
                for i in 0..len {
                    let idx = id_slice[i].0 as usize;
                    if idx < nodes {
                        let vn = vnom_slice[i].0.0;
                        bus_calc[idx].p_scale = vn * vn;
                        bus_calc[idx].i_scale = vn * INV_SQRT3;
                    }
                }
            }
        }
        *init_done = true;
    }

    // Direct read from solver polar state (NewtonCache) if present, eliminating atan2 & sqrt!
    // Zero branches in the permutation update loop for optimal pipeline throughput.
    if let Some(c) = cache.as_ref().filter(|c| {
        c.v_m.len() == nodes
            && c.v_a.len() == nodes
            && mat.from_perm.len() == nodes
            && v_slice.len() >= nodes
    }) {
        let vm_slice = c.v_m.as_slice();
        let va_slice = c.v_a.as_slice();
        for (new_idx, &orig_idx) in mat.from_perm.iter().enumerate() {
            let v = v_slice[orig_idx];
            bus_calc[orig_idx].vr = v.re;
            bus_calc[orig_idx].vi = v.im;
            bus_calc[orig_idx].vm = vm_slice[new_idx];
            bus_calc[orig_idx].va_deg = va_slice[new_idx] * RAD_TO_DEG;
        }
    } else {
        for i in 0..nodes.min(v_slice.len()) {
            let v = v_slice[i];
            let vr = v.re;
            let vi = v.im;
            bus_calc[i].vr = vr;
            bus_calc[i].vi = vi;
            bus_calc[i].vm = (vr * vr + vi * vi).sqrt();
            bus_calc[i].va_deg = vi.atan2(vr) * RAD_TO_DEG;
        }
    }

    // 1. Process In-service Lines using strictly contiguous slices with SIMD acceleration
    if let Ok(chunks) = lines_q.contiguous_iter_mut() {
        for (patch_slice, from_slice, to_slice, params_slice, mut res_slice) in chunks {
            let res = res_slice.bypass_change_detection();
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    unsafe {
                        process_lines_avx2(
                            patch_slice,
                            from_slice,
                            to_slice,
                            params_slice,
                            res,
                            &bus_calc,
                        );
                    }
                } else {
                    process_lines_portable(
                        patch_slice,
                        from_slice,
                        to_slice,
                        params_slice,
                        res,
                        &bus_calc,
                    );
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                process_lines_portable(
                    patch_slice,
                    from_slice,
                    to_slice,
                    params_slice,
                    res,
                    &bus_calc,
                );
            }
        }
    }

    // 2. Process In-service Transformers using strictly contiguous slices with SIMD acceleration
    if let Ok(chunks) = trafos_q.contiguous_iter_mut() {
        for (patch_slice, from_slice, to_slice, dev_slice, mut res_slice) in chunks {
            let res = res_slice.bypass_change_detection();
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    unsafe {
                        process_trafos_avx2(
                            patch_slice,
                            from_slice,
                            to_slice,
                            dev_slice,
                            res,
                            &bus_calc,
                        );
                    }
                } else {
                    process_trafos_portable(
                        patch_slice,
                        from_slice,
                        to_slice,
                        dev_slice,
                        res,
                        &bus_calc,
                    );
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                process_trafos_portable(
                    patch_slice,
                    from_slice,
                    to_slice,
                    dev_slice,
                    res,
                    &bus_calc,
                );
            }
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
        self.world_mut()
            .run_system_once(extract_res_branches)
            .unwrap();
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
        self.world_mut()
            .run_system_once(extract_res_branches)
            .unwrap();
    }
}
