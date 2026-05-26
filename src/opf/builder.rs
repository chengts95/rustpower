use std::collections::HashMap;
use std::f64::consts::PI;

use nalgebra::DVector;
use nalgebra_sparse::CooMatrix;
use num_complex::Complex64;

use crate::io::pandapower::{ExtGrid, Gen as GenData, Line, Network, Transformer};

use super::problem::OPFData;

/// Convert a pandapower Network into an OPFData suitable for AC-OPF.
///
/// Generator ordering: ext_grid entries first, then gen entries.
/// Cost coefficients default to zero (ext_grid) or linear c1=1/base_mva (gen).
/// Inject real costs via ECS `GenCost` components; see `patch_gen_cost`.
#[allow(non_snake_case)]
pub fn opf_data_from_network(net: &Network) -> OPFData {
    let base_mva = net.sn_mva;
    let wbase = 2.0 * PI * net.f_hz;

    // Bus ID → internal index (0..nb-1)
    let nb = net.bus.len();
    let mut bus_id_to_idx: HashMap<i64, usize> = HashMap::with_capacity(nb);
    for (i, bus) in net.bus.iter().enumerate() {
        bus_id_to_idx.insert(bus.index, i);
    }

    // Voltage limits
    let vm_min: Vec<f64> = net.bus.iter().map(|b| b.min_vm_pu.unwrap_or(0.9)).collect();
    let vm_max: Vec<f64> = net.bus.iter().map(|b| b.max_vm_pu.unwrap_or(1.1)).collect();

    // Vbase per bus [kV]
    let vbase: Vec<f64> = net.bus.iter().map(|b| b.vn_kv).collect();

    // Reference bus from ext_grid (first active one)
    let ext_grids = net.ext_grid.as_deref().unwrap_or(&[]);
    let ref_bus = ext_grids
        .iter()
        .filter(|eg| eg.in_service)
        .next()
        .map(|eg| bus_id_to_idx[&eg.bus])
        .unwrap_or(0);

    // Net load: s_load[bus] = Pd+jQd [p.u.] (loads and shunts)
    let mut s_load = DVector::zeros(nb);
    for load in net.load.as_deref().unwrap_or(&[]) {
        if !load.in_service { continue; }
        if let Some(&idx) = bus_id_to_idx.get(&load.bus) {
            s_load[idx] += Complex64::new(
                load.p_mw * load.scaling / base_mva,
                load.q_mvar * load.scaling / base_mva,
            );
        }
    }
    // sgen as negative load (fixed P/Q injection)
    for sg in net.sgen.as_deref().unwrap_or(&[]) {
        if !sg.in_service { continue; }
        if let Some(&idx) = bus_id_to_idx.get(&sg.bus) {
            s_load[idx] -= Complex64::new(
                sg.p_mw * sg.scaling / base_mva,
                sg.q_mvar * sg.scaling / base_mva,
            );
        }
    }
    // Shunts as constant admittance load approximation at V=1 pu
    for sh in net.shunt.as_deref().unwrap_or(&[]) {
        if !sh.in_service { continue; }
        if let Some(&idx) = bus_id_to_idx.get(&sh.bus) {
            let step = sh.step as f64;
            s_load[idx] += Complex64::new(
                sh.p_mw * step / base_mva,
                sh.q_mvar * step / base_mva,  // q_mvar is positive for capacitive (inductive load)
            );
        }
    }

    // ── Branch processing ──────────────────────────────────────────────────────
    // Per-branch: (f_bus, t_bus, Yf[l, f], Yf[l, t], Yt[l, f], Yt[l, t], rate_a)
    let mut f_buses: Vec<usize> = Vec::new();
    let mut t_buses: Vec<usize> = Vec::new();
    // COO entries for Yf (nl × nb) and Yt (nl × nb)
    let mut yf_rows: Vec<usize> = Vec::new();
    let mut yf_cols: Vec<usize> = Vec::new();
    let mut yf_vals: Vec<Complex64> = Vec::new();
    let mut yt_rows: Vec<usize> = Vec::new();
    let mut yt_cols: Vec<usize> = Vec::new();
    let mut yt_vals: Vec<Complex64> = Vec::new();
    let mut rate_a: Vec<f64> = Vec::new();

    // Lines
    for line in net.line.as_deref().unwrap_or(&[]) {
        if !line.in_service { continue; }
        let Some(&f) = bus_id_to_idx.get(&line.from_bus) else { continue };
        let Some(&t) = bus_id_to_idx.get(&line.to_bus) else { continue };
        let l = f_buses.len();
        f_buses.push(f);
        t_buses.push(t);

        let (yff, yft, ytf, ytt, smax) = line_admittances(line, vbase[f], base_mva, wbase);
        push_branch(l, f, t, yff, yft, ytf, ytt,
            &mut yf_rows, &mut yf_cols, &mut yf_vals,
            &mut yt_rows, &mut yt_cols, &mut yt_vals);
        rate_a.push(smax);
    }

    // Transformers
    for trafo in net.trafo.as_deref().unwrap_or(&[]) {
        if !trafo.in_service { continue; }
        let Some(&f) = bus_id_to_idx.get(&(trafo.hv_bus as i64)) else { continue };
        let Some(&t) = bus_id_to_idx.get(&(trafo.lv_bus as i64)) else { continue };
        let l = f_buses.len();
        f_buses.push(f);
        t_buses.push(t);

        let (yff, yft, ytf, ytt, smax) = trafo_admittances(trafo, base_mva);
        push_branch(l, f, t, yff, yft, ytf, ytt,
            &mut yf_rows, &mut yf_cols, &mut yf_vals,
            &mut yt_rows, &mut yt_cols, &mut yt_vals);
        rate_a.push(smax);
    }

    let nl = f_buses.len();

    // Build Yf (nl × nb) and Yt (nl × nb) as CSC
    let yf = coo_to_csc_complex(nl, nb, &yf_rows, &yf_cols, &yf_vals);
    let yt = coo_to_csc_complex(nl, nb, &yt_rows, &yt_cols, &yt_vals);

    // Build Ybus = Cf^T * Yf + Ct^T * Yt
    // Ybus[f[l], j] += Yf[l, j]   for each branch l
    // Ybus[t[l], j] += Yt[l, j]
    let ybus = build_ybus(nb, &f_buses, &t_buses, &yf, &yt);

    // Cf (nb × nl): Cf[f[l], l] = 1,  Ct (nb × nl): Ct[t[l], l] = 1
    let (cf, ct) = build_incidence_cx(nb, nl, &f_buses, &t_buses);

    // ── Generators ─────────────────────────────────────────────────────────────
    // ext_grid first, then gen — both sorted in original order
    // Voltage setpoint per bus [p.u.]: overridden by ext_grid/gen vm_pu below
    let mut vm_set: Vec<f64> = vec![1.0f64; nb];

    let mut gen_bus: Vec<usize> = Vec::new();
    let mut pg_min: Vec<f64> = Vec::new();
    let mut pg_max: Vec<f64> = Vec::new();
    let mut qg_min: Vec<f64> = Vec::new();
    let mut qg_max: Vec<f64> = Vec::new();
    let mut cost_coeffs: Vec<[f64; 3]> = Vec::new();
    let mut pg_init: Vec<f64> = Vec::new();

    for eg in ext_grids {
        if !eg.in_service { continue; }
        let bus = bus_id_to_idx[&eg.bus];
        vm_set[bus] = eg.vm_pu;
        push_ext_grid(eg, &bus_id_to_idx, base_mva,
            &mut gen_bus, &mut pg_min, &mut pg_max,
            &mut qg_min, &mut qg_max, &mut cost_coeffs, &mut pg_init);
    }
    for generator in net.r#gen.as_deref().unwrap_or(&[]) {
        if !generator.in_service { continue; }
        let bus = bus_id_to_idx[&generator.bus];
        vm_set[bus] = generator.vm_pu;
        push_gen(generator, &bus_id_to_idx, base_mva,
            &mut gen_bus, &mut pg_min, &mut pg_max,
            &mut qg_min, &mut qg_max, &mut cost_coeffs, &mut pg_init);
    }

    let ng = gen_bus.len();

    // Cg: nb × ng, Cg[bus, g] = 1
    let cg = {
        let mut rows = Vec::with_capacity(ng);
        let mut cols = Vec::with_capacity(ng);
        let mut vals = Vec::with_capacity(ng);
        for (g, &bus) in gen_bus.iter().enumerate() {
            rows.push(bus);
            cols.push(g);
            vals.push(1.0f64);
        }
        let coo = CooMatrix::try_from_triplets(nb, ng, rows, cols, vals).unwrap();
        nalgebra_sparse::CscMatrix::from(&coo)
    };

    OPFData {
        nb, nl, ng,
        ybus, yf, yt,
        f_buses, t_buses,
        s_load,
        vm_min, vm_max,
        ref_bus,
        gen_bus,
        cg,
        pg_min, pg_max,
        qg_min, qg_max,
        cost_coeffs,
        base_mva,
        rate_a,
        cf, ct,
        pg_init,
        vm_set,
    }
}

// ── per-element helpers ───────────────────────────────────────────────────────


/// Compute (Yf[l,f], Yf[l,t], Yt[l,f], Yt[l,t], smax_pu) for a line (π-model).
fn line_admittances(
    line: &Line,
    vbase_kv: f64,
    base_mva: f64,
    wbase: f64,
) -> (Complex64, Complex64, Complex64, Complex64, f64) {
    let parallel = line.parallel as f64;
    let length = line.length_km;

    // Physical series admittance [S]: y = N / (r+jx)
    let rl = line.r_ohm_per_km * length;
    let xl = line.x_ohm_per_km * length;
    let y_series_phys = parallel / Complex64::new(rl, xl);

    // Physical shunt admittance at each end [S]
    let b_phys = wbase * 1e-9 * line.c_nf_per_km * length * parallel;
    let g_phys = 1e-6 * line.g_us_per_km * length * parallel;
    let y_shunt_phys = 0.5 * Complex64::new(g_phys, b_phys);

    // Per-unit conversion: y_pu = y_phys * vbase_kv² / base_mva
    let scale = vbase_kv * vbase_kv / base_mva;
    let ys = y_series_phys * scale;
    let yc = y_shunt_phys * scale;

    // Flow limit [p.u.]: sqrt(3) factor already implicit in apparent power
    let smax_pu = line.max_i_ka * vbase_kv * (3.0f64).sqrt() / base_mva;

    (ys + yc, -ys, -ys, ys + yc, smax_pu)
}

/// Compute (Yf[l,f], Yf[l,t], Yt[l,f], Yt[l,t], smax_pu) for a transformer.
fn trafo_admittances(
    trafo: &Transformer,
    base_mva: f64,
) -> (Complex64, Complex64, Complex64, Complex64, f64) {
    let parallel = trafo.parallel as f64;
    let v_base = trafo.vn_lv_kv;  // LV side base [kV]

    // Series impedance on transformer's own base [Ω]
    let z_base = v_base * v_base / trafo.sn_mva;
    let vk = trafo.vk_percent * 0.01;
    let vkr = trafo.vkr_percent * 0.01;
    let z = z_base * vk;
    let re = z_base * vkr;
    let im = (z * z - re * re).max(0.0).sqrt();
    let y_series_phys = parallel / Complex64::new(re, im);

    // Magnetizing branch (core losses)
    let re_core = z_base * 0.001 * trafo.pfe_kw / trafo.sn_mva;
    let im_core = if trafo.i0_percent > 0.0 { z_base / (0.01 * trafo.i0_percent) } else { 0.0 };
    let y_m_phys = if re_core > 0.0 || im_core > 0.0 {
        parallel / Complex64::new(re_core, im_core)
    } else {
        Complex64::new(0.0, 0.0)
    };

    // Tap ratio
    let tap_m = trafo.tap_pos
        .zip(trafo.tap_neutral)
        .zip(trafo.tap_step_percent)
        .map(|((pos, neutral), step)| 1.0 + (pos - neutral) * 0.01 * step)
        .unwrap_or(1.0);
    let a = Complex64::from_polar(tap_m, trafo.shift_degree.to_radians());

    // Physical admittance matrix (π-model with ideal transformer)
    let yff_phys = (y_series_phys + 0.5 * y_m_phys) / a.norm_sqr();
    let yft_phys = -y_series_phys / a.conj();
    let ytf_phys = -y_series_phys / a;
    let ytt_phys = y_series_phys + 0.5 * y_m_phys;

    // Per-unit: scale by vbase_lv² / sbase_sys
    let scale = v_base * v_base / base_mva;

    let smax_pu = trafo.sn_mva * parallel
        * trafo.max_loading_percent.unwrap_or(100.0) / 100.0
        / base_mva;

    (yff_phys * scale, yft_phys * scale, ytf_phys * scale, ytt_phys * scale, smax_pu)
}

fn push_branch(
    l: usize, f: usize, t: usize,
    yff: Complex64, yft: Complex64, ytf: Complex64, ytt: Complex64,
    yf_rows: &mut Vec<usize>, yf_cols: &mut Vec<usize>, yf_vals: &mut Vec<Complex64>,
    yt_rows: &mut Vec<usize>, yt_cols: &mut Vec<usize>, yt_vals: &mut Vec<Complex64>,
) {
    yf_rows.push(l); yf_cols.push(f); yf_vals.push(yff);
    yf_rows.push(l); yf_cols.push(t); yf_vals.push(yft);
    yt_rows.push(l); yt_cols.push(f); yt_vals.push(ytf);
    yt_rows.push(l); yt_cols.push(t); yt_vals.push(ytt);
}

fn push_ext_grid(
    eg: &ExtGrid,
    lut: &HashMap<i64, usize>,
    base_mva: f64,
    gen_bus: &mut Vec<usize>,
    pg_min: &mut Vec<f64>,
    pg_max: &mut Vec<f64>,
    qg_min: &mut Vec<f64>,
    qg_max: &mut Vec<f64>,
    cost_coeffs: &mut Vec<[f64; 3]>,
    pg_init: &mut Vec<f64>,
) {
    let Some(&bus) = lut.get(&eg.bus) else { return };
    gen_bus.push(bus);
    // Use ±∞ when no limit is specified — matches PYPOWER/pandapower convention for slack bus.
    // Finite ±999 defaults inflate barrier slacks and destroy PIPS complementarity convergence.
    pg_min.push(eg.min_p_mw.map(|p| p / base_mva).unwrap_or(f64::NEG_INFINITY));
    pg_max.push(eg.max_p_mw.map(|p| p / base_mva).unwrap_or(f64::INFINITY));
    qg_min.push(eg.min_q_mvar.map(|q| q / base_mva).unwrap_or(f64::NEG_INFINITY));
    qg_max.push(eg.max_q_mvar.map(|q| q / base_mva).unwrap_or(f64::INFINITY));
    cost_coeffs.push([0.0, 0.0, 0.0]);
    pg_init.push(0.0); // slack P is free; not used in warm-start Sbus
}

fn push_gen(
    g: &GenData,
    lut: &HashMap<i64, usize>,
    base_mva: f64,
    gen_bus: &mut Vec<usize>,
    pg_min: &mut Vec<f64>,
    pg_max: &mut Vec<f64>,
    qg_min: &mut Vec<f64>,
    qg_max: &mut Vec<f64>,
    cost_coeffs: &mut Vec<[f64; 3]>,
    pg_init: &mut Vec<f64>,
) {
    let Some(&bus) = lut.get(&g.bus) else { return };
    gen_bus.push(bus);
    pg_min.push(g.min_p_mw / base_mva);
    pg_max.push(g.max_p_mw / base_mva);
    qg_min.push(g.min_q_mvar / base_mva);
    qg_max.push(g.max_q_mvar / base_mva);
    cost_coeffs.push([0.0, 1.0 / base_mva, 0.0]);
    pg_init.push(g.p_mw / base_mva);
}

// ── sparse matrix helpers ─────────────────────────────────────────────────────

/// Build a complex CSC matrix from COO triplets (summing duplicate entries).
fn coo_to_csc_complex(
    nrows: usize,
    ncols: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[Complex64],
) -> nalgebra_sparse::CscMatrix<Complex64> {
    let mut coo = CooMatrix::new(nrows, ncols);
    for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
        coo.push(r, c, v);
    }
    nalgebra_sparse::CscMatrix::from(&coo)
}

/// Assemble Ybus (nb×nb) from Yf and Yt:  Ybus = Cf^T*Yf + Ct^T*Yt
fn build_ybus(
    nb: usize,
    f_buses: &[usize],
    t_buses: &[usize],
    yf: &nalgebra_sparse::CscMatrix<Complex64>,
    yt: &nalgebra_sparse::CscMatrix<Complex64>,
) -> nalgebra_sparse::CscMatrix<Complex64> {
    let nl = f_buses.len();
    let mut coo: CooMatrix<Complex64> = CooMatrix::new(nb, nb);

    // Ybus[f[l], j] += Yf[l, j]
    for l in 0..nl {
        let f = f_buses[l];
        for idx in yf.col_offsets()[0]..yf.col_offsets()[nb] {
            // iterate row l of Yf (CSC: need CSR access)
            let _ = idx; // we'll do it column by column
        }
        // Use per-column iteration on yf and yt
        let _ = f;
    }
    // Better: iterate columns of Yf (which are buses j=0..nb)
    // Yf is (nl × nb), so column j gives all branches that connect to bus j
    // Ybus[f[l], j] += Yf[l, j] → for each (l, j) entry in Yf, add to Ybus[f[l], j]
    for j in 0..nb {
        for idx in yf.col_offsets()[j]..yf.col_offsets()[j + 1] {
            let l = yf.row_indices()[idx];
            let v = yf.values()[idx];
            coo.push(f_buses[l], j, v);
        }
        for idx in yt.col_offsets()[j]..yt.col_offsets()[j + 1] {
            let l = yt.row_indices()[idx];
            let v = yt.values()[idx];
            coo.push(t_buses[l], j, v);
        }
    }

    nalgebra_sparse::CscMatrix::from(&coo)
}

/// Cf (nb × nl): Cf[f[l], l] = 1.  Ct (nb × nl): Ct[t[l], l] = 1.
fn build_incidence_cx(
    nb: usize,
    nl: usize,
    f_buses: &[usize],
    t_buses: &[usize],
) -> (nalgebra_sparse::CscMatrix<Complex64>, nalgebra_sparse::CscMatrix<Complex64>) {
    let mut cf_coo: CooMatrix<Complex64> = CooMatrix::new(nb, nl);
    let mut ct_coo: CooMatrix<Complex64> = CooMatrix::new(nb, nl);
    let one = Complex64::new(1.0, 0.0);
    for (l, (&f, &t)) in f_buses.iter().zip(t_buses.iter()).enumerate() {
        cf_coo.push(f, l, one);
        ct_coo.push(t, l, one);
    }
    (
        nalgebra_sparse::CscMatrix::from(&cf_coo),
        nalgebra_sparse::CscMatrix::from(&ct_coo),
    )
}
