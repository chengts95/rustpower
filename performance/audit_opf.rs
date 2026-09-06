//! Reproduce the OPF audit from the exact numerical model exported by pandapower.
//! cargo run --release --features klu_dyn --example audit_opf -- INPUT.json 5
use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64;
use rustpower::{new_opf, opf};
use serde::Deserialize;
use serde_json::json;
use std::time::Instant;

#[derive(Deserialize)]
struct Sparse {
    rows: usize,
    cols: usize,
    cp: Vec<usize>,
    ri: Vec<usize>,
    re: Vec<f64>,
    im: Vec<f64>,
}
impl Sparse {
    fn matrix(self) -> CscMatrix<Complex64> {
        let vals = self
            .re
            .into_iter()
            .zip(self.im)
            .map(|(r, i)| Complex64::new(r, i))
            .collect();
        CscMatrix::try_from_csc_data(self.rows, self.cols, self.cp, self.ri, vals).unwrap()
    }
}
#[derive(Deserialize)]
struct HessianProbe {
    x: Vec<f64>,
    lam: Vec<f64>,
    mu: Vec<f64>,
    hessian: Sparse,
}
#[derive(Deserialize)]
struct Input {
    hessian_probe: HessianProbe,
    network: rustpower::io::pandapower::Network,
    case: String,
    nb: usize,
    ng: usize,
    nl: usize,
    base_mva: f64,
    ref_bus: usize,
    ybus: Sparse,
    yf: Sparse,
    yt: Sparse,
    f_buses: Vec<usize>,
    t_buses: Vec<usize>,
    s_load_re: Vec<f64>,
    s_load_im: Vec<f64>,
    gen_bus: Vec<usize>,
    rate_a: Vec<f64>,
    cost_coeffs: Vec<[f64; 3]>,
    xmin: Vec<f64>,
    xmax: Vec<f64>,
    x0: Vec<f64>,
}
fn incidence(nb: usize, buses: &[usize]) -> CscMatrix<Complex64> {
    let mut coo = CooMatrix::new(nb, buses.len());
    for (j, &i) in buses.iter().enumerate() {
        coo.push(i, j, Complex64::new(1., 0.));
    }
    CscMatrix::from(&coo)
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let inp: Input = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let repeats: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(5);
    let Input {
        hessian_probe,
        network,
        case,
        nb,
        ng,
        nl,
        base_mva,
        ref_bus,
        ybus,
        yf,
        yt,
        f_buses,
        t_buses,
        s_load_re,
        s_load_im,
        gen_bus,
        rate_a,
        cost_coeffs,
        xmin,
        xmax,
        x0,
    } = inp;
    let mut cg = CooMatrix::new(nb, ng);
    for (g, &b) in gen_bus.iter().enumerate() {
        cg.push(b, g, 1.);
    }
    let base = opf::OPFData {
        nb,
        ng,
        nl,
        base_mva,
        ref_bus,
        ybus: ybus.matrix(),
        yf: yf.matrix(),
        yt: yt.matrix(),
        cf: incidence(nb, &f_buses),
        ct: incidence(nb, &t_buses),
        f_buses,
        t_buses,
        s_load: DVector::from_iterator(
            nb,
            s_load_re
                .into_iter()
                .zip(s_load_im)
                .map(|(r, i)| Complex64::new(r, i)),
        ),
        vm_min: xmin[nb..2 * nb].to_vec(),
        vm_max: xmax[nb..2 * nb].to_vec(),
        pg_min: xmin[2 * nb..2 * nb + ng].to_vec(),
        pg_max: xmax[2 * nb..2 * nb + ng].to_vec(),
        qg_min: xmin[2 * nb + ng..].to_vec(),
        qg_max: xmax[2 * nb + ng..].to_vec(),
        gen_bus,
        cg: CscMatrix::from(&cg),
        cost_coeffs,
        rate_a,
        pg_init: x0[2 * nb..2 * nb + ng].to_vec(),
        vm_set: x0[nb..2 * nb].to_vec(),
    };
    // Separately audit native network conversion; the solver trial uses the exact exported model.
    let native = opf::opf_data_from_network(&network);
    let diff = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    };
    let yd = &native.ybus - &base.ybus;
    let diagnostic = json!({"case":case,"same_bus_count":native.nb==nb,"same_gen_order":native.gen_bus==base.gen_bus,"same_branch_order":native.f_buses==base.f_buses && native.t_buses==base.t_buses,
        "ybus_max_abs_diff":yd.values().iter().map(|z|z.norm()).fold(0.0_f64,f64::max),
        "vm_min_max_diff":diff(&native.vm_min,&base.vm_min),"vm_max_max_diff":diff(&native.vm_max,&base.vm_max),
        "rate_a_max_diff":diff(&native.rate_a,&base.rate_a),
        "s_load_max_diff":native.s_load.iter().zip(base.s_load.iter()).map(|(a,b)|(a-b).norm()).fold(0.0_f64,f64::max)});
    let input_path = std::path::Path::new(&args[1]);
    std::fs::write(
        input_path.with_file_name(format!("{case}_conversion.json")),
        serde_json::to_string_pretty(&diagnostic).unwrap(),
    )
    .unwrap();
    // Validate scalar curvature against pandapower at a nonuniform perturbed state.
    // Keep this outside the timed solver calls.
    let probe = hessian_probe;
    let reference = probe.hessian.matrix();
    let cache = new_opf::assembly::v3::symbolic::V3SymbolicCache::analyze(&base);
    let h4 = new_opf::assembly::v4::curvature::v4_rect_numeric_fill(
        &base, &cache, &probe.x, &probe.lam, &probe.mu, None, 1e-4,
    );
    let h1 = opf::hessian::opf_hessfcn(&base, &probe.x, &probe.lam, &probe.mu, 1e-4);
    let compare_hessian = |h: &CscMatrix<f64>| {
        let mut actual = std::collections::BTreeMap::new();
        for j in 0..h.ncols() {
            for k in h.col_offsets()[j]..h.col_offsets()[j + 1] {
                actual.insert((h.row_indices()[k], j), h.values()[k]);
            }
        }
        let mut expected = std::collections::BTreeMap::new();
        for j in 0..reference.ncols() {
            for k in reference.col_offsets()[j]..reference.col_offsets()[j + 1] {
                expected.insert((reference.row_indices()[k], j), reference.values()[k].re);
            }
        }
        let mut abs = 0.0_f64;
        let mut scaled = 0.0_f64;
        for key in actual.keys().chain(expected.keys()) {
            let a = actual.get(key).copied().unwrap_or(0.0);
            let b = expected.get(key).copied().unwrap_or(0.0);
            abs = abs.max((a - b).abs());
            scaled = scaled.max((a - b).abs() / (1.0 + a.abs().max(b.abs())));
        }
        json!({"max_abs_diff":abs,"max_scaled_diff":scaled})
    };
    let v4_check = compare_hessian(&h4);
    let v1_check = compare_hessian(&h1);
    assert!(
        v1_check["max_scaled_diff"].as_f64().unwrap() < 1e-10,
        "V1 disagrees with pandapower Hessian: {v1_check}"
    );
    assert!(
        v4_check["max_scaled_diff"].as_f64().unwrap() < 1e-10,
        "V4 disagrees with pandapower Hessian: {v4_check}"
    );
    std::fs::write(input_path.with_file_name(format!("{case}_derivatives.json")),serde_json::to_string_pretty(&json!({"V4_vs_pandapower":v4_check,"V1_vs_pandapower":v1_check,"state":"perturbed x0 with nonuniform positive multipliers"})).unwrap()).unwrap();
    let (lo, hi) = base.bounds();
    for i in 0..base.nx() {
        assert!((lo[i] == xmin[i]) || (lo[i].is_infinite() && xmin[i] <= -1e20));
        assert!((hi[i] == xmax[i]) || (hi[i].is_infinite() && xmax[i] >= 1e20));
    }
    for version in ["V1", "V4", "V5.0", "V5.2", "V5.3", "V5.5", "V5.6"] {
        for trial in 0..=repeats {
            let opt = opf::PipsOpt {
                max_it: 150,
                cost_mult: 1e-4,
                ..Default::default()
            };
            // Bounds and initial-vector copies precede timing on both paths.
            let (seed, lower, upper) = (x0.clone(), lo.clone(), hi.clone());
            let t = Instant::now();
            let result = if version == "V1" {
                opf::pips::pips(
                    |x| opf::cost::opf_costfcn(&base, x),
                    |x| {
                        let (g, h, dg, dh) = opf::constraints::opf_consfcn(&base, x);
                        (h, g, dh, dg)
                    },
                    |x, l, m, _z, c| opf::hessian::opf_hessfcn(&base, x, l, m, c),
                    seed,
                    lower,
                    upper,
                    opt,
                )
            } else {
                // Include NewOPFData's legacy cache build and wrapper-specific symbolic setup.
                let data = new_opf::model::NewOPFData::new(base.clone());
                let solve = match version {
                    "V4" => new_opf::configurations::pips,
                    "V5.0" => new_opf::configurations::pips_v5,
                    "V5.2" => new_opf::configurations::pips_v5_2,
                    "V5.3" => new_opf::configurations::pips_v5_3,
                    "V5.5" => new_opf::configurations::pips_v5_5,
                    _ => new_opf::configurations::pips_v5_6,
                };
                solve(&data, seed, lower, upper, opt)
            };
            let total_ms = t.elapsed().as_secs_f64() * 1e3;
            if trial == 0 {
                continue;
            }
            let tm = &result.timing;
            println!(
                "{}",
                json!({"case":case,"version":version,"trial":trial,"converged":result.converged,"iterations":result.iterations,"f":result.f,"x":result.x,"total_ms":total_ms,"hess_ms":tm.hess.as_secs_f64()*1e3,"gh_ms":tm.gh.as_secs_f64()*1e3,"kkt_ms":tm.kkt.as_secs_f64()*1e3,"first_solve_ms":tm.solve_sym.as_secs_f64()*1e3,"later_solves_ms":tm.solve_num.as_secs_f64()*1e3})
            );
        }
    }
}
