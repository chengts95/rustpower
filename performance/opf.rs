use rustpower::new_opf::*;
use rustpower::opf::PipsOpt;
use rustpower::opf::builder::opf_data_from_network;

pub fn bench_ablation_breakdown() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("\n| Case | Version | Iter | Hess | G/H | KKT | Solve | Overall |");
    println!("|---|---|---|---|---|---|---|---|");
    for case in ["IEEE39", "IEEE118", "pegase9241"] {
        let path = format!("{}/cases/{}/data.zip", dir, case);
        if !std::path::Path::new(&path).exists() {
            continue;
        }
        let net = crate::io::pandapower::load_csv_zip(&path).unwrap();
        let mut base_data = opf_data_from_network(&net);
        if let Some(cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
            if case == "IEEE39" {
                for g in 0..10i64 {
                    if let Some(r) = cfg.get("gen", g) {
                        base_data.cost_coeffs[g as usize] =
                            [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur];
                    }
                }
            } else {
                if let Some(r) = cfg.get("ext_grid", 0) {
                    base_data.cost_coeffs[0] = [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur];
                }
                for g in 0..54i64 {
                    if let Some(r) = cfg.get("gen", g) {
                        base_data.cost_coeffs[(1 + g) as usize] =
                            [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur];
                    }
                }
            }
        }
        let mi = if case == "pegase9241" { 30 } else { 150 };
        let x0 = base_data.warm_x0();
        let (xmin, xmax) = base_data.bounds();

        let row = |case: &str, ver: &str, r: &PipsResult, overall: std::time::Duration| {
            let t = &r.timing;
            println!(
                "| {} | {} | {} | {:?} | {:?} | {:?} | {:?} | {:?} | {:?} |",
                case, ver, r.iterations, t.hess, t.gh, t.kkt, t.solve_sym, t.solve_num, overall
            );
        };

        // V1 legacy (opf_hessfcn, no merged slacks)
        let t0 = std::time::Instant::now();
        let r1 = crate::opf::pips::pips(
            |x| crate::opf::cost::opf_costfcn(&base_data, x),
            |x| {
                let (g, h, dg, dh) = crate::opf::constraints::opf_consfcn(&base_data, x);
                (h, g, dh, dg)
            },
            |x, l, m, _z, c| crate::opf::hessian::opf_hessfcn(&base_data, x, l, m, c),
            x0.clone(),
            xmin.clone(),
            xmax.clone(),
            PipsOpt {
                max_it: mi,
                cost_mult: 1e-4,
                merged_slacks: false,
                ..Default::default()
            },
        );
        let d1 = t0.elapsed();
        row(case, "V1", &r1, d1);

        let data = NewOPFData::new(base_data.clone());
        let t0 = std::time::Instant::now();
        let r4 = pips(
            &data,
            x0.clone(),
            xmin.clone(),
            xmax.clone(),
            PipsOpt {
                max_it: mi,
                cost_mult: 1e-4,
                ..Default::default()
            },
        );
        let d4 = t0.elapsed();
        row(case, "V4", &r4, d4);

        let t0 = std::time::Instant::now();
        let r5 = pips::pips_v5(
            &data,
            x0.clone(),
            xmin.clone(),
            xmax.clone(),
            PipsOpt {
                max_it: mi,
                cost_mult: 1e-4,
                ..Default::default()
            },
        );
        let d5 = t0.elapsed();
        row(case, "V5.0", &r5, d5);

        let t0 = std::time::Instant::now();
        let r52 = pips::pips_v5_2(
            &data,
            x0.clone(),
            xmin.clone(),
            xmax.clone(),
            PipsOpt {
                max_it: mi,
                cost_mult: 1e-4,
                ..Default::default()
            },
        );
        let d52 = t0.elapsed();
        row(case, "V5.2", &r52, d52);

        let t0 = std::time::Instant::now();
        let r53 = pips::pips_v5_3(
            &data,
            x0.clone(),
            xmin.clone(),
            xmax.clone(),
            PipsOpt {
                max_it: mi,
                cost_mult: 1e-4,
                ..Default::default()
            },
        );
        let d53 = t0.elapsed();
        row(case, "V5.3", &r53, d53);

        let t0 = std::time::Instant::now();
        let r55 = pips::pips_v5_5(
            &data,
            x0.clone(),
            xmin.clone(),
            xmax.clone(),
            PipsOpt {
                max_it: mi,
                cost_mult: 1e-4,
                ..Default::default()
            },
        );
        let d55 = t0.elapsed();
        row(case, "V5.5", &r55, d55);
    }
}

pub fn bench_v4_vs_v5_endtoend() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("\n| Case | Path | f [EUR] | Iter | Total time |");
    println!("|---|---|---|---|---|");
    for case in ["IEEE39", "IEEE118", "pegase9241"] {
        let path = format!("{}/cases/{}/data.zip", dir, case);
        if !std::path::Path::new(&path).exists() {
            continue;
        }
        let net = crate::io::pandapower::load_csv_zip(&path).unwrap();
        let mut base_data = opf_data_from_network(&net);
        if let Some(cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
            let ng_cfg = if case == "IEEE39" { 10 } else { 54 };
            if case != "IEEE39" {
                if let Some(r) = cfg.get("ext_grid", 0) {
                    base_data.cost_coeffs[0] = [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur];
                }
                for g in 0..ng_cfg {
                    if let Some(r) = cfg.get("gen", g) {
                        base_data.cost_coeffs[(1 + g) as usize] =
                            [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur];
                    }
                }
            } else {
                for g in 0..ng_cfg {
                    if let Some(r) = cfg.get("gen", g) {
                        base_data.cost_coeffs[g as usize] =
                            [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur];
                    }
                }
            }
        }
        let data = NewOPFData::new(base_data);
        let x0 = data.warm_x0();
        let (xmin, xmax) = data.bounds();
        let mi = if case == "pegase9241" { 30 } else { 150 };

        let t4 = std::time::Instant::now();
        let r4 = pips(
            &data,
            x0.clone(),
            xmin.clone(),
            xmax.clone(),
            PipsOpt {
                max_it: mi,
                cost_mult: 1e-4,
                ..Default::default()
            },
        );
        let d4 = t4.elapsed();
        let t5 = std::time::Instant::now();
        let r5 = pips::pips_v5(
            &data,
            x0.clone(),
            xmin.clone(),
            xmax.clone(),
            PipsOpt {
                max_it: mi,
                cost_mult: 1e-4,
                ..Default::default()
            },
        );
        let d5 = t5.elapsed();
        println!(
            "| {} | V4.0 | {:.2} | {} | {:?} |",
            case, r4.f, r4.iterations, d4
        );
        println!(
            "| {} | V5.0 | {:.2} | {} | {:?} |",
            case, r5.f, r5.iterations, d5
        );
    }
}

pub fn bench_full_opf_all_cases() {
    let cases = ["IEEE39", "IEEE118", "pegase9241"];
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    println!("\n| Case | Method | f [EUR] | Iter | Total Time | Speedup |");
    println!("|---|---|---|---|---|---|");

    for case in cases {
        let path = format!("{}/cases/{}/data.zip", dir, case);
        if !std::path::Path::new(&path).exists() {
            continue;
        }

        let net = crate::io::pandapower::load_csv_zip(&path).unwrap();
        let mut base_data = opf_data_from_network(&net);

        if case == "IEEE118" {
            if let Some(opf_cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
                if let Some(row) = opf_cfg.get("ext_grid", 0) {
                    base_data.cost_coeffs[0] =
                        [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
                }
                for g in 0..54i64 {
                    if let Some(row) = opf_cfg.get("gen", g) {
                        base_data.cost_coeffs[(1 + g) as usize] =
                            [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
                    }
                }
            }
        } else if case == "IEEE39" {
            if let Some(opf_cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
                for g in 0..10i64 {
                    if let Some(row) = opf_cfg.get("gen", g) {
                        base_data.cost_coeffs[g as usize] =
                            [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
                    }
                }
            }
        }

        let data_v3 = NewOPFData::new(base_data.clone());
        let x0 = base_data.warm_x0();
        let (xmin, xmax) = base_data.bounds();

        let start_v1 = std::time::Instant::now();
        let res_v1 = crate::opf::pips::pips(
            |x| crate::opf::cost::opf_costfcn(&base_data, x),
            |x| {
                let (g, h, dg, dh) = crate::opf::constraints::opf_consfcn(&base_data, x);
                (h, g, dh, dg)
            },
            |x, l, m, _z, c| crate::opf::hessian::opf_hessfcn(&base_data, x, l, m, c),
            x0.clone(),
            xmin.clone(),
            xmax.clone(),
            PipsOpt {
                max_it: 150,
                cost_mult: 1e-4,
                merged_slacks: false,
                ..Default::default()
            },
        );
        let dur_v1 = start_v1.elapsed();

        let start_v3 = std::time::Instant::now();
        let res_v3 = pips(
            &data_v3,
            x0.clone(),
            xmin.clone(),
            xmax.clone(),
            PipsOpt {
                max_it: 150,
                cost_mult: 1e-4,
                merged_slacks: false,
                ..Default::default()
            },
        );
        let dur_v3 = start_v3.elapsed();

        // V4 Benchmark
        let v3_cache = v3_symbolic::V3SymbolicCache::analyze(&data_v3);
        let start_v4 = std::time::Instant::now();
        let res_v4 = crate::opf::pips::pips(
            |x| crate::opf::cost::opf_costfcn(&data_v3, x),
            |x| {
                let (g, h, dg, dh) = crate::opf::constraints::opf_consfcn(&data_v3, x);
                (h, g, dh, dg)
            },
            |x, lam_eq, mu_ineq, z_ineq, cost_mult| {
                v4_numeric_rect::v4_rect_numeric_fill(
                    &data_v3,
                    &v3_cache,
                    x,
                    lam_eq,
                    mu_ineq,
                    Some(z_ineq),
                    cost_mult,
                )
            },
            x0,
            xmin,
            xmax,
            PipsOpt {
                max_it: 150,
                cost_mult: 1e-4,
                merged_slacks: true,
                ..Default::default()
            },
        );
        let dur_v4 = start_v4.elapsed();

        let speedup3 = dur_v1.as_secs_f64() / dur_v3.as_secs_f64();
        let speedup4 = dur_v1.as_secs_f64() / dur_v4.as_secs_f64();
        println!(
            "| {} | V1 | {:.2} | {} | {:?} | - |",
            case, res_v1.f, res_v1.iterations, dur_v1
        );
        println!(
            "| {} | V3 | {:.2} | {} | {:?} | {:.2}x |",
            case, res_v3.f, res_v3.iterations, dur_v3, speedup3
        );
        println!(
            "| {} | V4 | {:.2} | {} | {:?} | {:.2}x |",
            case, res_v4.f, res_v4.iterations, dur_v4, speedup4
        );
    }
}
