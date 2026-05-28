pub mod symbolic;
pub mod numeric;
pub mod v3_symbolic;
pub mod v3_numeric;
pub mod v3_numeric_fused;
pub mod v3_numeric_scalar;
pub mod math_verify;
pub mod pips;
pub mod problem;
pub mod components;

pub use problem::NewOPFData;
pub use symbolic::SymbolicCache;
pub use pips::{pips, PipsOpt, PipsResult};
pub use components::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use crate::io::pandapower::load_csv_zip;
    use crate::opf::builder::opf_data_from_network;
    use crate::opf::PipsOpt;
    use nalgebra::DVector;
    use nalgebra_sparse::CscMatrix;
    use num_complex::Complex64;

    fn load_ieee39() -> crate::io::pandapower::Network {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE39/data.zip", dir);
        load_csv_zip(&path).unwrap()
    }

    #[test]
    fn test_new_opf_ieee39_run() {
        // Use the same embedded JSON case + cost mapping as opf::tests::test_opf_ieee39_run
        // so V1 and V3 solve the identical canonical pandapower case39 (ref ~41864 EUR).
        let net: crate::io::pandapower::Network =
            serde_json::from_str(crate::testcases::case_ieee39::IEEE_39).unwrap();
        let mut base_data = opf_data_from_network(&net);

        let opf_cfg = crate::io::pandapower::load_opf_cfg_json_str(
            crate::testcases::case_ieee39::IEEE_39,
        ).expect("poly_cost missing from case39 JSON");
        if let Some(row) = opf_cfg.get("ext_grid", 0) {
            base_data.cost_coeffs[0] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
        }
        for g in 0..9i64 {
            if let Some(row) = opf_cfg.get("gen", g) {
                base_data.cost_coeffs[(1 + g) as usize] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
            }
        }

        let data = NewOPFData::new(base_data);
        let x0 = data.warm_x0();
        let (xmin, xmax) = data.bounds();

        let result = pips(
            &data,
            x0, xmin, xmax,
            PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() },
        );

        println!(
            "New OPF 39: converged={} iter={} f={:.6} msg={}",
            result.converged, result.iterations, result.f, result.message
        );

        assert!(result.converged, "New OPF should converge");
        // Pandapower reference: ~41864.13 EUR
        assert!((result.f - 41864.13).abs() < 10.0, "Result mismatch: {}", result.f);
    }

    #[test]
    fn test_new_opf_ieee118_run() {
        let net = crate::io::pandapower::load_csv_zip(&format!("{}/cases/IEEE118/data.zip", std::env::var("CARGO_MANIFEST_DIR").unwrap())).unwrap();
        let mut base_data = opf_data_from_network(&net);

        // Load costs
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE118/data.zip", dir);
        if let Some(opf_cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
            if let Some(row) = opf_cfg.get("ext_grid", 0) {
                base_data.cost_coeffs[0] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
            }
            for g in 0..54i64 {
                if let Some(row) = opf_cfg.get("gen", g) {
                    base_data.cost_coeffs[(1 + g) as usize] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
                }
            }
        }

        let data = NewOPFData::new(base_data);
        let x0 = data.warm_x0();
        let (xmin, xmax) = data.bounds();

        let result = pips(
            &data,
            x0, xmin, xmax,
            PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() },
        );

        println!(
            "New OPF 118: converged={} iter={} f={:.6} msg={}",
            result.converged, result.iterations, result.f, result.message
        );

        assert!(result.converged, "New OPF 118 should converge");
        assert!(result.f > 120000.0 && result.f < 140000.0, "Objective out of expected range: {}", result.f);
    }

    #[test]
    fn test_baseline_opf_ieee39_run() {
        let net = load_ieee39();
        let mut base_data = opf_data_from_network(&net);
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE39/data.zip", dir);
        if let Some(opf_cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
             for g in 0..10i64 {
                if let Some(row) = opf_cfg.get("gen", g) {
                    base_data.cost_coeffs[g as usize] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
                }
            }
        }
        let x0 = base_data.warm_x0();
        let (xmin, xmax) = base_data.bounds();
        let result = crate::opf::pips::pips(
            |x| crate::opf::cost::opf_costfcn(&base_data, x),
            |x| {
                let (g, h, dg, dh) = crate::opf::constraints::opf_consfcn(&base_data, x);
                (h, g, dh, dg)
            },
            |x, l, m, c| crate::opf::hessian::opf_hessfcn(&base_data, x, l, m, c),
            x0, xmin, xmax,
            PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() },
        );
        println!("Baseline OPF 39: converged={} iter={} f={:.6}", result.converged, result.iterations, result.f);
    }

    #[test]
    fn test_exact_hessian_comparison_pegase9241() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/pegase9241/data.zip", dir);
        if !std::path::Path::new(&path).exists() { return; }
        
        let net = crate::io::pandapower::load_csv_zip(&path).unwrap();
        let base_data = opf_data_from_network(&net);
        let v3_cache = v3_symbolic::V3SymbolicCache::analyze(&base_data);

        let x = base_data.warm_x0();
        let nx = base_data.nx();
        // Branch-Hessian isolation: lam_eq = 0 (node balance off) keeps the Lagrangian O(1)
        // so finite differences are well-conditioned even on this 9241-bus synthetic point.
        // The node power-balance Hessian is FD-validated separately on IEEE118.
        let lam_eq = vec![0.0; 2 * base_data.nb];
        let mu_ineq = vec![0.05; 2 * base_data.nl];
        let cost_mult = 1e-4;

        let h_v1 = crate::opf::hessian::opf_hessfcn(&base_data, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);
        let h_v3 = v3_numeric_scalar::v3_scalar_numeric_fill(&base_data, &v3_cache, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);

        println!("Comparing V1 vs V3 branch Hessian ({} x {})", nx, nx);
        
        let mut diff_count = 0;
        let mut max_err: f64 = 0.0;
        let mut worst: (usize, usize) = (0, 0);
        let mut worst_v3val = 0.0f64;
        let mut worst_v1val = 0.0f64;

        use std::collections::HashMap;
        let mut v1_map = HashMap::new();
        for col in 0..nx {
            for idx in h_v1.col_offsets()[col]..h_v1.col_offsets()[col+1] {
                v1_map.insert((h_v1.row_indices()[idx], col), h_v1.values()[idx]);
            }
        }

        for col in 0..nx {
            for idx in h_v3.col_offsets()[col]..h_v3.col_offsets()[col+1] {
                let row = h_v3.row_indices()[idx];
                let val_v3 = h_v3.values()[idx];
                let val_v1 = *v1_map.get(&(row, col)).unwrap_or(&0.0);
                let err = (val_v1 - val_v3).abs();
                if err > 1e-7 {
                    diff_count += 1;
                    if err > max_err { max_err = err; worst = (row, col); worst_v3val = val_v3; worst_v1val = val_v1; }
                }
            }
        }

        // v3 and the legacy V1/d2ASbr path differ (V1 has a branch-Hessian diagonal bug).
        let _ = (worst, worst_v3val, worst_v1val);
        println!("Total v3-vs-V1 diffs: {}, Max Err: {:.6e} at {:?}", diff_count, max_err, worst);

        // Single-branch isolation FD: full-Lagrangian FD on a 9241-bus grid is dominated by
        // global roundoff (sum of all |Sf|^2), so adjudicate ONE branch at a time with a
        // one-hot multiplier — its penalty is O(1) and central FD is clean. We target a branch
        // touching the worst-differing bus to land on the legacy path's error.
        let worst_bus = if worst.0 < base_data.nb { worst.0 } else { worst.0 - base_data.nb };
        let l0 = (0..base_data.nl)
            .find(|&l| base_data.f_buses[l] == worst_bus || base_data.t_buses[l] == worst_bus)
            .unwrap_or(0);
        let f = base_data.f_buses[l0];
        let t = base_data.t_buses[l0];

        let mut mu1 = vec![0.0; 2 * base_data.nl];
        mu1[l0] = 0.05;
        mu1[base_data.nl + l0] = 0.05;
        let h1_v3 = v3_numeric_scalar::v3_scalar_numeric_fill(&base_data, &v3_cache, x.as_slice(), &lam_eq, &mu1, cost_mult);
        let h1_v1 = crate::opf::hessian::opf_hessfcn(&base_data, x.as_slice(), &lam_eq, &mu1, cost_mult);
        let get = |m: &CscMatrix<f64>, r: usize, c: usize| -> f64 {
            for idx in m.col_offsets()[c]..m.col_offsets()[c+1] {
                if m.row_indices()[idx] == r { return m.values()[idx]; }
            }
            0.0
        };

        let phi_l = |xv: &[f64]| -> f64 {
            let vv = base_data.v_from_x(xv);
            let ifv = &base_data.yf * &vv;
            let itv = &base_data.yt * &vv;
            let sf = vv[f] * ifv[l0].conj();
            let st = vv[t] * itv[l0].conj();
            0.05 * sf.norm_sqr() + 0.05 * st.norm_sqr()
        };
        let hstep = 1e-6;
        let d2 = |i: usize, k: usize| -> f64 {
            let mut a = x.clone(); a[i] += hstep; a[k] += hstep;
            let mut b = x.clone(); b[i] += hstep; b[k] -= hstep;
            let mut c = x.clone(); c[i] -= hstep; c[k] += hstep;
            let mut d = x.clone(); d[i] -= hstep; d[k] -= hstep;
            (phi_l(&a) - phi_l(&b) - phi_l(&c) + phi_l(&d)) / (4.0 * hstep * hstep)
        };
        let nb = base_data.nb;
        // Check all 16 entries of branch l0's [f,t] x {aa,av,va,vv} block.
        let nodes = [f, t];
        let mut worst_v3 = 0.0f64;
        let mut worst_v1 = 0.0f64;
        for ni in 0..2 {
            for nj in 0..2 {
                let r = nodes[ni];
                let c = nodes[nj];
                for &(rr, cc) in &[(r, c), (r, nb + c), (nb + r, c), (nb + r, nb + c)] {
                    let fd = d2(rr, cc);
                    let tol = 0.5 + 1e-3 * fd.abs();
                    worst_v3 = worst_v3.max((fd - get(&h1_v3, rr, cc)).abs() / tol);
                    worst_v1 = worst_v1.max((fd - get(&h1_v1, rr, cc)).abs() / tol);
                }
            }
        }
        println!("  single-branch l={} (f={},t={}) FD adjudication: worst v3 ratio={:.3e}  worst V1 ratio={:.3e}", l0, f, t, worst_v3, worst_v1);

        // v3 matches single-branch finite differences on the industrial grid; legacy path does not.
        assert!(worst_v3 < 1.0, "v3 branch Hessian disagrees with FD on pegase: ratio={:.3e}", worst_v3);
        assert!(worst_v1 > 1.0, "expected legacy d2ASbr to exceed FD tolerance on this branch");
    }

    #[test]
    #[ignore] // run explicitly: cargo test --release run_pegase9241_v3_convergence -- --ignored --nocapture
    fn run_pegase9241_v3_convergence() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/pegase9241/data.zip", dir);
        if !std::path::Path::new(&path).exists() {
            println!("pegase9241 data not present; skipping");
            return;
        }
        let net = crate::io::pandapower::load_csv_zip(&path).unwrap();
        let base_data = opf_data_from_network(&net);
        println!("pegase9241: nb={} nl={} ng={} nx={}", base_data.nb, base_data.nl, base_data.ng, base_data.nx());

        let data = NewOPFData::new(base_data);
        let x0 = data.warm_x0();
        let (xmin, xmax) = data.bounds();

        let t0 = std::time::Instant::now();
        let result = pips(
            &data,
            x0, xmin, xmax,
            PipsOpt { max_it: 200, cost_mult: 1e-4, ..Default::default() },
        );
        let dt = t0.elapsed();

        println!(
            "\n=== Corrected v3 on pegase9241 ===\nconverged={} iter={} f={:.4} msg={} time={:?}",
            result.converged, result.iterations, result.f, result.message, dt
        );
    }

    #[test]
    fn diag_hessian_breakdown_ieee118() {
        let net = crate::io::pandapower::load_csv_zip(&format!("{}/cases/IEEE118/data.zip", std::env::var("CARGO_MANIFEST_DIR").unwrap())).unwrap();
        let base_data = opf_data_from_network(&net);
        let v3_cache = v3_symbolic::V3SymbolicCache::analyze(&base_data);

        let nb = base_data.nb;
        let nx = base_data.nx();
        let x = base_data.warm_x0();
        let lam_eq = vec![0.1; 2 * base_data.nb];
        let mu_ineq = vec![0.05; 2 * base_data.nl];
        let cost_mult = 1e-4;

        let h_v1 = crate::opf::hessian::opf_hessfcn(&base_data, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);
        let h_vr = math_verify::verify_hessian(&base_data, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);
        let h_v3 = v3_numeric_scalar::v3_scalar_numeric_fill(&base_data, &v3_cache, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);

        use std::collections::HashMap;
        let to_map = |m: &CscMatrix<f64>| {
            let mut map = HashMap::new();
            for col in 0..nx {
                for idx in m.col_offsets()[col]..m.col_offsets()[col+1] {
                    map.insert((m.row_indices()[idx], col), m.values()[idx]);
                }
            }
            map
        };
        let m1 = to_map(&h_v1);
        let block = |r: usize, c: usize| -> &'static str {
            let rb = if r < nb {0} else if r < 2*nb {1} else {2};
            let cb = if c < nb {0} else if c < 2*nb {1} else {2};
            match (rb, cb) {
                (0,0)=>"aa",(0,1)=>"av",(1,0)=>"va",(1,1)=>"vv",
                (2,2)=>"gen",_=>"other",
            }
        };
        let compare = |label: &str, m2: &HashMap<(usize,usize), f64>| -> usize {
            let mut keys: std::collections::HashSet<(usize,usize)> = m1.keys().cloned().collect();
            keys.extend(m2.keys().cloned());
            let mut per_block: HashMap<&str,(usize,f64)> = HashMap::new();
            let mut diag_flip = 0usize;
            let mut total = 0usize;
            for k in keys {
                let a = *m1.get(&k).unwrap_or(&0.0);
                let b = *m2.get(&k).unwrap_or(&0.0);
                let err = (a-b).abs();
                if err > 1e-7 {
                    let e = per_block.entry(block(k.0,k.1)).or_insert((0,0.0));
                    e.0 += 1; e.1 = e.1.max(err);
                    total += 1;
                    if a*b < 0.0 && a.abs() > 1e-6 { diag_flip += 1; }
                }
            }
            println!("--- {} vs V1 ---", label);
            let mut blocks: Vec<_> = per_block.iter().collect();
            blocks.sort_by_key(|(k,_)| *k);
            for (blk, (cnt, mx)) in blocks {
                println!("  block {}: diffs={} maxErr={:.4e}", blk, cnt, mx);
            }
            println!("  sign-flipped entries: {}", diag_flip);
            total
        };
        // Informational: where v3 and the legacy V1/d2ASbr path disagree. V1 has a known
        // branch-Hessian diagonal error, so these are NOT v3 bugs (see FD gate below).
        let _ = compare("verify_hessian", &to_map(&h_vr));
        let d_v3_vs_v1 = compare("v3_scalar", &to_map(&h_v3));
        println!("  (v3 vs V1 diffs above are the legacy d2ASbr diagonal error, not v3 errors)");

        // === Gold standard: finite-difference the full Lagrangian and validate v3 ===
        // L(x) = cost_mult*f + lam_eq.g + mu_ineq.h   (same contraction as opf_hessfcn)
        let lag = |xv: &[f64]| -> f64 {
            let (f, _) = crate::opf::cost::opf_costfcn(&base_data, xv);
            let (g, hh, _, _) = crate::opf::constraints::opf_consfcn(&base_data, xv);
            let mut s = cost_mult * f;
            for i in 0..2 * nb { s += lam_eq[i] * g[i]; }
            for j in 0..2 * base_data.nl { s += mu_ineq[j] * hh[j]; }
            s
        };
        let hstep = 1e-5;
        let d2 = |i: usize, k: usize| -> f64 {
            let mut a = x.clone(); a[i] += hstep; a[k] += hstep;
            let mut b = x.clone(); b[i] += hstep; b[k] -= hstep;
            let mut c = x.clone(); c[i] -= hstep; c[k] += hstep;
            let mut d = x.clone(); d[i] -= hstep; d[k] -= hstep;
            (lag(&a) - lag(&b) - lag(&c) + lag(&d)) / (4.0 * hstep * hstep)
        };

        let m_v3 = to_map(&h_v3);
        let m_v1 = to_map(&h_v1);
        // Combined tolerance: ATOL absorbs the FD roundoff/truncation floor (~0.3 here at
        // h=1e-5), RTOL handles large entries. "ratio" = error / tol; ratio<1 means agreement.
        const ATOL: f64 = 0.5;
        const RTOL: f64 = 5e-4;
        let mut worst_v3 = (0usize, 0usize, 0.0f64, 0.0f64); // (r,c,abs,ratio)
        let mut worst_v1 = (0usize, 0usize, 0.0f64, 0.0f64);
        for (&(r, c), &val_v3) in m_v3.iter() {
            if r > c { continue; }
            if r >= 2 * nb && c >= 2 * nb { continue; } // gen cost: FD-trivial, skip
            let fd = d2(r, c);
            let tol = ATOL + RTOL * fd.abs();
            let (a3, q3) = ((fd - val_v3).abs(), (fd - val_v3).abs() / tol);
            if q3 > worst_v3.3 { worst_v3 = (r, c, a3, q3); }
            let val_v1 = *m_v1.get(&(r, c)).unwrap_or(&0.0);
            let (a1, q1) = ((fd - val_v1).abs(), (fd - val_v1).abs() / tol);
            if q1 > worst_v1.3 { worst_v1 = (r, c, a1, q1); }
        }
        println!("  FD gold standard (ratio = error / [ATOL+RTOL*|H|]):");
        println!("    worst v3: ratio={:.2e} abs={:.4e} at ({},{})", worst_v3.3, worst_v3.2, worst_v3.0, worst_v3.1);
        println!("    worst V1: ratio={:.2e} abs={:.4e} at ({},{})  <-- legacy d2ASbr branch-diagonal error", worst_v1.3, worst_v1.2, worst_v1.0, worst_v1.1);
        println!("    v3-vs-V1 structural diffs: {}", d_v3_vs_v1);

        // v3 agrees with analytic finite differences within the FD noise floor (true gold standard).
        assert!(worst_v3.3 < 1.0, "v3 Hessian disagrees with finite differences: ratio={:.3e}", worst_v3.3);
        // The legacy V1/d2ASbr path exceeds the FD noise floor — v3 fixes a real bug.
        assert!(worst_v1.3 > 1.0, "expected legacy path to exceed FD tolerance (known branch-diagonal error)");
    }

    #[test]
    fn bench_full_opf_all_cases() {
        let cases = ["IEEE39", "IEEE118", "pegase9241"];
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

        println!("\n| Case | Method | f [EUR] | Iter | Total Time | Speedup |");
        println!("|---|---|---|---|---|---|");

        for case in cases {
            let path = format!("{}/cases/{}/data.zip", dir, case);
            if !std::path::Path::new(&path).exists() { continue; }
            
            let net = crate::io::pandapower::load_csv_zip(&path).unwrap();
            let mut base_data = opf_data_from_network(&net);
            
            if case == "IEEE118" {
                if let Some(opf_cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
                    if let Some(row) = opf_cfg.get("ext_grid", 0) {
                        base_data.cost_coeffs[0] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
                    }
                    for g in 0..54i64 {
                        if let Some(row) = opf_cfg.get("gen", g) {
                            base_data.cost_coeffs[(1 + g) as usize] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
                        }
                    }
                }
            } else if case == "IEEE39" {
                 if let Some(opf_cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
                    for g in 0..10i64 {
                        if let Some(row) = opf_cfg.get("gen", g) {
                            base_data.cost_coeffs[g as usize] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
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
                |x, l, m, c| crate::opf::hessian::opf_hessfcn(&base_data, x, l, m, c),
                x0.clone(), xmin.clone(), xmax.clone(),
                PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() },
            );
            let dur_v1 = start_v1.elapsed();

            let start_v3 = std::time::Instant::now();
            let res_v3 = pips(
                &data_v3,
                x0, xmin, xmax,
                PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() },
            );
            let dur_v3 = start_v3.elapsed();

            let speedup = dur_v1.as_secs_f64() / dur_v3.as_secs_f64();
            println!("| {} | V1 | {:.2} | {} | {:?} | - |", case, res_v1.f, res_v1.iterations, dur_v1);
            println!("| {} | V3 | {:.2} | {} | {:?} | {:.2}x |", case, res_v3.f, res_v3.iterations, dur_v3, speedup);
        }
    }
}
