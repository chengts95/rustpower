pub mod symbolic;
pub mod numeric;
pub mod v3_symbolic;
pub mod v3_numeric;
pub mod v3_numeric_fused;
pub mod v3_numeric_scalar;
pub mod v4_numeric_rect;
pub mod v5_kkt;
pub mod v5_2_kernel;
pub mod v5_3_kernel;
pub mod math_verify;
pub mod pips;
pub mod problem;
pub mod components;
pub mod translate;

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
    use nalgebra_sparse::{CscMatrix, CooMatrix};
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
    fn test_v5_2_pips_ieee118() {
        let net = crate::io::pandapower::load_csv_zip(&format!("{}/cases/IEEE118/data.zip", std::env::var("CARGO_MANIFEST_DIR").unwrap())).unwrap();
        let mut base_data = opf_data_from_network(&net);
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
        let opt = PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() };
        let res52 = pips::pips_v5_2(&data, x0, xmin, xmax, opt);
        println!("V5.2: converged={} iter={} f={:.6}", res52.converged, res52.iterations, res52.f);
        assert!(res52.converged, "V5.2 should converge");
        assert!((res52.f - 129662.9725).abs() < 1e-2, "V5.2 result mismatch: {}", res52.f);
    }

    #[test]
    fn test_v5_versions_vs_v4_kkt_breakdown() {
        let net = crate::io::pandapower::load_csv_zip(&format!("{}/cases/IEEE118/data.zip", std::env::var("CARGO_MANIFEST_DIR").unwrap())).unwrap();
        let mut base_data = opf_data_from_network(&net);
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
        let nb = data.nb;
        let nx = data.nx();
        let x = data.warm_x0();
        let lam_eq = vec![0.1; 2 * data.nb];
        let mu_ineq = vec![0.05; 2 * data.nl];
        let z_ineq = vec![0.7; 2 * data.nl];
        let cost_mult = 1e-4;

        let v3c = crate::new_opf::v3_symbolic::V3SymbolicCache::analyze(&data);
        let v5_sym = crate::new_opf::v5_kkt::KKTSymbolicV5::build(&data);
        let v53_sym = v5_3_kernel::KKTSymbolicV5_3::build(&data);

        // --- 1. Baseline: V4 Lxx and opf_consfcn dg ---
        let lxx_v4 = crate::new_opf::v4_numeric_rect::v4_rect_numeric_fill(
            &data, &v3c, x.as_slice(), &lam_eq, &mu_ineq, Some(&z_ineq), cost_mult,
        );
        let (_, _, dgn, _) = crate::opf::constraints::opf_consfcn(&data, x.as_slice());
        // build merged dg
        let neqlin = v5_sym.ieq.len();
        let mut dg_coo = CooMatrix::<f64>::new(nx, 2 * nb + neqlin);
        for j in 0..dgn.ncols() {
            for idx in dgn.col_offsets()[j]..dgn.col_offsets()[j + 1] {
                dg_coo.push(dgn.row_indices()[idx], j, dgn.values()[idx]);
            }
        }
        for (r, &v) in v5_sym.ieq.iter().enumerate() {
            dg_coo.push(v, 2 * nb + r, 1.0);
        }
        let dg_full = CscMatrix::from(&dg_coo);
        let dg_full_t = dg_full.transpose();

        // Reference Full KKT values (V4 style)
        let ref_kkt = crate::opf::pips::build_saddle_point(&lxx_v4, &Some(dg_full), nx, v5_sym.neq);
        let ref_vals = ref_kkt.values();

        // --- 2. Evaluate V5.2 ---
        let mut vals52 = vec![0.0; v5_sym.row_idx.len()];
        let mut gens_at_bus: Vec<Vec<usize>> = vec![Vec::new(); nb];
        for g in 0..data.ng { gens_at_bus[data.gen_bus[g]].push(g); }
        v5_2_kernel::fill_variable_columns(&v5_sym, &data, &v3c.y_transpose_idx, &x, &lam_eq, cost_mult, &mut vals52);
        v5_2_kernel::fill_constraint_columns(&v5_sym, &data, &v3c.y_transpose_idx, &gens_at_bus, &x, &mut vals52);
        v5_2_kernel::fill_branch_hessian(&v5_sym, &data, &x, &mu_ineq, &z_ineq, &mut vals52);

        // --- 3. Evaluate V5.3 ---
        let mut vals53 = vec![0.0; v53_sym.base.row_idx.len()];
        v5_3_kernel::assemble_kkt_v5_3(&v53_sym, &data, &v3c.y_transpose_idx, &x, &lam_eq, &mu_ineq, &z_ineq, cost_mult, &mut vals53);

        let compare = |label: &str, candidate: &[f64]| {
            let mut max_diff: f64 = 0.0;
            let mut diff_count = 0;
            for i in 0..ref_vals.len() {
                let d = (ref_vals[i] - candidate[i]).abs();
                if d > 1e-11 {
                    if diff_count < 5 {
                        println!("{} Diff at nnz {}: ref={:.6e}, cand={:.6e} (err={:.3e})", label, i, ref_vals[i], candidate[i], d);
                    }
                    max_diff = max_diff.max(d);
                    diff_count += 1;
                }
            }
            println!("{} vs V4 KKT: diff_count={}, max_diff={:.3e}", label, diff_count, max_diff);
        };

        compare("V5.2", &vals52);
        compare("V5.3", &vals53);
    }

    #[test]
    fn test_v5_versions_vs_v4_lxx_only() {
        let net = crate::io::pandapower::load_csv_zip(&format!("{}/cases/IEEE118/data.zip", std::env::var("CARGO_MANIFEST_DIR").unwrap())).unwrap();
        let mut base_data = opf_data_from_network(&net);
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
        let nb = data.nb;
        let x = data.warm_x0();
        let lam_eq = vec![0.1; 2 * data.nb];
        let mu_ineq = vec![0.05; 2 * data.nl];
        let z_ineq = vec![0.7; 2 * data.nl];
        let cost_mult = 1e-4;

        let v3c = crate::new_opf::v3_symbolic::V3SymbolicCache::analyze(&data);
        let v5_sym = crate::new_opf::v5_kkt::KKTSymbolicV5::build(&data);
        let v53_sym = v5_3_kernel::KKTSymbolicV5_3::build(&data);

        // --- 1. Baseline: V4 Lxx only ---
        let lxx_v4 = crate::new_opf::v4_numeric_rect::v4_rect_numeric_fill(
            &data, &v3c, x.as_slice(), &lam_eq, &mu_ineq, Some(&z_ineq), cost_mult,
        );
        let ref_lxx = lxx_v4.values();

        // --- 2. Extract Lxx from V5.2 KKT ---
        let mut vals52 = vec![0.0; v5_sym.row_idx.len()];
        v5_2_kernel::fill_variable_columns(&v5_sym, &data, &v3c.y_transpose_idx, &x, &lam_eq, cost_mult, &mut vals52);
        v5_2_kernel::fill_branch_hessian(&v5_sym, &data, &x, &mu_ineq, &z_ineq, &mut vals52);
        
        // --- 3. Extract Lxx from V5.3 KKT ---
        let mut vals53 = vec![0.0; v53_sym.base.row_idx.len()];
        v5_3_kernel::assemble_kkt_v5_3(&v53_sym, &data, &v3c.y_transpose_idx, &x, &lam_eq, &mu_ineq, &z_ineq, cost_mult, &mut vals53);

        let compare_lxx = |label: &str, candidate_kkt: &[f64]| {
            let mut max_diff: f64 = 0.0;
            let mut diff_count = 0;
            let mut ref_idx = 0;
            // Variable columns (θ, Vm, Pg, Qg)
            for j in 0..data.nx() {
                let start = v5_sym.col_ptrs[j];
                let end = v5_sym.col_ptrs[j + 1];
                let deg = if j < 2*nb { (v3c.y_transpose_idx.len() / nb) } else { 0 }; // approximation
                // wait, the actual Lxx part in KKT column j is the first block
                let lxx_nnz = lxx_v4.col_offsets()[j + 1] - lxx_v4.col_offsets()[j];
                for off in 0..lxx_nnz {
                    let v_ref = ref_lxx[lxx_v4.col_offsets()[j] + off];
                    let v_cand = candidate_kkt[start + off];
                    let d = (v_ref - v_cand).abs();
                    if d > 1e-11 {
                        if diff_count < 5 {
                            println!("{} Lxx Diff at col {}, nnz {}: ref={:.6e}, cand={:.6e} (err={:.3e})", label, j, off, v_ref, v_cand, d);
                        }
                        max_diff = max_diff.max(d);
                        diff_count += 1;
                    }
                }
            }
            println!("{} vs V4 Lxx: diff_count={}, max_diff={:.3e}", label, diff_count, max_diff);
        };

        compare_lxx("V5.2", &vals52);
        compare_lxx("V5.3", &vals53);
    }

    #[test]
    fn test_v5_3_pips_ieee118() {
        let net = crate::io::pandapower::load_csv_zip(&format!("{}/cases/IEEE118/data.zip", std::env::var("CARGO_MANIFEST_DIR").unwrap())).unwrap();
        let mut base_data = opf_data_from_network(&net);
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
        let opt = PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() };
        let res53 = pips::pips_v5_3(&data, x0, xmin, xmax, opt);
        println!("V5.3: converged={} iter={} f={:.6}", res53.converged, res53.iterations, res53.f);
        assert!(res53.converged, "V5.3 should converge");
        assert!((res53.f - 129662.9725).abs() < 1e-2, "V5.3 result mismatch: {}", res53.f);
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
            |x, l, m, _z, c| crate::opf::hessian::opf_hessfcn(&base_data, x, l, m, c),
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
        let lam_eq = vec![0.1; 2 * base_data.nb];
        let mu_ineq = vec![0.05; 2 * base_data.nl];
        let cost_mult = 1e-4;

        let h_v1 = crate::opf::hessian::opf_hessfcn(&base_data, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);
        let h_v3 = v3_numeric_scalar::v3_scalar_numeric_fill(&base_data, &v3_cache, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);
        let h_v4 = v4_numeric_rect::v4_rect_numeric_fill(&base_data, &v3_cache, x.as_slice(), &lam_eq, &mu_ineq, None, cost_mult);

        println!("Comparing V1 vs V4 Hessian ({} x {})", nx, nx);
        
        // V4 accuracy vs V3
        let mut diff_v4_v3 = 0;
        let mut max_err_v4_v3: f64 = 0.0;
        for col in 0..nx {
            let start = h_v3.col_offsets()[col];
            let end = h_v3.col_offsets()[col+1];
            for i in start..end {
                let val_v3 = h_v3.values()[i];
                let val_v4 = h_v4.values()[i];
                let err = (val_v3 - val_v4).abs();
                if err > 1e-8 {
                    diff_v4_v3 += 1;
                    max_err_v4_v3 = max_err_v4_v3.max(err);
                }
            }
        }

        println!("Total v4-vs-V3 diffs: {}, Max Err: {:.6e}", diff_v4_v3, max_err_v4_v3);
        assert!(diff_v4_v3 == 0, "V4 must be mathematically identical to V3");
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
        let h_v4 = v4_numeric_rect::v4_rect_numeric_fill(&base_data, &v3_cache, x.as_slice(), &lam_eq, &mu_ineq, None, cost_mult);

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
            let mut total = 0usize;
            for k in keys {
                let a = *m1.get(&k).unwrap_or(&0.0);
                let b = *m2.get(&k).unwrap_or(&0.0);
                let err = (a-b).abs();
                if err > 1e-7 {
                    let e = per_block.entry(block(k.0,k.1)).or_insert((0,0.0));
                    e.0 += 1; e.1 = e.1.max(err);
                    total += 1;
                }
            }
            println!("--- {} vs V1 ---", label);
            let mut blocks: Vec<_> = per_block.iter().collect();
            blocks.sort_by_key(|(k,_)| *k);
            for (blk, (cnt, mx)) in blocks {
                println!("  block {}: diffs={} maxErr={:.4e}", blk, cnt, mx);
            }
            total
        };
        let _ = compare("v4_rect", &to_map(&h_v4));

        // === Gold standard: finite-difference validaton ===
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

        let m_v4 = to_map(&h_v4);
        const ATOL: f64 = 0.5;
        const RTOL: f64 = 5e-4;
        let mut worst_v4 = (0usize, 0usize, 0.0f64, 0.0f64); 
        for (&(r, c), &val_v4) in m_v4.iter() {
            if r > c { continue; }
            if r >= 2 * nb && c >= 2 * nb { continue; } 
            let fd = d2(r, c);
            let tol = ATOL + RTOL * fd.abs();
            let (a4, q4) = ((fd - val_v4).abs(), (fd - val_v4).abs() / tol);
            if q4 > worst_v4.3 { worst_v4 = (r, c, a4, q4); }
        }
        println!("  FD gold standard for V4: worst ratio={:.2e} abs={:.4e} at ({},{})", worst_v4.3, worst_v4.2, worst_v4.0, worst_v4.1);
        assert!(worst_v4.3 < 1.0, "v4 Hessian disagrees with finite differences");
    }

    /// Ablation breakdown: end-to-end PIPS per version (V1 legacy / V4 / V5.0), reporting
    /// per-stage wall-clock (Hess / G-H / KKT / Solve) + overall. This is the data source
    /// for the "KKT assembly shrinks from a big slice to invisible" ablation figure.
    /// cargo test --release bench_ablation_breakdown -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_ablation_breakdown() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        println!("\n| Case | Version | Iter | Hess | G/H | KKT | Solve | Overall |");
        println!("|---|---|---|---|---|---|---|---|");
        for case in ["IEEE39", "IEEE118", "pegase9241"] {
            let path = format!("{}/cases/{}/data.zip", dir, case);
            if !std::path::Path::new(&path).exists() { continue; }
            let net = crate::io::pandapower::load_csv_zip(&path).unwrap();
            let mut base_data = opf_data_from_network(&net);
            if let Some(cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
                if case == "IEEE39" {
                    for g in 0..10i64 { if let Some(r) = cfg.get("gen", g) { base_data.cost_coeffs[g as usize] = [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur]; } }
                } else {
                    if let Some(r) = cfg.get("ext_grid", 0) { base_data.cost_coeffs[0] = [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur]; }
                    for g in 0..54i64 { if let Some(r) = cfg.get("gen", g) { base_data.cost_coeffs[(1 + g) as usize] = [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur]; } }
                }
            }
            let mi = if case == "pegase9241" { 30 } else { 150 };
            let x0 = base_data.warm_x0();
            let (xmin, xmax) = base_data.bounds();

            let row = |case: &str, ver: &str, r: &PipsResult, overall: std::time::Duration| {
                let t = &r.timing;
                println!("| {} | {} | {} | {:?} | {:?} | {:?} | {:?} | {:?} |",
                    case, ver, r.iterations, t.hess, t.gh, t.kkt, t.solve, overall);
            };

            // V1 legacy (opf_hessfcn, no merged slacks)
            let t0 = std::time::Instant::now();
            let r1 = crate::opf::pips::pips(
                |x| crate::opf::cost::opf_costfcn(&base_data, x),
                |x| { let (g,h,dg,dh) = crate::opf::constraints::opf_consfcn(&base_data, x); (h,g,dh,dg) },
                |x,l,m,_z,c| crate::opf::hessian::opf_hessfcn(&base_data, x, l, m, c),
                x0.clone(), xmin.clone(), xmax.clone(),
                PipsOpt { max_it: mi, cost_mult: 1e-4, merged_slacks: false, ..Default::default() },
            );
            let d1 = t0.elapsed();
            row(case, "V1", &r1, d1);

            let data = NewOPFData::new(base_data.clone());
            let t0 = std::time::Instant::now();
            let r4 = pips(&data, x0.clone(), xmin.clone(), xmax.clone(), PipsOpt { max_it: mi, cost_mult: 1e-4, ..Default::default() });
            let d4 = t0.elapsed();
            row(case, "V4", &r4, d4);

            let t0 = std::time::Instant::now();
            let r5 = pips::pips_v5(&data, x0.clone(), xmin.clone(), xmax.clone(), PipsOpt { max_it: mi, cost_mult: 1e-4, ..Default::default() });
            let d5 = t0.elapsed();
            row(case, "V5.0", &r5, d5);

            let t0 = std::time::Instant::now();
            let r52 = pips::pips_v5_2(&data, x0.clone(), xmin.clone(), xmax.clone(), PipsOpt { max_it: mi, cost_mult: 1e-4, ..Default::default() });
            let d52 = t0.elapsed();
            row(case, "V5.2", &r52, d52);

            let t0 = std::time::Instant::now();
            let r53 = pips::pips_v5_3(&data, x0.clone(), xmin.clone(), xmax.clone(), PipsOpt { max_it: mi, cost_mult: 1e-4, ..Default::default() });
            let d53 = t0.elapsed();
            row(case, "V5.3", &r53, d53);
        }
    }

    #[test]
    #[ignore] // cargo test --release bench_v4_vs_v5_endtoend -- --ignored --nocapture
    fn bench_v4_vs_v5_endtoend() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        println!("\n| Case | Path | f [EUR] | Iter | Total time |");
        println!("|---|---|---|---|---|");
        for case in ["IEEE39", "IEEE118", "pegase9241"] {
            let path = format!("{}/cases/{}/data.zip", dir, case);
            if !std::path::Path::new(&path).exists() { continue; }
            let net = crate::io::pandapower::load_csv_zip(&path).unwrap();
            let mut base_data = opf_data_from_network(&net);
            if let Some(cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
                let ng_cfg = if case == "IEEE39" { 10 } else { 54 };
                if case != "IEEE39" {
                    if let Some(r) = cfg.get("ext_grid", 0) { base_data.cost_coeffs[0] = [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur]; }
                    for g in 0..ng_cfg { if let Some(r) = cfg.get("gen", g) { base_data.cost_coeffs[(1 + g) as usize] = [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur]; } }
                } else {
                    for g in 0..ng_cfg { if let Some(r) = cfg.get("gen", g) { base_data.cost_coeffs[g as usize] = [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur]; } }
                }
            }
            let data = NewOPFData::new(base_data);
            let x0 = data.warm_x0();
            let (xmin, xmax) = data.bounds();
            let mi = if case == "pegase9241" { 30 } else { 150 };

            let t4 = std::time::Instant::now();
            let r4 = pips(&data, x0.clone(), xmin.clone(), xmax.clone(), PipsOpt { max_it: mi, cost_mult: 1e-4, ..Default::default() });
            let d4 = t4.elapsed();
            let t5 = std::time::Instant::now();
            let r5 = pips::pips_v5(&data, x0.clone(), xmin.clone(), xmax.clone(), PipsOpt { max_it: mi, cost_mult: 1e-4, ..Default::default() });
            let d5 = t5.elapsed();
            println!("| {} | V4.0 | {:.2} | {} | {:?} |", case, r4.f, r4.iterations, d4);
            println!("| {} | V5.0 | {:.2} | {} | {:?} |", case, r5.f, r5.iterations, d5);
        }
    }

    #[test]
    fn test_v5_pips_matches_v4_ieee118() {
        let net = crate::io::pandapower::load_csv_zip(&format!("{}/cases/IEEE118/data.zip", std::env::var("CARGO_MANIFEST_DIR").unwrap())).unwrap();
        let mut base_data = opf_data_from_network(&net);
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
        let r4 = pips(&data, x0.clone(), xmin.clone(), xmax.clone(), PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() });
        let r5 = pips::pips_v5(&data, x0, xmin, xmax, PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() });

        println!("V4: converged={} iter={} f={:.6}", r4.converged, r4.iterations, r4.f);
        println!("V5: converged={} iter={} f={:.6}", r5.converged, r5.iterations, r5.f);
        assert!(r5.converged, "V5 must converge");
        assert_eq!(r4.iterations, r5.iterations, "V5 must take same iterations as V4");
        assert!((r4.f - r5.f).abs() < 1e-6, "V5 objective must match V4 (diff={:.3e})", (r4.f - r5.f).abs());
        let mut max_dx = 0.0f64;
        for (a, b) in r4.x.iter().zip(r5.x.iter()) { max_dx = max_dx.max((a - b).abs()); }
        println!("V5 vs V4 max |Δx| = {:.3e}", max_dx);
        assert!(max_dx < 1e-9, "V5 solution must match V4 (max|Δx|={:.3e})", max_dx);
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
                |x, l, m, _z, c| crate::opf::hessian::opf_hessfcn(&base_data, x, l, m, c),
                x0.clone(), xmin.clone(), xmax.clone(),
                PipsOpt { max_it: 150, cost_mult: 1e-4, merged_slacks: false, ..Default::default() },
            );
            let dur_v1 = start_v1.elapsed();

            let start_v3 = std::time::Instant::now();
            let res_v3 = pips(
                &data_v3,
                x0.clone(), xmin.clone(), xmax.clone(),
                PipsOpt { max_it: 150, cost_mult: 1e-4, merged_slacks: false, ..Default::default() },
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
                    v4_numeric_rect::v4_rect_numeric_fill(&data_v3, &v3_cache, x, lam_eq, mu_ineq, Some(z_ineq), cost_mult)
                },
                x0, xmin, xmax,
                PipsOpt { max_it: 150, cost_mult: 1e-4, merged_slacks: true, ..Default::default() },
            );
            let dur_v4 = start_v4.elapsed();

            let speedup3 = dur_v1.as_secs_f64() / dur_v3.as_secs_f64();
            let speedup4 = dur_v1.as_secs_f64() / dur_v4.as_secs_f64();
            println!("| {} | V1 | {:.2} | {} | {:?} | - |", case, res_v1.f, res_v1.iterations, dur_v1);
            println!("| {} | V3 | {:.2} | {} | {:?} | {:.2}x |", case, res_v3.f, res_v3.iterations, dur_v3, speedup3);
            println!("| {} | V4 | {:.2} | {} | {:?} | {:.2}x |", case, res_v4.f, res_v4.iterations, dur_v4, speedup4);
        }
    }
}
