pub mod symbolic;
pub mod numeric;
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

    fn load_ieee39() -> crate::io::pandapower::Network {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE39/data.zip", dir);
        load_csv_zip(&path).unwrap()
    }

    #[test]
    fn test_new_opf_ieee39_run() {
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
            |x| crate::opf::constraints::opf_consfcn(&base_data, x),
            |x, l, m, c| crate::opf::hessian::opf_hessfcn(&base_data, x, l, m, c),
            x0, xmin, xmax,
            PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() },
        );
        println!("Baseline OPF 39: converged={} iter={} f={:.6}", result.converged, result.iterations, result.f);
    }

    #[test]
    fn test_compare_hessian() {
        let net = load_ieee39();
        let base_data = opf_data_from_network(&net);
        let data = NewOPFData::new(base_data);

        let x = data.warm_x0();
        let nx = data.nx();
        let lam_eq = vec![0.1; 2 * data.nb];
        let mu_ineq = vec![0.05; 2 * data.nl];
        let cost_mult = 1.0;

        let h_base = crate::opf::hessian::opf_hessfcn(&data, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);
        let (h_new, dg_new) = numeric::numeric_fill(&data, &data.cache, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);

        let (_, _, dg_base, _) = crate::opf::constraints::opf_consfcn(&data, x.as_slice());

        // Compare Hessian
        assert_eq!(h_base.nrows(), h_new.nrows());
        assert_eq!(h_base.ncols(), h_new.ncols());
        
        let mut h_diff_count = 0;
        let mut h_max_diff = 0.0f64;

        // Compare Jacobian
        assert_eq!(dg_base.nrows(), dg_new.nrows());
        assert_eq!(dg_base.ncols(), dg_new.ncols());
        let mut g_diff_count = 0;
        let mut g_max_diff = 0.0f64;

        use nalgebra::DMatrix;
        let mut m_h_base = DMatrix::zeros(nx, nx);
        let mut m_h_new = DMatrix::zeros(nx, nx);
        let mut m_g_base = DMatrix::zeros(nx, 2 * data.nb);
        let mut m_g_new = DMatrix::zeros(nx, 2 * data.nb);

        for j in 0..nx {
            for idx in h_base.col_offsets()[j]..h_base.col_offsets()[j+1] {
                m_h_base[(h_base.row_indices()[idx], j)] = h_base.values()[idx];
            }
            for idx in h_new.col_offsets()[j]..h_new.col_offsets()[j+1] {
                m_h_new[(h_new.row_indices()[idx], j)] = h_new.values()[idx];
            }
        }

        for j in 0..2 * data.nb {
            for idx in dg_base.col_offsets()[j]..dg_base.col_offsets()[j+1] {
                m_g_base[(dg_base.row_indices()[idx], j)] = dg_base.values()[idx];
            }
            for idx in dg_new.col_offsets()[j]..dg_new.col_offsets()[j+1] {
                m_g_new[(dg_new.row_indices()[idx], j)] = dg_new.values()[idx];
            }
        }

        for i in 0..nx {
            for j in 0..nx {
                let d = (m_h_base[(i, j)] - m_h_new[(i, j)]).abs();
                if d > 1e-8 {
                    h_diff_count += 1;
                    h_max_diff = h_max_diff.max(d);
                }
            }
            for j in 0..2 * data.nb {
                let d = (m_g_base[(i, j)] - m_g_new[(i, j)]).abs();
                if d > 1e-8 {
                    g_diff_count += 1;
                    g_max_diff = g_max_diff.max(d);
                }
            }
        }

        println!("Hessian Comparison: diff_count={}, max_diff={}", h_diff_count, h_max_diff);
        println!("Jacobian Comparison: diff_count={}, max_diff={}", g_diff_count, g_max_diff);
        
        assert!(h_max_diff < 1e-8, "Hessian values differ too much!");
        assert!(g_max_diff < 1e-8, "Jacobian values differ too much!");
    }
}
