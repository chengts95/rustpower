use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64;
use crate::io::pandapower::Network;
use std::collections::HashMap;
use crate::opf::builder::{line_admittances, trafo_admittances};

/// Build Ybus using the expansion approach: Ybus = A^T * Y_prim * A
/// A is (2b x n) incidence matrix.
/// Y_prim is (2b x 2b) block diagonal matrix.
pub fn build_ybus_binary(net: &Network) -> (CscMatrix<Complex64>, CscMatrix<Complex64>, CscMatrix<Complex64>) {
    let nb = net.bus.len();
    let base_mva = net.sn_mva;
    let wbase = 2.0 * std::f64::consts::PI * net.f_hz;
    
    let bus_id_to_idx: HashMap<i64, usize> = net.bus.iter().enumerate().map(|(i, b)| (b.index, i)).collect();
    let vbase: Vec<f64> = net.bus.iter().map(|b| b.vn_kv).collect();

    let mut branches: Vec<(usize, usize, [Complex64; 4])> = Vec::new();

    // 1. Process Lines
    for line in net.line.as_deref().unwrap_or(&[]) {
        if !line.in_service { continue; }
        let f = bus_id_to_idx[&line.from_bus];
        let t = bus_id_to_idx[&line.to_bus];
        let (yff, yft, ytf, ytt, _) = line_admittances(line, vbase[f], base_mva, wbase);
        branches.push((f, t, [yff, yft, ytf, ytt]));
    }

    // 2. Process Transformers
    for trafo in net.trafo.as_deref().unwrap_or(&[]) {
        if !trafo.in_service { continue; }
        let f = bus_id_to_idx[&(trafo.hv_bus as i64)];
        let t = bus_id_to_idx[&(trafo.lv_bus as i64)];
        let (yff, yft, ytf, ytt, _) = trafo_admittances(trafo, base_mva);
        branches.push((f, t, [yff, yft, ytf, ytt]));
    }

    let nl = branches.len();
    
    // 3. Build A matrix (2nl x nb)
    let mut a_coo = CooMatrix::<Complex64>::new(2 * nl, nb);
    let one = Complex64::new(1.0, 0.0);
    for (l, &(f, t, _)) in branches.iter().enumerate() {
        a_coo.push(2 * l, f, one);
        a_coo.push(2 * l + 1, t, one);
    }
    let a_mat = CscMatrix::from(&a_coo);

    // 4. Build Y_prim matrix (2nl x 2nl) block-diagonal
    let mut y_prim_coo = CooMatrix::<Complex64>::new(2 * nl, 2 * nl);
    for (l, &(_, _, [yff, yft, ytf, ytt])) in branches.iter().enumerate() {
        y_prim_coo.push(2 * l,     2 * l,     yff);
        y_prim_coo.push(2 * l,     2 * l + 1, yft);
        y_prim_coo.push(2 * l + 1, 2 * l,     ytf);
        y_prim_coo.push(2 * l + 1, 2 * l + 1, ytt);
    }
    let y_prim = CscMatrix::from(&y_prim_coo);

    // 5. Compute Ybus = A^T * Y_prim * A
    // M = Y_prim * A (2nl x nb)
    let m_mat = &y_prim * &a_mat;
    let mut ybus = &a_mat.transpose() * &m_mat;

    // 6. Add Shunts
    for sh in net.shunt.as_deref().unwrap_or(&[]) {
        if !sh.in_service { continue; }
        if let Some(&idx) = bus_id_to_idx.get(&sh.bus) {
            let step = sh.step as f64;
            let g_pu = sh.p_mw * step / base_mva;
            let b_pu = sh.q_mvar * step / base_mva;
            let y_sh = Complex64::new(g_pu, b_pu);
            
            // Add to Ybus diagonal
            add_to_csc_diagonal(&mut ybus, idx, y_sh);
        }
    }

    // 7. Extract Yf and Yt (nl x nb)
    // Yf is even rows of M, Yt is odd rows of M
    let mut yf_coo = CooMatrix::<Complex64>::new(nl, nb);
    let mut yt_coo = CooMatrix::<Complex64>::new(nl, nb);
    
    let m_cp = m_mat.col_offsets();
    let m_ri = m_mat.row_indices();
    let m_v = m_mat.values();
    
    for j in 0..nb {
        for idx in m_cp[j]..m_cp[j+1] {
            let row = m_ri[idx];
            let val = m_v[idx];
            if row % 2 == 0 {
                yf_coo.push(row / 2, j, val);
            } else {
                yt_coo.push(row / 2, j, val);
            }
        }
    }

    (ybus, CscMatrix::from(&yf_coo), CscMatrix::from(&yt_coo))
}

fn add_to_csc_diagonal(mat: &mut CscMatrix<Complex64>, idx: usize, val: Complex64) {
    let start = mat.col_offsets()[idx];
    let end = mat.col_offsets()[idx + 1];
    for i in start..end {
        if mat.row_indices()[i] == idx {
            mat.values_mut()[i] += val;
            return;
        }
    }
    // If diagonal was zero (not in structure), this is bad for CSC but should not happen in Ybus
    panic!("Ybus missing diagonal for bus {}, cannot add shunt.", idx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use crate::io::pandapower::load_csv_zip;
    use crate::opf::builder::opf_data_from_network;

    fn load_ieee118() -> Network {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE118/data.zip", dir);
        load_csv_zip(&path).unwrap()
    }

    fn load_ieee39() -> Network {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE39/data.zip", dir);
        load_csv_zip(&path).unwrap()
    }

    #[test]
    fn test_ybus_binary_equivalence_118() {
        let net = load_ieee118();
        
        // Baseline
        let data = opf_data_from_network(&net);
        let ybus_base = data.ybus;

        // New Binary logic
        let (ybus_new, _, _) = build_ybus_binary(&net);

        assert_eq!(ybus_base.nrows(), ybus_new.nrows());
        assert_eq!(ybus_base.ncols(), ybus_new.ncols());
        // Note: NNZ might slightly differ if some terms are structurally zero but kept in one path
        // But the values MUST match.

        let mut max_diff = 0.0f64;
        
        // Compare using dense map or coordinate check for robustness
        use std::collections::HashMap;
        let mut base_map = HashMap::new();
        for j in 0..ybus_base.ncols() {
            for idx in ybus_base.col_offsets()[j]..ybus_base.col_offsets()[j+1] {
                base_map.insert((ybus_base.row_indices()[idx], j), ybus_base.values()[idx]);
            }
        }

        for j in 0..ybus_new.ncols() {
            for idx in ybus_new.col_offsets()[j]..ybus_new.col_offsets()[j+1] {
                let r = ybus_new.row_indices()[idx];
                let v_new = ybus_new.values()[idx];
                let v_base = base_map.get(&(r, j)).cloned().unwrap_or(Complex64::new(0.0, 0.0));
                max_diff = max_diff.max((v_base - v_new).norm());
            }
        }

        println!("IEEE 118 Ybus Max Diff: {:.2e}", max_diff);
        assert!(max_diff < 1e-12, "Ybus values differ too much on IEEE 118!");
    }

    #[test]
    fn compare_new_pf_performance() {
        use crate::new_pf::solver;
        
        let net = load_ieee39();
        
        // 1. Setup Old Path
        let base_data = opf_data_from_network(&net);
        let ybus_old = base_data.ybus.clone();
        
        let mut sbus_old_vec = base_data.s_load.map(|e| -e);
        for g in 0..base_data.ng {
            let b = base_data.gen_bus[g];
            sbus_old_vec[b] += Complex64::new(base_data.pg_init[g], 0.0);
        }
        
        let v_init_x = base_data.warm_x0();
        let v0_old = base_data.v_from_x(&v_init_x);
        
        let mut bus_type = vec![2u8; base_data.nb];
        bus_type[base_data.ref_bus] = 3;
        for &b in &base_data.gen_bus {
            if b != base_data.ref_bus { bus_type[b] = 1; }
        }
        let npq = (0..base_data.nb).filter(|&b| bus_type[b] == 2).count();
        let npv = (0..base_data.nb).filter(|&b| bus_type[b] == 1).count();

        let mut solver = crate::basic::solver::RSparseSolver::default();
        
        // Run Old for accuracy baseline
        let (v_final_old, _) = crate::basic::newtonpf::newton_pf(
            &ybus_old, &sbus_old_vec, &v0_old, 
            npv, npq, 
            Some(1e-8), Some(10), &mut solver
        ).expect("Old PF failed");

        // Warm up and then bench
        let start_old = std::time::Instant::now();
        for _ in 0..10 {
            let _ = crate::basic::newtonpf::newton_pf(
                &ybus_old, &sbus_old_vec, &v0_old, 
                npv, npq, 
                Some(1e-8), Some(10), &mut solver
            );
        }
        let duration_old = start_old.elapsed() / 10;

        // 2. Setup New Path
        let (ybus_new, _, _) = build_ybus_binary(&net);
        
        let (v_final_new, _) = solver::run_newton_pf(
            &ybus_new, &sbus_old_vec, &v0_old,
            npv, npq,
            &mut solver, 10, 1e-8
        ).expect("New PF failed");
        
        let start_new = std::time::Instant::now();
        for _ in 0..10 {
            let _ = solver::run_newton_pf(
                &ybus_new, &sbus_old_vec, &v0_old,
                npv, npq,
                &mut solver, 10, 1e-8
            );
        }
        let duration_new = start_new.elapsed() / 10;

        // 3. Compare Results
        let mut max_err = 0.0f64;
        for i in 0..v_final_old.len() {
            let err = (v_final_old[i] - v_final_new[i]).norm();
            max_err = max_err.max(err);
        }

        println!("--- Performance Comparison (Release-like iteration) ---");
        println!("Old PF Path Avg: {:?}", duration_old);
        println!("New PF Path Avg: {:?}", duration_new);
        println!("Speedup: {:.2}x", duration_old.as_secs_f64() / duration_new.as_secs_f64());
        println!("Result Consistency (Max V Diff): {:.2e}", max_err);
        
        assert!(max_err < 1e-10, "New PF results diverged from baseline!");
    }
}



