//! The Figure-1 experiment: **NE-COO / AUG-COO / AUG-SDF** on one table.
//!
//! All three paths solve the *same* GN-LM step with the *same* μ policy and
//! the *same* QDLDL backend; the only variables are assembly strategy and
//! symbolic reuse:
//!
//! | path | assembly | symbolic |
//! |---|---|---|
//! | AUG-FS   | full 2n_bus J (all quadrants, slack/PV waste) + slice + COO stack | fresh solver every μ try |
//! | NE-COO   | COO→CSC J, spgemm JᵀJ pattern+values every iteration (`dumb_mode`) | per outer iteration |
//! | AUG-COO  | COO push + sort/convert of `[μI Jᵀ; J −I]` every μ try | fresh solver every μ try |
//! | AUG-SDF  | direct CSC fill from Ybus offsets (`GnDriver`) | once, numeric-only after |
//!
//! Run (release, klu for the IEEE39 fixture loader):
//! ```text
//! cargo test --release --features klu lm_ablation -- --nocapture
//! ```

#[cfg(all(test, feature = "klu"))]
mod tests {
    use crate::basic::ecs::elements::PPNetwork;
    use crate::basic::ecs::network::{DataOps, PowerFlow, PowerGrid};
    use crate::basic::ecs::powerflow::systems::PowerFlowMat;
    use crate::basic::solver::QDLDLSolver;
    use crate::io::pandapower::{Network, load_csv_zip};
    use crate::lm::baseline::aug_coo::AugCooDriver;
    use crate::lm::baseline::full_slice::AugFsDriver;
    use crate::lm::gn_flat::GnDriver;
    use crate::lm::normal_eq::NeDriver;
    use crate::lm::residual::fixtures::load_ieee39_mat;
    use nalgebra_sparse::CscMatrix;
    use num_complex::Complex64;
    use std::time::Instant;

    struct Case {
        name: &'static str,
        ybus: CscMatrix<Complex64>,
        npv: usize,
        npq: usize,
        sbus: Vec<Complex64>,
        v_init: Vec<Complex64>,
    }

    fn from_mat(name: &'static str, mat: &PowerFlowMat) -> Case {
        Case {
            name,
            ybus: mat.y_bus.clone(),
            npv: mat.npv,
            npq: mat.npq,
            sbus: mat.s_bus.iter().copied().collect(),
            v_init: mat.v_bus_init.iter().copied().collect(),
        }
    }

    fn load_zip_case(name: &'static str, path: &str) -> Option<Case> {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let net: Network = load_csv_zip(&format!("{dir}/{path}")).ok()?;
        let mut pf = PowerGrid::default();
        pf.world_mut().insert_resource(PPNetwork(net));
        pf.init_pf_net();
        let mat = pf.world().get_resource::<PowerFlowMat>().unwrap().clone();
        Some(from_mat(name, &mat))
    }

    fn run_aug_sdf(c: &Case) {
        let mut d = GnDriver::build(&c.ybus, c.npv, c.npq, c.sbus.clone());
        let mut s = QDLDLSolver::default();
        let mut v = c.v_init.clone();
        let t = Instant::now();
        let res = d.solve_gn(&c.ybus, &mut s, &mut v, 1e-8, 100);
        let wall = t.elapsed();
        let its = res.iterations.max(1) as f64;
        println!(
            "  AUG-SDF | it={:<3} conv={:<5} wall={:>9.3?} | fill={:.3}ms ({:.1}µs/it) solve={:.3}ms ({:.1}µs/solve, {} solves)",
            res.iterations, res.converged, wall,
            d.prof_fill_ns as f64 / 1e6, d.prof_fill_ns as f64 / 1e3 / its,
            d.prof_solve_ns as f64 / 1e6, d.prof_solve_ns as f64 / 1e3 / d.n_solves.max(1) as f64,
            d.n_solves,
        );
        assert!(res.converged, "AUG-SDF must converge on {}", c.name);
    }

    fn run_aug_coo(c: &Case) {
        let mut d = AugCooDriver::build(&c.ybus, c.npv, c.npq, c.sbus.clone());
        let mut v = c.v_init.clone();
        let t = Instant::now();
        let res = d.solve_aug_coo(&c.ybus, &mut v, 1e-8, 100);
        let wall = t.elapsed();
        let its = res.iterations.max(1) as f64;
        println!(
            "  AUG-COO | it={:<3} conv={:<5} wall={:>9.3?} | fill={:.3}ms coo={:.3}ms ({:.1}µs/try) solve={:.3}ms ({:.1}µs/try, {} solves)",
            res.iterations, res.converged, wall,
            d.prof_fill_ns as f64 / 1e6,
            d.prof_coo_ns as f64 / 1e6, d.prof_coo_ns as f64 / 1e3 / d.n_solves.max(1) as f64,
            d.prof_solve_ns as f64 / 1e6, d.prof_solve_ns as f64 / 1e3 / d.n_solves.max(1) as f64,
            d.n_solves,
        );
        assert!(res.converged, "AUG-COO must converge on {}", c.name);
        let _ = its;
    }

    fn run_ne_coo(c: &Case) {
        let mut d = NeDriver::build(&c.ybus, c.npv, c.npq, c.sbus.clone());
        d.dumb_mode = true;
        let n_state = c.npv + 2 * c.npq; // n_act + npq
        let mut s = QDLDLSolver::with_dsigns(vec![1i8; n_state]);
        let mut v = c.v_init.clone();
        let t = Instant::now();
        let res = d.solve_ne(&c.ybus, &mut s, &mut v, 1e-8, 100);
        let wall = t.elapsed();
        let its = res.iterations.max(1) as f64;
        println!(
            "  NE-COO  | it={:<3} conv={:<5} wall={:>9.3?} | fill={:.3}ms spgemm={:.3}ms ({:.1}µs/it) numeric={:.3}ms",
            res.iterations, res.converged, wall,
            d.prof_fill_ns as f64 / 1e6,
            d.prof_spgemm_ns as f64 / 1e6, d.prof_spgemm_ns as f64 / 1e3 / its,
            d.prof_numeric_ns as f64 / 1e6,
        );
        assert!(res.converged, "NE-COO must converge on {}", c.name);
    }

    fn run_aug_fs(c: &Case) {
        let mut d = AugFsDriver::build(&c.ybus, c.npv, c.npq, c.sbus.clone());
        let mut v = c.v_init.clone();
        let t = Instant::now();
        let res = d.solve_aug_fs(&c.ybus, &mut v, 1e-8, 100);
        let wall = t.elapsed();
        println!(
            "  AUG-FS  | it={:<3} conv={:<5} wall={:>9.3?} | fullJ={:.3}ms ({:.1}µs/it) slice+coo={:.3}ms ({:.1}µs/try) solve={:.3}ms ({} solves)",
            res.iterations, res.converged, wall,
            d.prof_full_j_ns as f64 / 1e6, d.prof_full_j_ns as f64 / 1e3 / res.iterations.max(1) as f64,
            d.prof_slice_coo_ns as f64 / 1e6, d.prof_slice_coo_ns as f64 / 1e3 / d.n_solves.max(1) as f64,
            d.prof_solve_ns as f64 / 1e6,
            d.n_solves,
        );
        assert!(res.converged, "AUG-FS must converge on {}", c.name);
    }

    #[test]
    fn lm_ablation_three_way() {
        let mut cases = vec![from_mat("IEEE39", &load_ieee39_mat())];
        for (name, path) in [
            ("IEEE118", "cases/IEEE118/data.zip"),
            ("PEGASE9241", "cases/pegase9241/data.zip"),
        ] {
            match load_zip_case(name, path) {
                Some(c) => cases.push(c),
                None => println!("--- {name}: skipped (archive missing) ---"),
            }
        }
        for c in &cases {
            println!("=== {} (n_bus={}, npv={}, npq={}) ===", c.name, c.ybus.ncols(), c.npv, c.npq);
            run_aug_sdf(c);
            run_aug_fs(c);
            run_aug_coo(c);
            run_ne_coo(c);
        }
    }

    /// Cross-validation: the independently-written textbook full-J of AUG-FS,
    /// after slicing, must reproduce the production v4 kernel's reduced J.
    /// Runs on IEEE39 at the PF initial point; tolerates only fp-level noise
    /// (different summation algebra on the diagonal terms).
    #[test]
    fn aug_fs_j_matches_v4() {
        use crate::basic::new_dsdvbus4::fill_jacobian_v4;
        use crate::lm::pattern::KktPattern;
        let mat = load_ieee39_mat();
        let ybus = &mat.y_bus;
        let (npv, npq) = (mat.npv, mat.npq);
        let nb = ybus.ncols();
        let n_act = npv + npq;
        let n_state = n_act + npq;

        // Reference: production reduced J (block CSC layout) at a
        // NON-TRIVIAL voltage (large angles and spread magnitudes — flat
        // start alone would not exercise the sin/cos cross terms).
        let pat = KktPattern::build(ybus, npv, npq);
        let v: Vec<Complex64> = (0..nb)
            .map(|k| {
                let ang = 0.3 * (1.3 * k as f64).sin() - 0.02 * k as f64;
                let mag = 1.05 + 0.04 * (2.1 * k as f64).cos();
                Complex64::from_polar(mag, ang)
            })
            .collect();
        let mut ibus = vec![Complex64::new(0.0, 0.0); nb];
        for j in 0..nb {
            for p in ybus.col_offsets()[j]..ybus.col_offsets()[j + 1] {
                ibus[ybus.row_indices()[p]] += ybus.values()[p] * v[j];
            }
        }
        let scalc: Vec<Complex64> = (0..nb).map(|i| v[i] * ibus[i].conj()).collect();
        let vnorm: Vec<Complex64> = (0..nb)
            .map(|i| {
                let m = v[i].norm();
                if m > 1e-12 { v[i] / m } else { Complex64::new(1.0, 0.0) }
            })
            .collect();
        let cache = &pat.cache;
        let mut j_ref = vec![0.0; pat.graph.nnz];
        fill_jacobian_v4::<false>(
            ybus, &v, &vnorm, &scalc,
            &pat.graph.col_starts, cache.pq_ends(), cache.active_ends(), cache.diag_ptrs(),
            npv, npq, &mut j_ref,
        );

        // AUG-FS full J, sliced into a reduced CSC for comparison.
        let mut fs = AugFsDriver::build(ybus, npv, npq, mat.s_bus.iter().copied().collect());
        let full = fs.full_j_coo_pub(ybus, &v);
        let mut red = vec![vec![0.0f64; n_state]; n_state];
        for k in 0..full.nnz() {
            let (fr, fc, fv) = (full.row_indices()[k], full.col_indices()[k], full.values()[k]);
            let (rr, cc) = (fs.map_row(fr), fs.map_col(fc));
            if rr != usize::MAX && cc != usize::MAX {
                red[rr][cc] += fv;
            }
        }

        // Compare: reference column c has rows graph.col_rows(c) with values
        // j_ref[cs[c]..cs[c+1]].
        let mut max_diff = 0.0f64;
        let mut cs = pat.graph.col_starts.clone();
        cs.push(pat.graph.nnz);
        for c in 0..n_state {
            let mut seen = vec![false; n_state];
            for p in cs[c]..cs[c + 1] {
                let r = pat.graph.row_indices[p];
                seen[r] = true;
                max_diff = max_diff.max((j_ref[p] - red[r][c]).abs());
            }
            for r in 0..n_state {
                if !seen[r] {
                    assert!(red[r][c].abs() < 1e-12, "AUG-FS has an entry v4 lacks at ({r},{c})");
                }
            }
        }
        println!("AUG-FS vs v4 reduced-J max|Δ| = {max_diff:.3e}");
        assert!(max_diff < 1e-9, "AUG-FS full-J disagrees with v4 kernel");
    }
}
