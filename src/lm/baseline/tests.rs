#[cfg(all(test, any(feature = "klu", feature = "klu_dyn")))]
mod tests {
    use crate::basic::ecs::elements::PPNetwork;
    use crate::basic::ecs::network::{DataOps, PowerFlow, PowerGrid};
    use crate::basic::ecs::powerflow::systems::PowerFlowMat;
    use crate::io::pandapower::{Network, load_csv_zip};
    use crate::lm::baseline::full_slice::AugFsDriver;
    use crate::lm::residual::fixtures::load_ieee39_mat;
    use nalgebra_sparse::{CooMatrix, CscMatrix};
    use num_complex::Complex64;

    #[test]
    fn coo_conversion_preserves_solver_buffers_until_pattern_changes() {
        use crate::lm::baseline::CooSystem;
        let mut system = CooSystem::default();
        let mut pointers = None;
        for (mu, j) in [(0.1, 2.0), (0.5, 3.0)] {
            let mut coo = CooMatrix::new(2, 2);
            for (row, col, value) in [(0, 0, mu), (1, 0, j), (0, 1, j), (1, 1, -1.0)] {
                coo.push(row, col, value);
            }
            system.update(&CscMatrix::from(&coo));
            let current = (system.cols.as_ptr(), system.rows.as_ptr(), system.values.as_ptr());
            if let Some(previous) = pointers {
                assert_eq!(current, previous);
                assert_eq!(system.solver.positive_inertia(), Some(1));
            }
            pointers = Some(current);
            let mut rhs = [0.0, -1.0];
            assert!(system.solve(&mut rhs));
            assert!((mu * rhs[0] + j * rhs[1]).abs() < 1e-12);
            assert!((j * rhs[0] - rhs[1] + 1.0).abs() < 1e-12);
        }
        let mut diagonal = CooMatrix::new(2, 2);
        diagonal.push(0, 0, 0.5);
        diagonal.push(1, 1, -1.0);
        system.update(&CscMatrix::from(&diagonal));
        assert_eq!(system.solver.positive_inertia(), None);
        let mut rhs = [1.0, -1.0];
        assert!(system.solve(&mut rhs));
        assert_eq!(rhs, [2.0, 1.0]);
    }

    struct Case {
        ybus: CscMatrix<Complex64>,
        npv: usize,
        npq: usize,
        sbus: Vec<Complex64>,
    }

    fn from_mat(mat: &PowerFlowMat) -> Case {
        Case {
            ybus: mat.y_bus.clone(),
            npv: mat.npv,
            npq: mat.npq,
            sbus: mat.s_bus.iter().copied().collect(),
        }
    }

    fn load_zip_case(_name: &'static str, path: &str) -> Option<Case> {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let net: Network = load_csv_zip(&format!("{dir}/{path}")).ok()?;
        let mut pf = PowerGrid::default();
        pf.world_mut().insert_resource(PPNetwork(net));
        pf.init_pf_net();
        let mat = pf.world().get_resource::<PowerFlowMat>().unwrap().clone();
        Some(from_mat(&mat))
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
                if m > 1e-12 {
                    v[i] / m
                } else {
                    Complex64::new(1.0, 0.0)
                }
            })
            .collect();
        let cache = &pat.cache;
        let mut j_ref = vec![0.0; pat.graph.nnz];
        fill_jacobian_v4::<false>(
            ybus,
            &v,
            &vnorm,
            &scalc,
            &pat.graph.col_starts,
            cache.pq_ends(),
            cache.active_ends(),
            cache.diag_ptrs(),
            npv,
            npq,
            &mut j_ref,
        );

        // AUG-FS full J, sliced into a reduced CSC for comparison.
        let mut fs = AugFsDriver::build(ybus, npv, npq, mat.s_bus.iter().copied().collect());
        let full = fs.full_j_coo_pub(ybus, &v);
        let mut red = vec![vec![0.0f64; n_state]; n_state];
        for k in 0..full.nnz() {
            let (fr, fc, fv) = (
                full.row_indices()[k],
                full.col_indices()[k],
                full.values()[k],
            );
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
                    assert!(
                        red[r][c].abs() < 1e-12,
                        "AUG-FS has an entry v4 lacks at ({r},{c})"
                    );
                }
            }
        }
        println!("AUG-FS vs v4 reduced-J max|Δ| = {max_diff:.3e}");
        assert!(max_diff < 1e-9, "AUG-FS full-J disagrees with v4 kernel");
    }

    /// Same cross-validation as `aug_fs_j_matches_v4` but on PEGASE9241 and
    /// sparse (dense would need 2.3 GB). Reports the worst entries so a
    /// systematic difference can be localized by (row, col) pattern.
    #[test]
    fn aug_fs_j_matches_v4_pegase9241() {
        use crate::basic::new_dsdvbus4::fill_jacobian_v4;
        use crate::lm::pattern::KktPattern;
        let Some(c) = load_zip_case("PEGASE9241", "cases/pegase9241/data.zip") else {
            println!("skipped (archive missing)");
            return;
        };
        let ybus = &c.ybus;
        let (npv, npq) = (c.npv, c.npq);
        let nb = ybus.ncols();
        let n_state = npv + 2 * npq;

        let pat = KktPattern::build(ybus, npv, npq);
        // Non-trivial voltage with real angle spread.
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
                if m > 1e-12 {
                    v[i] / m
                } else {
                    Complex64::new(1.0, 0.0)
                }
            })
            .collect();
        let cache = &pat.cache;
        let mut j_ref = vec![0.0; pat.graph.nnz];
        fill_jacobian_v4::<false>(
            ybus,
            &v,
            &vnorm,
            &scalc,
            &pat.graph.col_starts,
            cache.pq_ends(),
            cache.active_ends(),
            cache.diag_ptrs(),
            npv,
            npq,
            &mut j_ref,
        );

        let mut fs = AugFsDriver::build(ybus, npv, npq, c.sbus.clone());
        let full = fs.full_j_coo_pub(ybus, &v);
        // Filtered reduced triplets → CSC (sparse comparison).
        let mut red_coo = CooMatrix::new(n_state, n_state);
        for k in 0..full.nnz() {
            let (rr, cc) = (
                fs.map_row(full.row_indices()[k]),
                fs.map_col(full.col_indices()[k]),
            );
            if rr != usize::MAX && cc != usize::MAX {
                red_coo.push(rr, cc, full.values()[k]);
            }
        }
        let red = CscMatrix::from(&red_coo);

        let mut cs = pat.graph.col_starts.clone();
        cs.push(pat.graph.nnz);
        let mut worst: Vec<(f64, usize, usize, f64, f64)> = Vec::new();
        for cc in 0..n_state {
            let (mut p, mut q) = (cs[cc], red.col_offsets()[cc]);
            let (pe, qe) = (cs[cc + 1], red.col_offsets()[cc + 1]);
            while p < pe || q < qe {
                let rr_ref = if p < pe {
                    pat.graph.row_indices[p]
                } else {
                    usize::MAX
                };
                let rr_fs = if q < qe {
                    red.row_indices()[q]
                } else {
                    usize::MAX
                };
                if rr_ref == rr_fs {
                    let d = (j_ref[p] - red.values()[q]).abs();
                    if d > 1e-9 {
                        worst.push((d, rr_ref, cc, j_ref[p], red.values()[q]));
                    }
                    p += 1;
                    q += 1;
                } else if rr_ref < rr_fs {
                    worst.push((j_ref[p].abs(), rr_ref, cc, j_ref[p], 0.0));
                    p += 1;
                } else {
                    worst.push((red.values()[q].abs(), rr_fs, cc, 0.0, red.values()[q]));
                    q += 1;
                }
            }
        }
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!("PEGASE9241: entries with |Δ|>1e-9: {}", worst.len());
        for (d, r, cc, a, b) in worst.iter().take(10) {
            println!("  row={r} col={cc} ref={a:.6e} fs={b:.6e} |Δ|={d:.3e}");
        }
        assert!(
            worst.is_empty(),
            "AUG-FS full-J disagrees with v4 on PEGASE9241"
        );
    }
}
