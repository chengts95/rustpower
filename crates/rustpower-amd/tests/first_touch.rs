use rustpower_amd::{exact, order, powerflow::YbusPattern, Control};
#[test]
fn first_touch_keeps_scalar_amd_for_mixed_types_dense_and_sparse() {
    let mut rng = 31u64;
    let mut workspace = rustpower_amd::on_demand::Workspace::default();
    let mut control = rustpower_amd::on_demand::FullTableControl::default();
    for a in [5usize, 17, 64, 128] {
        for q in [0, a / 2, a] {
            for density in [5u64, 85] {
                let mut adj = vec![vec![false; a]; a];
                for b in 0..a {
                    adj[b][b] = true;
                    if b > 0 {
                        adj[b - 1][b] = true;
                        adj[b][b - 1] = true;
                    }
                }
                for b in 0..a {
                    for c in b + 1..a {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        if rng % 100 < density {
                            adj[b][c] = true;
                            adj[c][b] = true;
                        }
                    }
                }
                let mut p = vec![0];
                let mut rows = vec![];
                for col in &adj {
                    rows.extend(col.iter().enumerate().filter(|(_, v)| **v).map(|(i, _)| i));
                    p.push(rows.len());
                }
                rows.push(a);
                p.push(rows.len());
                let view = YbusPattern::analyze(&p, &rows, a - q, q).unwrap();
                let j = view.reduced_jacobian();
                for aggressive in [false, true] {
                    for dense in [-1., 10.] {
                        let ctl = Control { dense, aggressive };
                        let expected = order(
                            view.scalar_dimension(),
                            j.column_offsets(),
                            j.row_indices(),
                            &ctl,
                        )
                        .unwrap();
                        let lazy = workspace.order(&view, &ctl).unwrap();
                        let full = control.order(&view, &ctl);
                        assert_eq!(full.permutation, expected.0);
                        assert_eq!(lazy.counts.generated_neighbor_ids_saved, 0);
                        assert_eq!(lazy.permutation, expected.0);
                        assert_eq!(lazy.inverse, expected.1);
                        assert_eq!(lazy.info.lnz, expected.2.lnz);
                        for share in [false, true] {
                            let got = exact::order(&view, &ctl, share);
                            assert_eq!(got.permutation, expected.0);
                            assert_eq!(got.inverse, expected.1);
                            assert_eq!(got.info.lnz, expected.2.lnz);
                            assert_eq!(got.info.n_mult_subs_lu, expected.2.n_mult_subs_lu);
                        }
                        #[cfg(feature = "trace")]
                        {
                            let (_, original) = rustpower_amd::trace::capture(|| {
                                order(
                                    view.scalar_dimension(),
                                    j.column_offsets(),
                                    j.row_indices(),
                                    &ctl,
                                )
                                .unwrap()
                            });
                            let (_, fused) =
                                rustpower_amd::trace::capture(|| exact::order(&view, &ctl, true));
                            assert_eq!(original, fused);
                            let (_, lazy) = rustpower_amd::trace::capture(|| {
                                rustpower_amd::on_demand::order(&view, &ctl).unwrap()
                            });
                            assert_eq!(original, lazy);
                        }
                    }
                }
            }
        }
    }
}
