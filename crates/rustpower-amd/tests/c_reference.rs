// System C AMD is a test oracle only; the library itself has no C dependency.
use rustpower_amd::{order, order_unchecked, Control};
#[link(name = "amd")]
unsafe extern "C" {
    fn amd_l_order(
        n: i64,
        p: *const i64,
        i: *const i64,
        q: *mut i64,
        control: *const f64,
        info: *mut f64,
    ) -> i32;
}
fn compare(n: usize, p: &[usize], i: &[usize], ctl: &Control) {
    let cp: Vec<_> = p.iter().map(|&x| x as i64).collect();
    let ci: Vec<_> = i.iter().map(|&x| x as i64).collect();
    let mut cq = vec![0i64; n];
    let mut info = [0.; 20];
    let control = [ctl.dense, f64::from(ctl.aggressive), 0., 0., 0.];
    let status = unsafe {
        amd_l_order(
            n as i64,
            cp.as_ptr(),
            ci.as_ptr(),
            cq.as_mut_ptr(),
            control.as_ptr(),
            info.as_mut_ptr(),
        )
    };
    assert!(status >= 0);
    let a = order(n, p, i, ctl).unwrap();
    let b = unsafe { order_unchecked(n, p, i, ctl) }.unwrap();
    assert_eq!(a.0, b.0, "checked/unchecked n={n}");
    assert_eq!(
        a.0,
        cq.iter().map(|&x| x as usize).collect::<Vec<_>>(),
        "C permutation n={n},ctl={ctl:?}"
    );
    assert_eq!(a.2.lnz, info[9] as usize, "L estimate n={n}");
    for (k, &v) in a.0.iter().enumerate() {
        assert_eq!(a.1[v], k);
    }
}
#[test]
fn real_scalar_amd_matches_c_across_structures_and_controls() {
    let mut seed = 7u64;
    for &n in &[1usize, 2, 4, 7, 16, 31, 64, 127, 193] {
        for symmetric in [false, true] {
            for dense in [false, true] {
                let mut cols = vec![vec![]; n];
                for j in 0..n {
                    for i in 0..n {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        if i == j || seed % 100 < if dense { 50 } else { 7 } {
                            cols[j].push(i);
                            if symmetric {
                                cols[i].push(j);
                            }
                        }
                    }
                }
                let mut p = vec![0];
                let mut i = vec![];
                for c in &mut cols {
                    c.sort_unstable();
                    c.dedup();
                    i.extend_from_slice(c);
                    p.push(i.len());
                }
                for aggressive in [false, true] {
                    for alpha in [10., -1.] {
                        compare(
                            n,
                            &p,
                            &i,
                            &Control {
                                dense: alpha,
                                aggressive,
                            },
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn empty_and_jumbled_inputs_match_c() {
    compare(4, &[0, 0, 0, 0, 0], &[], &Control::default());
    compare(
        4,
        &[0, 3, 5, 7, 9],
        &[2, 0, 0, 3, 1, 0, 2, 3, 1],
        &Control::default(),
    );
    assert!(order(4, &[0, 10], &[], &Control::default()).is_err());
}
