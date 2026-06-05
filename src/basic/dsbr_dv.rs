use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

/// Partial derivatives of branch complex power flows w.r.t. voltage.
///
/// Returns (dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St).
/// Output matrices have the same sparsity structure as Yf and Yt respectively.
///
/// Formulae (MATPOWER TN2, §4):
///   If = Yf·V,  Sf = diag(Vf)·conj(If)    (Vf = V[f])
///   dSf/dVa[l,j] = j·( conj(If[l])·V[f[l]]·δ(j=f[l])  − Vf[l]·conj(Yf[l,j]·V[j]) )
///   dSf/dVm[l,j] = Vf[l]·conj(Yf[l,j]·Vnorm[j])  +  conj(If[l])·Vnorm[f[l]]·δ(j=f[l])
/// (dSt analogous with It, Vt = V[t], extra term at j = t[l])
#[allow(non_snake_case)]
pub fn dSbr_dV(
    Yf: &CscMatrix<Complex64>,
    Yt: &CscMatrix<Complex64>,
    f_buses: &[usize],
    t_buses: &[usize],
    v: &DVector<Complex64>,
    v_norm: &DVector<Complex64>,
) -> (
    CscMatrix<Complex64>, // dSf_dVa
    CscMatrix<Complex64>, // dSf_dVm
    CscMatrix<Complex64>, // dSt_dVa
    CscMatrix<Complex64>, // dSt_dVm
    DVector<Complex64>,   // Sf
    DVector<Complex64>,   // St
) {
    let nl = Yf.nrows();
    let nb = v.len();
    let j_unit = Complex64::i();

    let If: DVector<Complex64> = Yf * v;
    let It: DVector<Complex64> = Yt * v;

    let v_s = v.as_slice();
    let vn_s = v_norm.as_slice();
    let if_s = If.as_slice();
    let it_s = It.as_slice();

    // ── from-bus side ──────────────────────────────────────────────────────
    let nnzf = Yf.nnz();
    let yf_cp = Yf.col_offsets();
    let yf_ri = Yf.row_indices();
    let yf_v = Yf.values();
    let mut dsf_dva = vec![Complex64::new(0.0, 0.0); nnzf];
    let mut dsf_dvm = vec![Complex64::new(0.0, 0.0); nnzf];

    for j in 0..nb {
        for idx in yf_cp[j]..yf_cp[j + 1] {
            let l = yf_ri[idx];
            let y = yf_v[idx];
            let fl = f_buses[l];
            let vf = v_s[fl];
            let if_conj = if_s[l].conj();
            let yvj = y * v_s[j];

            if fl == j {
                dsf_dva[idx] = j_unit * (if_conj * vf - vf * yvj.conj());
                dsf_dvm[idx] = vf * (y * vn_s[j]).conj() + if_conj * vn_s[fl];
            } else {
                dsf_dva[idx] = j_unit * (-vf * yvj.conj());
                dsf_dvm[idx] = vf * (y * vn_s[j]).conj();
            }
        }
    }

    // ── to-bus side ────────────────────────────────────────────────────────
    let nnzt = Yt.nnz();
    let yt_cp = Yt.col_offsets();
    let yt_ri = Yt.row_indices();
    let yt_v = Yt.values();
    let mut dst_dva = vec![Complex64::new(0.0, 0.0); nnzt];
    let mut dst_dvm = vec![Complex64::new(0.0, 0.0); nnzt];

    for j in 0..nb {
        for idx in yt_cp[j]..yt_cp[j + 1] {
            let l = yt_ri[idx];
            let y = yt_v[idx];
            let tl = t_buses[l];
            let vt = v_s[tl];
            let it_conj = it_s[l].conj();
            let yvj = y * v_s[j];

            if tl == j {
                dst_dva[idx] = j_unit * (it_conj * vt - vt * yvj.conj());
                dst_dvm[idx] = vt * (y * vn_s[j]).conj() + it_conj * vn_s[tl];
            } else {
                dst_dva[idx] = j_unit * (-vt * yvj.conj());
                dst_dvm[idx] = vt * (y * vn_s[j]).conj();
            }
        }
    }

    let sf = DVector::from_iterator(nl, (0..nl).map(|l| v_s[f_buses[l]] * if_s[l].conj()));
    let st = DVector::from_iterator(nl, (0..nl).map(|l| v_s[t_buses[l]] * it_s[l].conj()));

    (
        CscMatrix::try_from_csc_data(nl, nb, yf_cp.to_vec(), yf_ri.to_vec(), dsf_dva).unwrap(),
        CscMatrix::try_from_csc_data(nl, nb, yf_cp.to_vec(), yf_ri.to_vec(), dsf_dvm).unwrap(),
        CscMatrix::try_from_csc_data(nl, nb, yt_cp.to_vec(), yt_ri.to_vec(), dst_dva).unwrap(),
        CscMatrix::try_from_csc_data(nl, nb, yt_cp.to_vec(), yt_ri.to_vec(), dst_dvm).unwrap(),
        sf,
        st,
    )
}

/// Partial derivatives of squared apparent power flow magnitudes w.r.t. voltage.
///
/// Returns (dAf_dVa, dAf_dVm, dAt_dVa, dAt_dVm) as real-valued CSC matrices.
///
/// Af = |Sf|² = Re(Sf)² + Im(Sf)²
///   dAf/dVa[l,j] = 2·Re(Sf[l])·Re(dSf/dVa[l,j]) + 2·Im(Sf[l])·Im(dSf/dVa[l,j])
#[allow(non_snake_case)]
pub fn dAbr_dV(
    dSf_dVa: &CscMatrix<Complex64>,
    dSf_dVm: &CscMatrix<Complex64>,
    dSt_dVa: &CscMatrix<Complex64>,
    dSt_dVm: &CscMatrix<Complex64>,
    Sf: &DVector<Complex64>,
    St: &DVector<Complex64>,
) -> (
    CscMatrix<f64>, // dAf_dVa
    CscMatrix<f64>, // dAf_dVm
    CscMatrix<f64>, // dAt_dVa
    CscMatrix<f64>, // dAt_dVm
) {
    let scale_row = |dS: &CscMatrix<Complex64>, S: &DVector<Complex64>| -> CscMatrix<f64> {
        let s = S.as_slice();
        let ri = dS.row_indices();
        let vals: Vec<f64> = dS
            .values()
            .iter()
            .enumerate()
            .map(|(idx, &ds)| 2.0 * s[ri[idx]].re * ds.re + 2.0 * s[ri[idx]].im * ds.im)
            .collect();
        CscMatrix::try_from_csc_data(
            dS.nrows(),
            dS.ncols(),
            dS.col_offsets().to_vec(),
            dS.row_indices().to_vec(),
            vals,
        )
        .unwrap()
    };

    (
        scale_row(dSf_dVa, Sf),
        scale_row(dSf_dVm, Sf),
        scale_row(dSt_dVa, St),
        scale_row(dSt_dVm, St),
    )
}
