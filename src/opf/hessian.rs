use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

use crate::basic::d2sbr_dv2::d2ASbr_dV2;
use crate::basic::d2sbus_dv2::d2Sbus_dV2;
use crate::basic::dsbr_dv::dSbr_dV;

use super::cost::opf_cost_d2f;
use super::problem::OPFData;

/// Hessian of the Lagrangian for AC-OPF.
///
/// L = cost_mult·f + lam_eq^T·g + mu_ineq^T·h
///
/// Returns Lxx as nb×nb ... actually (nx×nx) symmetric sparse matrix.
///
/// lam_eq  : 2·nb equality multipliers   [P_lam (nb), Q_lam (nb)]
/// mu_ineq : 2·nl inequality multipliers [mu_f (nl), mu_t (nl)]
#[allow(non_snake_case)]
pub fn opf_hessfcn(
    data: &OPFData,
    x: &[f64],
    lam_eq: &[f64],   // length 2·nb
    mu_ineq: &[f64],  // length 2·nl
    cost_mult: f64,
) -> CscMatrix<f64> {
    let nb = data.nb;
    let nl = data.nl;
    let ng = data.ng;
    let nx = data.nx();

    let v = data.v_from_x(x);
    let v_norm: DVector<Complex64> = v.map(|vi| vi / vi.norm());

    // ── cost Hessian ─────────────────────────────────────────────────────────
    // d²f/dPg² diagonal, placed at Pg rows/cols (2nb..2nb+ng)
    let d2f = opf_cost_d2f(data);

    // ── power balance Hessian (equality) ─────────────────────────────────────
    // lam_P = lam_eq[0..nb], lam_Q = lam_eq[nb..2nb]
    let lam_p = DVector::from_iterator(nb, lam_eq[..nb].iter().map(|&v| Complex64::new(v, 0.0)));
    let lam_q = DVector::from_iterator(nb, lam_eq[nb..].iter().map(|&v| Complex64::new(v, 0.0)));

    let (Gpaa, Gpav, Gpva, Gpvv) = d2Sbus_dV2(&data.ybus, &v, &lam_p);
    let (Gqaa, Gqav, Gqva, Gqvv) = d2Sbus_dV2(&data.ybus, &v, &lam_q);

    // d2G = Re(Gp_block) + Im(Gq_block)  (2nb × 2nb block)
    // where block = [[Gaa, Gav], [Gva, Gvv]]
    let d2G_aa = cx_block_to_real(&Gpaa, &Gqaa, false);
    let d2G_av = cx_block_to_real(&Gpav, &Gqav, false);
    let d2G_va = cx_block_to_real(&Gpva, &Gqva, false);
    let d2G_vv = cx_block_to_real(&Gpvv, &Gqvv, false);

    // ── branch flow Hessian (inequality) ─────────────────────────────────────
    let mu_f = &mu_ineq[..nl];
    let mu_t = &mu_ineq[nl..];

    let (dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St) =
        dSbr_dV(&data.yf, &data.yt, &data.f_buses, &data.t_buses, &v, &v_norm);

    let mu_f_vec = DVector::from_column_slice(mu_f);
    let mu_t_vec = DVector::from_column_slice(mu_t);

    let (Hfaa, Hfav, Hfva, Hfvv) =
        d2ASbr_dV2(&dSf_dVa, &dSf_dVm, &Sf, &data.cf, &data.yf, &v, &mu_f_vec);
    let (Htaa, Htav, Htva, Htvv) =
        d2ASbr_dV2(&dSt_dVa, &dSt_dVm, &St, &data.ct, &data.yt, &v, &mu_t_vec);

    // d2H = Hf_block + Ht_block
    let d2H_aa = f64_add(&Hfaa, &Htaa);
    let d2H_av = f64_add(&Hfav, &Htav);
    let d2H_va = f64_add(&Hfva, &Htva);
    let d2H_vv = f64_add(&Hfvv, &Htvv);

    // ── Assemble full Hessian (nx × nx) ──────────────────────────────────────
    // Layout:
    //       Va (0..nb)  Vm (nb..2nb)  Pg  Qg
    // Va  [ d2G_aa+d2H_aa  d2G_av+d2H_av   0   0 ]
    // Vm  [ d2G_va+d2H_va  d2G_vv+d2H_vv   0   0 ]
    // Pg  [      0              0           D2f  0 ]
    // Qg  [      0              0            0   0 ]
    //
    // plus cost_mult scaling on the cost part.

    let aa_block = f64_add(&d2G_aa, &d2H_aa);
    let av_block = f64_add(&d2G_av, &d2H_av);
    let va_block = f64_add(&d2G_va, &d2H_va);
    let vv_block = f64_add(&d2G_vv, &d2H_vv);

    assemble_hessian(nb, ng, nx, cost_mult, &d2f, &aa_block, &av_block, &va_block, &vv_block)
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Re(Gp_block) + Im(Gq_block) for complex sparse matrix pair.
/// If `imag_only` then Im(Gp) + Im(Gq), else Re(Gp) + Im(Gq).
fn cx_block_to_real(
    gp: &CscMatrix<Complex64>,
    gq: &CscMatrix<Complex64>,
    _imag_only: bool,
) -> CscMatrix<f64> {
    // Re(Gp) + Im(Gq), same sparsity (union)
    let nb = gp.ncols();
    let gp_cp = gp.col_offsets();
    let gp_ri = gp.row_indices();
    let gp_v = gp.values();
    let gq_cp = gq.col_offsets();
    let gq_ri = gq.row_indices();
    let gq_v = gq.values();

    let mut c_cp = vec![0usize; nb + 1];
    let mut c_ri: Vec<usize> = Vec::new();
    let mut c_v: Vec<f64> = Vec::new();

    for j in 0..nb {
        let mut ia = gp_cp[j];
        let mut ib = gq_cp[j];
        let ea = gp_cp[j + 1];
        let eb = gq_cp[j + 1];
        while ia < ea || ib < eb {
            let ra = if ia < ea { gp_ri[ia] } else { usize::MAX };
            let rb = if ib < eb { gq_ri[ib] } else { usize::MAX };
            if ra < rb {
                c_ri.push(ra);
                c_v.push(gp_v[ia].re);
                ia += 1;
            } else if rb < ra {
                c_ri.push(rb);
                c_v.push(gq_v[ib].im);
                ib += 1;
            } else {
                c_ri.push(ra);
                c_v.push(gp_v[ia].re + gq_v[ib].im);
                ia += 1;
                ib += 1;
            }
        }
        c_cp[j + 1] = c_ri.len();
    }

    CscMatrix::try_from_csc_data(nb, nb, c_cp, c_ri, c_v).unwrap()
}

fn f64_add(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    let nb = a.ncols();
    let a_cp = a.col_offsets();
    let a_ri = a.row_indices();
    let a_v = a.values();
    let b_cp = b.col_offsets();
    let b_ri = b.row_indices();
    let b_v = b.values();

    let mut c_cp = vec![0usize; nb + 1];
    let mut c_ri: Vec<usize> = Vec::new();
    let mut c_v: Vec<f64> = Vec::new();

    for j in 0..nb {
        let mut ia = a_cp[j];
        let mut ib = b_cp[j];
        let ea = a_cp[j + 1];
        let eb = b_cp[j + 1];
        while ia < ea || ib < eb {
            let ra = if ia < ea { a_ri[ia] } else { usize::MAX };
            let rb = if ib < eb { b_ri[ib] } else { usize::MAX };
            if ra < rb {
                c_ri.push(ra);
                c_v.push(a_v[ia]);
                ia += 1;
            } else if rb < ra {
                c_ri.push(rb);
                c_v.push(b_v[ib]);
                ib += 1;
            } else {
                c_ri.push(ra);
                c_v.push(a_v[ia] + b_v[ib]);
                ia += 1;
                ib += 1;
            }
        }
        c_cp[j + 1] = c_ri.len();
    }

    CscMatrix::try_from_csc_data(a.nrows(), nb, c_cp, c_ri, c_v).unwrap()
}

/// Assemble the full (nx × nx) Hessian from 2×2 blocks + cost diagonal.
fn assemble_hessian(
    nb: usize,
    ng: usize,
    nx: usize,
    cost_mult: f64,
    d2f: &[f64],        // ng cost diagonals
    aa: &CscMatrix<f64>, // nb × nb
    av: &CscMatrix<f64>, // nb × nb
    va: &CscMatrix<f64>, // nb × nb
    vv: &CscMatrix<f64>, // nb × nb
) -> CscMatrix<f64> {
    // Build column by column.
    // Col j ∈ [0..nb]:         Va block → aa col j, va col j
    // Col j ∈ [nb..2nb]:       Vm block → av col (j-nb), vv col (j-nb)
    // Col j ∈ [2nb..2nb+ng]:   Pg → cost diagonal
    // Col j ∈ [2nb+ng..nx]:    Qg → zeros

    let mut c_cp = vec![0usize; nx + 1];
    let mut c_ri: Vec<usize> = Vec::new();
    let mut c_v: Vec<f64> = Vec::new();

    // Va columns (0..nb): use aa and va
    for j in 0..nb {
        // aa col j → rows 0..nb
        for idx in aa.col_offsets()[j]..aa.col_offsets()[j + 1] {
            c_ri.push(aa.row_indices()[idx]);          // row ∈ 0..nb
            c_v.push(aa.values()[idx]);
        }
        // va col j → rows nb..2nb
        for idx in va.col_offsets()[j]..va.col_offsets()[j + 1] {
            c_ri.push(nb + va.row_indices()[idx]);     // row ∈ nb..2nb
            c_v.push(va.values()[idx]);
        }
        c_cp[j + 1] = c_ri.len();
    }

    // Vm columns (nb..2nb): use av and vv
    for j in 0..nb {
        let col = nb + j;
        for idx in av.col_offsets()[j]..av.col_offsets()[j + 1] {
            c_ri.push(av.row_indices()[idx]);           // row ∈ 0..nb
            c_v.push(av.values()[idx]);
        }
        for idx in vv.col_offsets()[j]..vv.col_offsets()[j + 1] {
            c_ri.push(nb + vv.row_indices()[idx]);      // row ∈ nb..2nb
            c_v.push(vv.values()[idx]);
        }
        c_cp[col + 1] = c_ri.len();
    }

    // Pg columns (2nb..2nb+ng): cost Hessian diagonal
    for g in 0..ng {
        let col = 2 * nb + g;
        let val = cost_mult * d2f[g];
        if val != 0.0 {
            c_ri.push(col);  // diagonal
            c_v.push(val);
        }
        c_cp[col + 1] = c_ri.len();
    }

    // Qg columns: all zero
    for g in 0..ng {
        let col = 2 * nb + ng + g;
        c_cp[col + 1] = c_ri.len();
    }

    CscMatrix::try_from_csc_data(nx, nx, c_cp, c_ri, c_v).unwrap()
}
