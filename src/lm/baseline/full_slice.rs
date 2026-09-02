//! **AUG-FS** — the dumbest baseline: no permutation exploitation, no
//! reduced-layout awareness. The stranger computes the FULL `2n_bus × 2n_bus`
//! polar Jacobian (slack row/column, PV Q-rows, PV |V|-columns and all —
//! roughly twice the necessary derivative work), then extracts the reduced
//! quadrants by index-set slicing, then stacks `[μI Jᵀ; J −I]` as COO, then
//! pays the sort/convert and a fresh symbolic factorization per μ try.
//!
//! This is what the LM step looks like written by someone who knows power
//! flow but not sparse-matrix architecture — the MATLAB-style
//! `J_full` → `J_red = J_full(idx, idx)` → `A = [μI, J'; J, -I]` pipeline.
//! It exists to answer one question in the paper: how much of the gap is
//! placement strategy (also covered by AUG-COO) and how much is *not even
//! knowing which rows and columns you need*.
//!
//! The polar formulas below are the textbook ones (derivatives of
//! `S_i = |V_i| Σ_j |V_j|(G_ij + jB_ij) e^{jθ_ij}` w.r.t. `θ`, `|V|`),
//! implemented independently of the v4 kernel; the `fs_matches_v4` test
//! cross-validates the sliced result against `fill_jacobian_v4` bitwise-
//! neighbourhood, so the baseline doubles as an independent check of the
//! production kernel.

use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64;

use super::super::residual::residual;
use crate::basic::solver::{QDLDLSolver, Solve};

/// AUG-FS driver: full-J compute + slice + COO stack + fresh symbolic.
pub struct AugFsDriver {
    pub sbus: Vec<Complex64>,
    n_bus: usize,
    n_act: usize,
    npq: usize,
    n_state: usize,
    /// Reduced-system remap: full row/col index → reduced index (or
    /// `usize::MAX`). Rows: P of active buses (i), Q of PQ buses (nb + i);
    /// cols: θ of active (j), |V| of PQ (nb + j).
    row_map: Vec<usize>,
    col_map: Vec<usize>,
    /// Ybus transpose (CSC of Yᵀ = row-wise view of Y). Built once at
    /// symbolic time: the full-J walk needs `Y_ij` per *row* i, and our CSC
    /// only serves columns. Required for correctness on phase-shifter cases
    /// where Ybus is numerically asymmetric (PEGASE9241); walking column i
    /// as "row i" silently substitutes `Y_ji` there.
    ybus_t: CscMatrix<Complex64>,
    // Scratch.
    ibus: Vec<Complex64>,
    r: Vec<f64>,
    rt: Vec<f64>,
    g: Vec<f64>,
    b: Vec<f64>,
    vt: Vec<Complex64>,
    // Profiling (ns), same convention as the other baselines.
    /// Full-J computation (all quadrants, all buses — the 2× waste).
    pub prof_full_j_ns: u64,
    /// Slice + augmented COO stack + sort/convert.
    pub prof_slice_coo_ns: u64,
    /// Fresh-solver solves (symbolic + numeric + triangular, per μ try).
    pub prof_solve_ns: u64,
    pub n_solves: u64,
}

/// Outcome of one run (same shape as the other baselines').
pub struct AugFsResult {
    pub iterations: usize,
    pub converged: bool,
    pub res_inf: f64,
}

impl AugFsDriver {
    pub fn build(ybus: &CscMatrix<Complex64>, n_pv: usize, n_pq: usize, sbus: Vec<Complex64>) -> Self {
        let nb = ybus.ncols();
        let n_act = n_pv + n_pq;
        let n_state = n_act + n_pq;
        // Buses arrive as [PQ | PV | slack]; the stranger ignores this, but
        // slicing still needs the index sets — built here once (symbolic).
        let mut row_map = vec![usize::MAX; 2 * nb];
        let mut col_map = vec![usize::MAX; 2 * nb];
        for i in 0..n_act {
            row_map[i] = i; // P row of active bus
            col_map[i] = i; // θ col of active bus
        }
        for i in 0..n_pq {
            row_map[nb + i] = n_act + i; // Q row of PQ bus
            col_map[nb + i] = n_act + i; // |V| col of PQ bus
        }
        Self {
            sbus,
            n_bus: nb,
            n_act,
            npq: n_pq,
            n_state,
            row_map,
            col_map,
            ybus_t: ybus.transpose(),
            ibus: vec![Complex64::new(0.0, 0.0); nb],
            r: vec![0.0; n_state],
            rt: vec![0.0; n_state],
            g: vec![0.0; n_state],
            b: vec![0.0; 2 * n_state],
            vt: vec![Complex64::new(0.0, 0.0); nb],
            prof_full_j_ns: 0,
            prof_slice_coo_ns: 0,
            prof_solve_ns: 0,
            n_solves: 0,
        }
    }

    /// Test-only accessors for the cross-validation bench.
    #[cfg(test)]
    pub(crate) fn full_j_coo_pub(&mut self, ybus: &CscMatrix<Complex64>, v: &[Complex64]) -> CooMatrix<f64> {
        self.full_j_coo(ybus, v)
    }
    #[cfg(test)]
    pub(crate) fn map_row(&self, r: usize) -> usize {
        self.row_map[r]
    }
    #[cfg(test)]
    pub(crate) fn map_col(&self, c: usize) -> usize {
        self.col_map[c]
    }

    /// The full 2nb×2nb polar Jacobian as COO triplets — every quadrant of
    /// every bus, slack and PV waste included. Returns the raw triplets so
    /// the slicing stage is a separate, measurable pass (and testable).
    fn full_j_coo(&mut self, ybus: &CscMatrix<Complex64>, v: &[Complex64]) -> CooMatrix<f64> {
        let t = std::time::Instant::now();
        let nb = self.n_bus;
        // ibus/scalc for the diagonal terms (the stranger computes these too).
        for x in self.ibus.iter_mut() {
            *x = Complex64::new(0.0, 0.0);
        }
        for j in 0..nb {
            for p in ybus.col_offsets()[j]..ybus.col_offsets()[j + 1] {
                self.ibus[ybus.row_indices()[p]] += ybus.values()[p] * v[j];
            }
        }
        let mut coo = CooMatrix::new(2 * nb, 2 * nb);
        // Walk ROW i via the transpose: entries (i, j, Y_ij). (Walking
        // column i of Ybus would silently substitute Y_ji — wrong for
        // phase-shifter branches where Ybus is numerically asymmetric.)
        let (y_cp, y_ri, y_v) = (
            self.ybus_t.col_offsets(),
            self.ybus_t.row_indices(),
            self.ybus_t.values(),
        );
        for i in 0..nb {
            let (mi, thi) = v[i].to_polar();
            let si = v[i] * self.ibus[i].conj(); // P_i + jQ_i
            for p in y_cp[i]..y_cp[i + 1] {
                let j = y_ri[p];
                let y = y_v[p]; // G + jB
                let (mj, thj) = v[j].to_polar();
                if i != j {
                    let (sij, cij) = (thi - thj).sin_cos();
                    let mm = mi * mj;
                    // off-diagonal quadrant entries
                    let j11 = mm * (y.re * sij - y.im * cij);
                    let j21 = -mm * (y.re * cij + y.im * sij);
                    let j12 = mi * (y.re * cij + y.im * sij);
                    let j22 = mi * (y.re * sij - y.im * cij);
                    coo.push(i, j, j11);
                    coo.push(nb + i, j, j21);
                    coo.push(i, nb + j, j12);
                    coo.push(nb + i, nb + j, j22);
                } else {
                    let m2 = mi * mi;
                    coo.push(i, i, -si.im - y.im * m2); // -Q_i - B_ii |V_i|²
                    coo.push(nb + i, i, si.re - y.re * m2); //  P_i - G_ii |V_i|²
                    coo.push(i, nb + i, si.re / mi + y.re * mi); // P_i/|V_i| + G_ii |V_i|
                    coo.push(nb + i, nb + i, si.im / mi - y.im * mi); // Q_i/|V_i| - B_ii |V_i|
                }
            }
        }
        self.prof_full_j_ns += t.elapsed().as_nanos() as u64;
        coo
    }

    /// Slice the full J down to the reduced quadrants, then stack the
    /// augmented system as COO and pay the sort/convert. Jᵀ is "free" here
    /// too (swap the indices) — the sort collects the tax instead.
    fn slice_stack_convert(&mut self, full: &CooMatrix<f64>, mu: f64) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let t = std::time::Instant::now();
        let n = self.n_state;
        let (fr, fc, fv) = (full.row_indices(), full.col_indices(), full.values());
        let mut coo = CooMatrix::new(2 * n, 2 * n);
        for c in 0..n {
            coo.push(c, c, mu);
            coo.push(n + c, n + c, -1.0);
        }
        for k in 0..full.nnz() {
            let (rr, cc) = (self.row_map[fr[k]], self.col_map[fc[k]]);
            if rr != usize::MAX && cc != usize::MAX {
                let v = fv[k];
                coo.push(n + rr, cc, v); // J block
                coo.push(cc, n + rr, v); // Jᵀ block
            }
        }
        let csc = CscMatrix::from(&coo);
        let triple = (
            csc.col_offsets().to_vec(),
            csc.row_indices().to_vec(),
            csc.values().to_vec(),
        );
        self.prof_slice_coo_ns += t.elapsed().as_nanos() as u64;
        triple
    }

    pub fn solve_aug_fs(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
    ) -> AugFsResult {
        let n = self.n_state;
        let (n_act, npq) = (self.n_act, self.npq);
        let mut mu = 1e-2f64;
        let mut res_inf;
        for it in 0..maxit {
            let f;
            (res_inf, f) = residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
            if res_inf < tol {
                return AugFsResult { iterations: it, converged: true, res_inf };
            }
            let full = self.full_j_coo(ybus, v);

            // g = Jᵀ r from the full J (before slicing — the stranger has
            // nothing else). Only surviving entries contribute.
            {
                let (fr, fc, fv) = (full.row_indices(), full.col_indices(), full.values());
                for x in self.g.iter_mut() {
                    *x = 0.0;
                }
                for k in 0..full.nnz() {
                    let (rr, cc) = (self.row_map[fr[k]], self.col_map[fc[k]]);
                    if rr != usize::MAX && cc != usize::MAX {
                        self.g[cc] += fv[k] * self.r[rr];
                    }
                }
            }

            let mut accepted = false;
            for _ in 0..30 {
                let (mut ap, mut ai, mut ax) = self.slice_stack_convert(&full, mu);
                // fresh solver per try: symbolic + numeric from scratch
                let t = std::time::Instant::now();
                let mut solver = QDLDLSolver::default();
                self.b[..n].fill(0.0);
                for i in 0..n {
                    self.b[n + i] = -self.r[i];
                }
                let solve_ok = solver.solve(&mut ap, &mut ai, &mut ax, &mut self.b, 2 * n).is_ok();
                self.prof_solve_ns += t.elapsed().as_nanos() as u64;
                self.n_solves += 1;
                let delta: Vec<f64> = self.b[..n].to_vec();
                let finite = solve_ok && delta.iter().all(|x| x.is_finite());
                if !finite {
                    mu *= 10.0;
                    if mu > 1e12 {
                        return AugFsResult { iterations: it, converged: false, res_inf };
                    }
                    continue;
                }

                self.vt.copy_from_slice(v);
                for k in 0..n_act {
                    let mut mag = self.vt[k].norm();
                    let ang = self.vt[k].arg() + delta[k];
                    if k < npq {
                        mag += delta[n_act + k];
                    }
                    self.vt[k] = Complex64::from_polar(mag, ang);
                }
                if self.vt.iter().any(|x| !x.re.is_finite() || !x.im.is_finite()) {
                    mu *= 10.0;
                    if mu > 1e12 {
                        return AugFsResult { iterations: it, converged: false, res_inf };
                    }
                    continue;
                }

                let (_, f_new) = residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, &self.vt, &mut self.rt);
                let pred: f64 = -0.5
                    * self.g.iter().zip(delta.iter()).map(|(g, d)| g * d).sum::<f64>();
                let rho = if pred > 0.0 { (f - f_new) / pred } else { -1.0 };
                if std::env::var("RUSTPOWER_LM_DEBUG").is_ok() {
                    eprintln!("it={it} tryμ={mu:.3e} res={res_inf:.3e} f={f:.4e} f_new={f_new:.4e} pred={pred:.4e} ρ={rho:.4}");
                }
                if rho > 1e-4 {
                    v.copy_from_slice(&self.vt);
                    if rho > 0.75 {
                        mu = (mu / 3.0).max(1e-12);
                    }
                    accepted = true;
                    break;
                }
                mu *= 2.0;
                if mu > 1e12 {
                    return AugFsResult { iterations: it, converged: false, res_inf };
                }
            }
            if !accepted {
                return AugFsResult { iterations: it, converged: false, res_inf };
            }
        }
        let (res_inf, _) = residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
        AugFsResult { iterations: maxit, converged: res_inf < tol, res_inf }
    }
}
