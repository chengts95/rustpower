use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64;

/// All data needed for an AC-OPF formulation.
///
/// Follows the MATPOWER/pypower variable layout:
///   x = [Va (nb); Vm (nb); Pg (ng); Qg (ng)]
///
/// Equality constraints g(x) = 0:
///   P mismatch: Re(V·conj(Ybus·V)) - Cg·Pg + Pd = 0  for all buses
///   Q mismatch: Im(V·conj(Ybus·V)) - Cg·Qg + Qd = 0  for all buses
///
/// Inequality constraints h(x) ≤ 0:
///   Branch apparent-power limits: |Sf|² - Smax_f² ≤ 0, |St|² - Smax_t² ≤ 0
///   (variable bounds Pg/Qg/Vm enforced via xmin/xmax in PIPS)
#[derive(Clone)]
pub struct OPFData {
    // ── network ──────────────────────────────────────────────────────────────
    pub nb: usize, // number of buses
    pub nl: usize, // number of branches
    pub ng: usize, // number of generators

    pub ybus: CscMatrix<Complex64>, // nb × nb admittance matrix
    pub yf: CscMatrix<Complex64>,   // nl × nb "from-bus" admittance
    pub yt: CscMatrix<Complex64>,   // nl × nb "to-bus"  admittance
    pub f_buses: Vec<usize>,        // from-bus indices per branch
    pub t_buses: Vec<usize>,        // to-bus indices per branch

    // ── bus data ─────────────────────────────────────────────────────────────
    /// Net load injection (Pd + jQd) at each bus [p.u.].  Generators not included.
    pub s_load: DVector<Complex64>,
    /// Voltage magnitude lower bounds [p.u.]
    pub vm_min: Vec<f64>,
    /// Voltage magnitude upper bounds [p.u.]
    pub vm_max: Vec<f64>,
    /// Reference bus index (Va fixed to 0 by PIPS bounds)
    pub ref_bus: usize,

    // ── generator data ───────────────────────────────────────────────────────
    /// Bus index for each generator
    pub gen_bus: Vec<usize>,
    /// Cg: nb × ng incidence matrix (Cg[bus, gen] = 1)
    pub cg: CscMatrix<f64>,
    pub pg_min: Vec<f64>, // [p.u.]
    pub pg_max: Vec<f64>,
    pub qg_min: Vec<f64>,
    pub qg_max: Vec<f64>,
    /// Polynomial cost coefficients per generator: [c2, c1, c0] (cost in MW²)
    /// cost(Pg) = c2·Pg² + c1·Pg + c0   (Pg in p.u. × baseMVA = MW)
    pub cost_coeffs: Vec<[f64; 3]>,
    pub base_mva: f64,

    // ── branch limits ────────────────────────────────────────────────────────
    /// Maximum apparent power per branch [p.u.].  0 = unconstrained.
    pub rate_a: Vec<f64>,

    // ── warm-start data ──────────────────────────────────────────────────────
    /// Nominal Pg per generator [p.u.] from network data (used for NR warm start).
    /// Ext_grid entry is 0.0 (slack P is free).
    pub pg_init: Vec<f64>,
    /// Voltage setpoint per bus [p.u.]: vm_pu from ext_grid/gen, 1.0 for load buses.
    pub vm_set: Vec<f64>,

    // ── connection matrices (for Hessian) ────────────────────────────────────
    /// Cf: nb × nl, Cf[f[l], l] = 1
    pub cf: CscMatrix<Complex64>,
    /// Ct: nb × nl, Ct[t[l], l] = 1
    pub ct: CscMatrix<Complex64>,
}

impl OPFData {
    /// Variable dimension: 2·nb + 2·ng
    pub fn nx(&self) -> usize {
        2 * self.nb + 2 * self.ng
    }

    /// Index range for Va in x
    pub fn va_range(&self) -> std::ops::Range<usize> {
        0..self.nb
    }
    /// Index range for Vm in x
    pub fn vm_range(&self) -> std::ops::Range<usize> {
        self.nb..2 * self.nb
    }
    /// Index range for Pg in x
    pub fn pg_range(&self) -> std::ops::Range<usize> {
        2 * self.nb..2 * self.nb + self.ng
    }
    /// Index range for Qg in x
    pub fn qg_range(&self) -> std::ops::Range<usize> {
        2 * self.nb + self.ng..2 * self.nb + 2 * self.ng
    }

    /// Reconstruct complex voltage vector from x = [Va; Vm; ...]
    pub fn v_from_x(&self, x: &[f64]) -> DVector<Complex64> {
        DVector::from_iterator(self.nb, (0..self.nb).map(|i| {
            let va = x[i];
            let vm = x[self.nb + i];
            Complex64::from_polar(vm, va)
        }))
    }

    /// Compute net power injection Sbus = Cg·(Pg+jQg) - Sload
    pub fn sbus_from_x(&self, x: &[f64]) -> DVector<Complex64> {
        let mut s = -self.s_load.clone();
        let pg_off = 2 * self.nb;
        let qg_off = pg_off + self.ng;
        for g in 0..self.ng {
            let bus = self.gen_bus[g];
            s[bus] += Complex64::new(x[pg_off + g], x[qg_off + g]);
        }
        s
    }

    /// Branch flow limits squared [p.u.²], ∞ where rate_a == 0.
    pub fn flow_max_sq(&self) -> Vec<f64> {
        self.rate_a
            .iter()
            .map(|&r| if r == 0.0 { f64::INFINITY } else { r * r })
            .collect()
    }

    /// Build variable bounds vectors (xmin, xmax) for PIPS.
    pub fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let nx = self.nx();
        let mut xmin = vec![f64::NEG_INFINITY; nx];
        let mut xmax = vec![f64::INFINITY; nx];

        // Va: reference bus fixed
        xmin[self.ref_bus] = 0.0;
        xmax[self.ref_bus] = 0.0;

        // Vm bounds
        for i in 0..self.nb {
            xmin[self.nb + i] = self.vm_min[i];
            xmax[self.nb + i] = self.vm_max[i];
        }
        // Pg bounds
        for g in 0..self.ng {
            xmin[2 * self.nb + g] = self.pg_min[g];
            xmax[2 * self.nb + g] = self.pg_max[g];
        }
        // Qg bounds
        for g in 0..self.ng {
            xmin[2 * self.nb + self.ng + g] = self.qg_min[g];
            xmax[2 * self.nb + self.ng + g] = self.qg_max[g];
        }
        (xmin, xmax)
    }

    /// Flat starting point: Va=0, Vm=1, Pg=Pg_max/2, Qg=0.
    pub fn x0(&self) -> Vec<f64> {
        let mut x = vec![0.0f64; self.nx()];
        for i in 0..self.nb {
            x[self.nb + i] = 1.0; // Vm = 1 pu
        }
        for g in 0..self.ng {
            x[2 * self.nb + g] = 0.5 * (self.pg_min[g] + self.pg_max[g]);
        }
        x
    }

    /// Warm-start initial point: run NR power flow first, use Va/Vm as voltage seed.
    ///
    /// Mirrors the PYPOWER/pandapower approach: `runpp()` → use solution as x0.
    /// Falls back to the flat start if NR diverges (uses best-effort voltage).
    pub fn warm_x0(&self) -> Vec<f64> {
        use crate::basic::{newton_pf, solver::DefaultSolver};

        // Sbus = scheduled generation − load (slack bus contribution excluded)
        let mut sbus = -self.s_load.clone();
        for g in 0..self.ng {
            let b = self.gen_bus[g];
            if b != self.ref_bus {
                sbus[b] += Complex64::new(self.pg_init[g], 0.0);
            }
        }

        // Classify buses: 3=slack, 1=PV, 2=PQ
        let mut bus_type = vec![2u8; self.nb];
        bus_type[self.ref_bus] = 3;
        for &b in &self.gen_bus {
            if b != self.ref_bus {
                bus_type[b] = 1;
            }
        }

        // Build permutation order [PQ ... | PV ... | slack ...]
        let pq: Vec<usize> = (0..self.nb).filter(|&b| bus_type[b] == 2).collect();
        let pv: Vec<usize> = (0..self.nb).filter(|&b| bus_type[b] == 1).collect();
        let slk: Vec<usize> = (0..self.nb).filter(|&b| bus_type[b] == 3).collect();
        let npq = pq.len();
        let npv = pv.len();

        let mut perm = Vec::with_capacity(self.nb);
        perm.extend_from_slice(&pq);
        perm.extend_from_slice(&pv);
        perm.extend_from_slice(&slk);

        // inv_perm[orig] = permuted index
        let mut inv_perm = vec![0usize; self.nb];
        for (new_i, &orig) in perm.iter().enumerate() {
            inv_perm[orig] = new_i;
        }

        // Permute Ybus: Y_perm[inv_perm[i], inv_perm[j]] = Y[i, j]
        let ybus_p = {
            let mut coo: CooMatrix<Complex64> = CooMatrix::new(self.nb, self.nb);
            for j in 0..self.nb {
                for idx in self.ybus.col_offsets()[j]..self.ybus.col_offsets()[j + 1] {
                    let i = self.ybus.row_indices()[idx];
                    let v = self.ybus.values()[idx];
                    coo.push(inv_perm[i], inv_perm[j], v);
                }
            }
            CscMatrix::from(&coo)
        };

        let sbus_p = DVector::from_fn(self.nb, |i, _| sbus[perm[i]]);
        let v_init_p = DVector::from_fn(self.nb, |i, _| {
            Complex64::from_polar(self.vm_set[perm[i]], 0.0)
        });

        // Run Newton-Raphson power flow
        let mut solver = DefaultSolver::default();
        let v_p = match newton_pf(&ybus_p, &sbus_p, &v_init_p, npv, npq, None, None, &mut solver) {
            Ok((v, _)) => v,
            Err((_, v,_)) => v,
        };

        // Unpermute: original bus b is at permuted index inv_perm[b]
        let va: Vec<f64> = (0..self.nb).map(|b| v_p[inv_perm[b]].arg()).collect();
        let vm: Vec<f64> = (0..self.nb).map(|b| v_p[inv_perm[b]].norm()).collect();

        // Compute net P and Q injection per bus from PF result.
        // Using PF-derived Pg for ALL generators (including slack) avoids the
        // large P-mismatch that occurs when setting x0[Pg_slack] = pg_init[slack] = 0
        // while the NR PF assigns a non-zero slack generation.
        let v_orig = DVector::from_fn(self.nb, |b, _| v_p[inv_perm[b]]);
        let ibus = &self.ybus * &v_orig;
        let mut p_inj = vec![0.0f64; self.nb];
        let mut q_inj = vec![0.0f64; self.nb];
        let mut gen_count = vec![0u32; self.nb];
        for b in 0..self.nb {
            let s = v_orig[b] * ibus[b].conj();
            p_inj[b] = s.re + self.s_load[b].re; // = Pg - Pd + Pd = Pg
            q_inj[b] = s.im + self.s_load[b].im; // = Qg - Qd + Qd = Qg
        }
        for &b in &self.gen_bus {
            gen_count[b] += 1;
        }

        // Assemble x0 = [Va; Vm; Pg; Qg]
        let mut x0 = vec![0.0f64; self.nx()];
        for i in 0..self.nb {
            x0[i] = va[i];
            x0[self.nb + i] = vm[i];
        }
        for g in 0..self.ng {
            let b = self.gen_bus[g];
            let n = gen_count[b].max(1) as f64;
            x0[2 * self.nb + g] = p_inj[b] / n;
            x0[2 * self.nb + self.ng + g] = q_inj[b] / n;
        }

        // Clamp to variable bounds
        let (xmin, xmax) = self.bounds();
        for i in 0..self.nx() {
            if xmin[i].is_finite() { x0[i] = x0[i].max(xmin[i]); }
            if xmax[i].is_finite() { x0[i] = x0[i].min(xmax[i]); }
        }

        x0
    }
}
