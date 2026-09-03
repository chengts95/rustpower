use nalgebra::{Matrix2, Vector2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Output flow results for a two-port branch (Line or Trafo).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BranchFlowResult {
    pub p_from_mw: f64,
    pub q_from_mvar: f64,
    pub p_to_mw: f64,
    pub q_to_mvar: f64,
    pub pl_mw: f64,
    pub ql_mvar: f64,
    pub i_from_ka: f64,
    pub i_to_ka: f64,
    pub i_ka: f64,
    pub vm_from_pu: f64,
    pub va_from_degree: f64,
    pub vm_to_pu: f64,
    pub va_to_degree: f64,
    pub loading_percent: f64,
}

/// Static model of a 2-port branch using nalgebra Matrix2 for SIMD-accelerated math.
#[derive(Debug, Clone)]
pub struct Branch2Port {
    pub from_bus: usize,
    pub to_bus: usize,
    /// 2x2 admittance matrix in per-unit system (system base sn_mva):
    /// [ [Y_ff, Y_ft],
    ///   [Y_tf, Y_tt] ]
    pub y_mat: Matrix2<Complex64>,
    pub base_i_from_ka: f64,
    pub base_i_to_ka: f64,
    pub max_i_ka: f64,
    pub sn_trafo_mva: f64,
    pub in_service: bool,
}

impl Branch2Port {
    /// Compute power and current flows given full bus voltage vector in per-unit.
    #[inline(always)]
    pub fn compute_flow(&self, v: &[Complex64], sn_mva: f64) -> BranchFlowResult {
        if !self.in_service {
            return BranchFlowResult::default();
        }

        let vf = v[self.from_bus];
        let vt = v[self.to_bus];

        let v_vec = Vector2::new(vf, vt);
        // Guaranteed vectorised 2x2 complex matrix-vector multiplication via nalgebra
        let i_vec = self.y_mat * v_vec;
        let i_f = i_vec[0];
        let i_t = i_vec[1];

        // Complex power in MVA: S = V * conj(I) * S_base
        let sf = vf * i_f.conj() * sn_mva;
        let st = vt * i_t.conj() * sn_mva;

        let i_from_ka = i_f.norm() * self.base_i_from_ka;
        let i_to_ka = i_t.norm() * self.base_i_to_ka;
        let i_ka = i_from_ka.max(i_to_ka);

        let loading_percent = if self.max_i_ka > 0.0 {
            (i_ka / self.max_i_ka) * 100.0
        } else if self.sn_trafo_mva > 0.0 {
            let s_max = sf.norm().max(st.norm());
            (s_max / self.sn_trafo_mva) * 100.0
        } else {
            0.0
        };

        BranchFlowResult {
            p_from_mw: sf.re,
            q_from_mvar: sf.im,
            p_to_mw: st.re,
            q_to_mvar: st.im,
            pl_mw: sf.re + st.re,
            ql_mvar: sf.im + st.im,
            i_from_ka,
            i_to_ka,
            i_ka,
            vm_from_pu: vf.norm(),
            va_from_degree: vf.arg().to_degrees(),
            vm_to_pu: vt.norm(),
            va_to_degree: vt.arg().to_degrees(),
            loading_percent,
        }
    }
}

/// Compute flows for all branches sequentially.
pub fn compute_all_branch_flows(
    branches: &[Branch2Port],
    v: &[Complex64],
    sn_mva: f64,
    results: &mut [BranchFlowResult],
) {
    assert_eq!(branches.len(), results.len());
    for (b, res) in branches.iter().zip(results.iter_mut()) {
        *res = b.compute_flow(v, sn_mva);
    }
}
