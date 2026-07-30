//! 增广矩阵 [B Jᵀ; J −I] 的数值填充，v3 架构：
//! 模式 = Ybus 偏移本身，无槽位图、无运行时查找。
//!
//! 数组布局（调用方分配，全部按 Ybus CSC 形状或压缩形状）：
//!   J  四象限  j_pe/j_pf        —— 与 active Ybus 同形（len = nnz）
//!              j_qe/j_qf        —— 压缩形（PQ 邻居段 + PV 自身槽）
//!   Jᵀ 四象限 jt_pe/jt_pf       —— 与 Ybus 同形，写入位置 = y_trans[p]
//!             jt_qe/jt_qf       —— 压缩形（PQ 列全度 / PV 列长度 1）
//!   B  四象限  b_ee/b_ef/b_fe/b_ff —— 与 Ybus 同形，B(i,j) 在边 (i,j) 自身位置写
//!
//! 一趟主循环写全部耦合项（每个槽位恰好写一次），随后对角修正循环
//! 写入 yv 项、PV 的 |V|² 行、B 对角。μ 不进主填充，走 apply_mu_delta。
//!
//! 约定：r = [P; Q/V²] 长度 2·n_act；yv 含 slack 作用（全量 Ybus·v），
//! slack 的物理从 yv 通道进入，模式中永远无 slack。

use crate::{Cx, YbusC};

pub struct AugPattern {
    pub n_act: usize,
    pub npq: usize,
    pub y_trans: Vec<u32>,    // 边 (i,j) → 镜像边 (j,i) 的 Ybus 偏移
    pub diag_off: Vec<u32>,   // 对角边在列内的局部偏移
    pub pq_cnt: Vec<u32>,     // 每列 PQ 邻居数（排序后天然为前缀）
    pub q_starts: Vec<u32>,   // J 的 Q 象限列段起点（len n_act+1）
    pub jtq_starts: Vec<u32>, // Jᵀ 的 Q 象限列段起点（len n_act+1）
    pub q_total: usize,
    pub jtq_total: usize,
}

impl AugPattern {
    pub fn build(ybus: &YbusC, npq: usize) -> Self {
        let n_act = ybus.nb;
        let nnz = ybus.ri.len();
        let mut y_trans = vec![0u32; nnz];
        let mut diag_off = vec![0u32; n_act];
        let mut pq_cnt = vec![0u32; n_act];
        let mut q_starts = vec![0u32; n_act + 1];
        let mut jtq_starts = vec![0u32; n_act + 1];
        for j in 0..n_act {
            let col = &ybus.ri[ybus.cp[j]..ybus.cp[j + 1]];
            let pc = col.partition_point(|&i| i < npq);
            pq_cnt[j] = pc as u32;
            for (t, &i) in col.iter().enumerate() {
                let p = ybus.cp[j] + t;
                if i == j {
                    diag_off[j] = t as u32;
                }
                let col_i = &ybus.ri[ybus.cp[i]..ybus.cp[i + 1]];
                y_trans[p] = (ybus.cp[i] + col_i.binary_search(&j).unwrap()) as u32;
            }
            // J 的 Q 象限：PQ 邻居 + PV 时自身一个槽
            q_starts[j + 1] = q_starts[j] + pc as u32 + if j >= npq { 1 } else { 0 };
            // Jᵀ 的 Q 象限：PQ 列全度，PV 列仅自身
            jtq_starts[j + 1] = jtq_starts[j] + if j < npq { col.len() as u32 } else { 1 };
        }
        AugPattern {
            n_act,
            npq,
            y_trans,
            diag_off,
            pq_cnt,
            q_starts: q_starts.clone(),
            jtq_starts: jtq_starts.clone(),
            q_total: q_starts[n_act] as usize,
            jtq_total: jtq_starts[n_act] as usize,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn fill_augmented(
    ybus: &YbusC,
    pat: &AugPattern,
    v: &[Cx],
    yv: &[Cx],
    r: &[f64],
    lam: &mut [Cx],
    j_pe: &mut [f64],
    j_pf: &mut [f64],
    j_qe: &mut [f64],
    j_qf: &mut [f64],
    jt_pe: &mut [f64],
    jt_pf: &mut [f64],
    jt_qe: &mut [f64],
    jt_qf: &mut [f64],
    b_ee: &mut [f64],
    b_ef: &mut [f64],
    b_fe: &mut [f64],
    b_ff: &mut [f64],
) {
    let n_act = pat.n_act;
    let npq = pat.npq;
    let (cp, ri, vals) = (&ybus.cp, &ybus.ri, &ybus.vals);

    for i in 0..n_act {
        lam[i] = if i < npq {
            Cx::new(r[i], r[n_act + i])
        } else {
            Cx::new(r[i], 0.0)
        };
    }

    // ---- 主循环：逐列逐边，耦合项一次写完 ----
    for j in 0..n_act {
        for t in 0..(cp[j + 1] - cp[j]) {
            let p = cp[j] + t;
            let i = ri[p];
            let y = vals[p];
            let (g, b) = (y.re, y.im);
            let (ei, fi) = (v[i].re, v[i].im);

            // J 块耦合部分（对角 yv 项在对角修正循环补）
            let dpe = ei * g + fi * b;
            let dpf = -ei * b + fi * g;
            let dqe = fi * g - ei * b;
            let dqf = -fi * b - ei * g;

            unsafe {
                *j_pe.get_unchecked_mut(p) = dpe;
                *j_pf.get_unchecked_mut(p) = dpf;

                // Jᵀ：转置位置由 y_trans 直接给出
                let tp = pat.y_trans[p] as usize;
                *jt_pe.get_unchecked_mut(tp) = dpe;
                *jt_pf.get_unchecked_mut(tp) = dpf;

                if i < npq {
                    // Q 行：J 侧压缩段内偏移即 t（PQ 邻居是前缀）
                    let q = *pat.q_starts.get_unchecked(j) as usize + t;
                    *j_qe.get_unchecked_mut(q) = dqe;
                    *j_qf.get_unchecked_mut(q) = dqf;
                    // Jᵀ 侧：列 i 全度，偏移 = 镜像边局部偏移
                    let tq = *pat.jtq_starts.get_unchecked(i) as usize + (tp - cp[i]);
                    *jt_qe.get_unchecked_mut(tq) = dqe;
                    *jt_qf.get_unchecked_mut(tq) = dqf;
                }

                if i != j {
                    // B(i,j) 块 = M(λ_i·Y_ij) + M(conj(λ_j·Y_ji))
                    let s = lam[i] * y + (lam[j] * vals[tp]).conj();
                    *b_ee.get_unchecked_mut(p) = s.re;
                    *b_ef.get_unchecked_mut(p) = -s.im;
                    *b_fe.get_unchecked_mut(p) = s.im;
                    *b_ff.get_unchecked_mut(p) = s.re;
                }
            }
        }
    }

    // ---- 对角修正：yv 项、PV 的 |V|² 行、B 对角 ----
    for k in 0..n_act {
        let dk = cp[k] + pat.diag_off[k] as usize;
        let (ak, bk) = (yv[k].re, yv[k].im);
        let (ek, fk) = (v[k].re, v[k].im);
        unsafe {
            *j_pe.get_unchecked_mut(dk) += ak;
            *j_pf.get_unchecked_mut(dk) += bk;
            *jt_pe.get_unchecked_mut(dk) += ak;
            *jt_pf.get_unchecked_mut(dk) += bk;

            let d = if k < npq {
                let q = *pat.q_starts.get_unchecked(k) as usize + pat.diag_off[k] as usize;
                *j_qe.get_unchecked_mut(q) += -bk;
                *j_qf.get_unchecked_mut(q) += ak;
                let tq = *pat.jtq_starts.get_unchecked(k) as usize + pat.diag_off[k] as usize;
                *jt_qe.get_unchecked_mut(tq) += -bk;
                *jt_qf.get_unchecked_mut(tq) += ak;
                2.0 * (vals[dk].re * r[k] - vals[dk].im * r[n_act + k])
            } else {
                // PV：|V|² 行梯度 (2e_k, 2f_k)，写在压缩段尾部自身槽
                let q = *pat.q_starts.get_unchecked(k) as usize + *pat.pq_cnt.get_unchecked(k) as usize;
                *j_qe.get_unchecked_mut(q) = 2.0 * ek;
                *j_qf.get_unchecked_mut(q) = 2.0 * fk;
                let tq = *pat.jtq_starts.get_unchecked(k) as usize;
                *jt_qe.get_unchecked_mut(tq) = 2.0 * ek;
                *jt_qf.get_unchecked_mut(tq) = 2.0 * fk;
                2.0 * vals[dk].re * r[k] + 2.0 * r[n_act + k]
            };
            *b_ee.get_unchecked_mut(dk) = d;
            *b_ef.get_unchecked_mut(dk) = 0.0;
            *b_fe.get_unchecked_mut(dk) = 0.0;
            *b_ff.get_unchecked_mut(dk) = d;
        }
    }
}

/// μ 增量更新（LM 内循环）：只写 B 的两个对角象限。
pub fn apply_mu_delta(ybus: &YbusC, pat: &AugPattern, b_ee: &mut [f64], b_ff: &mut [f64], dmu: f64) {
    for k in 0..pat.n_act {
        let dk = ybus.cp[k] + pat.diag_off[k] as usize;
        unsafe {
            *b_ee.get_unchecked_mut(dk) += dmu;
            *b_ff.get_unchecked_mut(dk) += dmu;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::YbusC;

    fn mismatch(ybus: &YbusC, npq: usize, v: &[Cx]) -> Vec<f64> {
        let nb = ybus.nb;
        let yv = ybus.spmv(v);
        let mut f = vec![0.0; 2 * nb];
        for k in 0..nb {
            let s = v[k] * yv[k].conj();
            f[k] = s.re;
            f[nb + k] = if k < npq { s.im } else { v[k].norm2() };
        }
        f
    }

    struct FillOut {
        j_pe: Vec<f64>, j_pf: Vec<f64>, j_qe: Vec<f64>, j_qf: Vec<f64>,
        b_ee: Vec<f64>, b_ef: Vec<f64>, b_fe: Vec<f64>, b_ff: Vec<f64>,
    }

    fn run_fill(ybus: &YbusC, pat: &AugPattern, v: &[Cx], r: &[f64]) -> FillOut {
        let nnz = ybus.ri.len();
        let yv = ybus.spmv(v);
        let mut lam = vec![Cx::default(); pat.n_act];
        let mut o = FillOut {
            j_pe: vec![0.0; nnz], j_pf: vec![0.0; nnz],
            j_qe: vec![0.0; pat.q_total], j_qf: vec![0.0; pat.q_total],
            b_ee: vec![0.0; nnz], b_ef: vec![0.0; nnz],
            b_fe: vec![0.0; nnz], b_ff: vec![0.0; nnz],
        };
        let (mut jt_pe, mut jt_pf) = (vec![0.0; nnz], vec![0.0; nnz]);
        let (mut jt_qe, mut jt_qf) = (vec![0.0; pat.jtq_total], vec![0.0; pat.jtq_total]);
        fill_augmented(
            ybus, pat, v, &yv, r, &mut lam,
            &mut o.j_pe, &mut o.j_pf, &mut o.j_qe, &mut o.j_qf,
            &mut jt_pe, &mut jt_pf, &mut jt_qe, &mut jt_qf,
            &mut o.b_ee, &mut o.b_ef, &mut o.b_fe, &mut o.b_ff,
        );
        o
    }

    /// 从象限数组复原稠密 J（或转置关系的验证在外层做）
    fn dense_j(ybus: &YbusC, pat: &AugPattern, o: &FillOut) -> Vec<f64> {
        let nb = pat.n_act;
        let npq = pat.npq;
        let n = 2 * nb;
        let mut jd = vec![0.0; n * n];
        for j in 0..nb {
            for t in 0..(ybus.cp[j + 1] - ybus.cp[j]) {
                let p = ybus.cp[j] + t;
                let i = ybus.ri[p];
                jd[i * n + j] = o.j_pe[p];
                jd[i * n + nb + j] = o.j_pf[p];
                if i < npq {
                    let q = pat.q_starts[j] as usize + t;
                    jd[(nb + i) * n + j] = o.j_qe[q];
                    jd[(nb + i) * n + nb + j] = o.j_qf[q];
                } else if i == j {
                    let q = pat.q_starts[j] as usize + pat.pq_cnt[j] as usize;
                    jd[(nb + i) * n + j] = o.j_qe[q];
                    jd[(nb + i) * n + nb + j] = o.j_qf[q];
                }
            }
        }
        jd
    }

    fn dense_b(ybus: &YbusC, pat: &AugPattern, o: &FillOut) -> Vec<f64> {
        let nb = pat.n_act;
        let n = 2 * nb;
        let mut bd = vec![0.0; n * n];
        for j in 0..nb {
            for t in 0..(ybus.cp[j + 1] - ybus.cp[j]) {
                let p = ybus.cp[j] + t;
                let i = ybus.ri[p];
                bd[i * n + j] = o.b_ee[p];
                bd[i * n + nb + j] = o.b_ef[p];
                bd[(nb + i) * n + j] = o.b_fe[p];
                bd[(nb + i) * n + nb + j] = o.b_ff[p];
            }
        }
        bd
    }

    #[test]
    fn j_and_b_against_finite_differences() {
        let nb = 3;
        let npq = 2;
        let branches = [(0, 1), (1, 2), (0, 2)];
        let ybus = YbusC::from_branches(nb, &branches, Cx::new(2.0, -7.0), 0.03);
        let pat = AugPattern::build(&ybus, npq);
        let n = 2 * nb;

        let v: Vec<Cx> = vec![
            Cx::new(1.02, -0.03),
            Cx::new(0.99, -0.08),
            Cx::new(1.01, -0.05),
        ];
        let r: Vec<f64> = (0..n).map(|k| 0.05 * k as f64 + 0.1).collect();

        let o = run_fill(&ybus, &pat, &v, &r);
        let jd = dense_j(&ybus, &pat, &o);
        let bd = dense_b(&ybus, &pat, &o);

        // J 对失配的有限差分
        let eps = 1e-7;
        let mut max_j = 0.0f64;
        let mut max_b = 0.0f64;
        for c in 0..n {
            let (mut vp, mut vm) = (v.clone(), v.clone());
            if c < nb {
                vp[c].re += eps;
                vm[c].re -= eps;
            } else {
                vp[c - nb].im += eps;
                vm[c - nb].im -= eps;
            }
            let fp = mismatch(&ybus, npq, &vp);
            let fm = mismatch(&ybus, npq, &vm);
            // H(r) 数值列：r · (J(x+) − J(x−)) / 2eps
            let jp = dense_j(&ybus, &pat, &run_fill(&ybus, &pat, &vp, &r));
            let jm = dense_j(&ybus, &pat, &run_fill(&ybus, &pat, &vm, &r));
            for row in 0..n {
                let num = (fp[row] - fm[row]) / (2.0 * eps);
                max_j = max_j.max((num - jd[row * n + c]).abs());
            }
            // H(m,c) = Σ_k r_k·∂J[k,m]/∂x_c
            for m in 0..n {
                let mut hn = 0.0;
                for k in 0..n {
                    hn += r[k] * (jp[k * n + m] - jm[k * n + m]);
                }
                max_b = max_b.max((hn / (2.0 * eps) - bd[m * n + c]).abs());
            }
        }
        assert!(max_j < 1e-5, "J 与有限差分不符: {}", max_j);
        assert!(max_b < 1e-4, "B 与 r 加权 J 差分不符: {}", max_b);

        // B 严格对称
        let mut max_sym = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                max_sym = max_sym.max((bd[i * n + j] - bd[j * n + i]).abs());
            }
        }
        assert!(max_sym < 1e-12, "B 不对称: {}", max_sym);
    }
}
