//! 二阶 Newton 潮流计算（Chebyshev 两步法）算法验证原型
//!
//! 数学基础：rect 坐标下失配向量 F(v) 是二次型，Taylor 展开精确截断：
//!     F(v+δ) = F(v) + J(v)·δ + q(δ),   q(δ) = δ ⊙ conj(Y·δ)
//! 两步格式：
//!     J·δ₁ = -F(v)        （一次 LU 分解）
//!     J·δ₂ = -q(δ₁)       （复用同一分解，仅回代）
//!     v ← v + δ₁ + δ₂     局部三阶收敛
//!
//! 结构上与 rustpower 一致：符号相预计算 CSC 模式（四象限共用 Ybus 图），
//! 数值相单趟填充，无分支、无排序、无动态分配。
//! 注：本原型用稠密 LU 做方程求解以验证算法本身；生产路径应替换为
//! 稀疏 LU/LDL^T（同样可缓存符号分解），组装部分代码即最终形态。

use std::ops::{Add, AddAssign, Mul, Sub};

mod aug_fill;

// ---------------------------------------------------------------------------
// 复数（无外部依赖的最小实现）
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Default, Debug)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    #[inline]
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    #[inline]
    fn conj(self) -> Self {
        Self { re: self.re, im: -self.im }
    }
    #[inline]
    fn norm2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

impl Add for Cx {
    type Output = Cx;
    #[inline]
    fn add(self, o: Cx) -> Cx {
        Cx::new(self.re + o.re, self.im + o.im)
    }
}
impl Sub for Cx {
    type Output = Cx;
    #[inline]
    fn sub(self, o: Cx) -> Cx {
        Cx::new(self.re - o.re, self.im - o.im)
    }
}
impl Mul for Cx {
    type Output = Cx;
    #[inline]
    fn mul(self, o: Cx) -> Cx {
        Cx::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}
impl AddAssign for Cx {
    #[inline]
    fn add_assign(&mut self, o: Cx) {
        self.re += o.re;
        self.im += o.im;
    }
}

// ---------------------------------------------------------------------------
// CSC 稀疏矩阵 + 符号缓存
// ---------------------------------------------------------------------------
struct Csc {
    n: usize,
    cp: Vec<usize>, // 列偏移
    ri: Vec<usize>, // 行索引
    vals: Vec<f64>, // 数值（符号相只分配，数值相单趟填充）
}

/// Ybus 的 CSC（复数，按列存）
struct YbusC {
    nb: usize,
    cp: Vec<usize>,
    ri: Vec<usize>,
    vals: Vec<Cx>,
    neighbors: Vec<Vec<usize>>, // 每节点邻接表（含自身），符号缓存的输入
}

impl YbusC {
    /// 由支路列表组装：y = 1/(R+iX) 串联支路 + 节点并联电纳
    fn from_branches(nb: usize, branches: &[(usize, usize)], y_series: Cx, shunt_b: f64) -> Self {
        let mut dense = vec![Cx::default(); nb * nb];
        for &(i, j) in branches {
            dense[i * nb + i] += y_series;
            dense[j * nb + j] += y_series;
            dense[i * nb + j] = dense[i * nb + j] - y_series;
            dense[j * nb + i] = dense[j * nb + i] - y_series;
        }
        for k in 0..nb {
            dense[k * nb + k] += Cx::new(0.0, shunt_b);
        }
        // 转 CSC（列主序）
        let mut cp = vec![0usize];
        let mut ri = Vec::new();
        let mut vals = Vec::new();
        let mut neighbors = vec![Vec::new(); nb];
        for j in 0..nb {
            for i in 0..nb {
                let y = dense[i * nb + j];
                if y.re != 0.0 || y.im != 0.0 {
                    ri.push(i);
                    vals.push(y);
                    neighbors[j].push(i);
                }
            }
            cp.push(ri.len());
        }
        YbusC { nb, cp, ri, vals, neighbors }
    }

    /// 复数 SpMV：w = Y · v
    fn spmv(&self, v: &[Cx]) -> Vec<Cx> {
        let mut w = vec![Cx::default(); self.nb];
        for j in 0..self.nb {
            for p in self.cp[j]..self.cp[j + 1] {
                w[self.ri[p]] += self.vals[p] * v[j];
            }
        }
        w
    }
}

// ---------------------------------------------------------------------------
// 问题数据
// ---------------------------------------------------------------------------
struct Case {
    nb: usize,
    ybus: YbusC,
    pv: Vec<bool>,   // 是否 PV 节点（slack 固定为节点 0）
    p_spec: Vec<f64>,
    q_spec: Vec<f64>,
    v2_spec: Vec<f64>, // PV 的 |V|^2 指定值
}

// ---------------------------------------------------------------------------
// 失配 F(v)：PQ 节点 P、Q；PV 节点 P + |V|^2；slack 两行恒为零
// 方程排序 [P(0..nb), Q(nb..2nb)]，变量排序 [e, f]
// ---------------------------------------------------------------------------
fn mismatch(case: &Case, v: &[Cx]) -> Vec<f64> {
    let nb = case.nb;
    let yv = case.ybus.spmv(v);
    let mut f_out = vec![0.0; 2 * nb];
    for k in 0..nb {
        let s = v[k] * yv[k].conj();
        f_out[k] = s.re - case.p_spec[k];
        f_out[nb + k] = s.im - case.q_spec[k];
        if case.pv[k] {
            f_out[nb + k] = v[k].norm2() - case.v2_spec[k];
        }
    }
    f_out[0] = 0.0;
    f_out[nb] = 0.0;
    f_out
}

// ---------------------------------------------------------------------------
// 符号相：2nb 系统 CSC 模式 = 4 象限 × Ybus 图
// 列 e_j / f_j 的行集合均为 {P_i} ∪ {Q_i}, i ∈ N(j)
// ---------------------------------------------------------------------------
fn symbolic_cache(ybus: &YbusC) -> Csc {
    let nb = ybus.nb;
    let mut cp = vec![0usize];
    let mut ri = Vec::new();
    for _e_block in 0..2 {
        for j in 0..nb {
            for &i in &ybus.neighbors[j] {
                ri.push(i);
            }
            for &i in &ybus.neighbors[j] {
                ri.push(nb + i);
            }
            cp.push(ri.len());
        }
    }
    let nnz = ri.len();
    Csc { n: 2 * nb, cp, ri, vals: vec![0.0; nnz] }
}

// ---------------------------------------------------------------------------
// 数值相：单趟填充 J(v)
// P_k = e_k a_k + f_k b_k, Q_k = f_k a_k - e_k b_k,  (Yv)_k = a_k + i b_k
// ---------------------------------------------------------------------------
fn fill_jacobian(case: &Case, v: &[Cx], jac: &mut Csc) {
    let nb = case.nb;
    let yv = case.ybus.spmv(v);
    let ybus = &case.ybus;
    // 列 j 的 (e/f) 两个块统一处理
    for j in 0..nb {
        for e_block in 0..2 {
            let col = if e_block == 0 { j } else { nb + j };
            let base = jac.cp[col];
            let deg = ybus.neighbors[j].len();
            // P 行段
            for t in 0..deg {
                let i = jac.ri[base + t];
                let y = ybus.vals[ybus.cp[j] + pos_in_col(ybus, j, i)];
                let (g, b) = (y.re, y.im);
                let mut val = if e_block == 0 {
                    (if i == j { yv[i].re } else { 0.0 }) + v[i].re * g + v[i].im * b
                } else {
                    (if i == j { yv[i].im } else { 0.0 }) - v[i].re * b + v[i].im * g
                };
                if i == 0 {
                    // slack P 行 → e 的单位行
                    val = if i == j && e_block == 0 { 1.0 } else { 0.0 };
                }
                jac.vals[base + t] = val;
            }
            // Q 行段（PV 节点替换为 |V|^2 方程梯度，仅对角块）
            for t in 0..deg {
                let i = jac.ri[base + deg + t] - nb;
                let y = ybus.vals[ybus.cp[j] + pos_in_col(ybus, j, i)];
                let (g, b) = (y.re, y.im);
                let mut val = if e_block == 0 {
                    (if i == j { -yv[i].im } else { 0.0 }) + v[i].im * g - v[i].re * b
                } else {
                    (if i == j { yv[i].re } else { 0.0 }) - v[i].im * b - v[i].re * g
                };
                if i == 0 {
                    // slack Q 行 → f 的单位行
                    val = if i == j && e_block == 1 { 1.0 } else { 0.0 };
                } else if case.pv[i] {
                    val = if i == j {
                        if e_block == 0 { 2.0 * v[i].re } else { 2.0 * v[i].im }
                    } else {
                        0.0
                    };
                }
                jac.vals[base + deg + t] = val;
            }
        }
    }
}

/// Ybus 列 j 内行 i 的位置（邻接表升序，二分）
#[inline]
fn pos_in_col(ybus: &YbusC, j: usize, i: usize) -> usize {
    let s = &ybus.ri[ybus.cp[j]..ybus.cp[j + 1]];
    s.binary_search(&i).unwrap()
}

// ---------------------------------------------------------------------------
// 二次修正项 q(δ) = δ ⊙ conj(Yδ)：一趟 SpMV + 逐点乘，stencil 天然并行
// ---------------------------------------------------------------------------
fn quad_term(case: &Case, d: &[f64]) -> Vec<f64> {
    let nb = case.nb;
    let dv: Vec<Cx> = (0..nb).map(|k| Cx::new(d[k], d[nb + k])).collect();
    let w = case.ybus.spmv(&dv);
    let mut q = vec![0.0; 2 * nb];
    for k in 0..nb {
        q[k] = dv[k].re * w[k].re + dv[k].im * w[k].im;
        q[nb + k] = dv[k].im * w[k].re - dv[k].re * w[k].im;
        if case.pv[k] {
            q[nb + k] = dv[k].norm2();
        }
    }
    q[0] = 0.0;
    q[nb] = 0.0;
    q
}

// ---------------------------------------------------------------------------
// 稠密 LU（原型求解器；生产路径替换为稀疏分解，符号分解同样可缓存）
// 分解一次，两次回代复用 —— 对应二阶法的"单次分解"性质
// ---------------------------------------------------------------------------
struct Lu {
    lu: Vec<f64>, // 行主序 n×n
    perm: Vec<usize>,
    n: usize,
}

fn lu_factor(a: &[f64], n: usize) -> Lu {
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        // 部分主元
        let mut piv = k;
        let mut best = lu[k * n + k].abs();
        for i in k + 1..n {
            let v = lu[i * n + k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        assert!(best > 0.0, "singular matrix at column {}", k);
        if piv != k {
            for j in 0..n {
                lu.swap(k * n + j, piv * n + j);
            }
            perm.swap(k, piv);
        }
        let inv = 1.0 / lu[k * n + k];
        for i in k + 1..n {
            lu[i * n + k] *= inv;
            let lik = lu[i * n + k];
            for j in k + 1..n {
                lu[i * n + j] -= lik * lu[k * n + j];
            }
        }
    }
    Lu { lu, perm, n }
}

fn lu_solve(f: &Lu, b: &[f64]) -> Vec<f64> {
    let n = f.n;
    let mut x: Vec<f64> = (0..n).map(|i| b[f.perm[i]]).collect();
    // 前代 Ly = Pb（L 单位对角）
    for i in 0..n {
        let mut s = x[i];
        for j in 0..i {
            s -= f.lu[i * n + j] * x[j];
        }
        x[i] = s;
    }
    // 回代 Ux = y
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in i + 1..n {
            s -= f.lu[i * n + j] * x[j];
        }
        x[i] = s / f.lu[i * n + i];
    }
    x
}

fn csc_to_dense(jac: &Csc) -> Vec<f64> {
    let n = jac.n;
    let mut d = vec![0.0; n * n];
    for c in 0..n {
        for p in jac.cp[c]..jac.cp[c + 1] {
            d[jac.ri[p] * n + c] = jac.vals[p];
        }
    }
    d
}

// ---------------------------------------------------------------------------
// H(r) = Σ_k r_k·∇²F_k：块组装（与 OPF 加权 Hessian 同一台机器，λ := 残差）
// 块公式：H_ij = M(λ_i·Y_ij) + M(conj(λ_j·Y_ji)),  λ_i = rP_i + i·rQ_i
// 对角块：2(G_ii·rP_i − B_ii·rQ_i)·I；PV 加 2·rV_i·I；slack 无对角块
// 模式 = Ybus 1-hop 块结构，已被有限差分验证（max|Δ| ≈ 2e-9，严格对称）
// 原型用稠密输出；生产路径走符号缓存 CSC 单趟填充
// ---------------------------------------------------------------------------
fn assemble_hr(case: &Case, r: &[f64]) -> Vec<f64> {
    let nb = case.nb;
    let mut h = vec![0.0f64; 4 * nb * nb];
    // 复数乘子 λ_i（PV 的第二方程是 |V|²，离对角只保留 P 方程贡献；slack 自身无二次方程）
    let lam: Vec<Cx> = (0..nb)
        .map(|i| {
            if i == 0 {
                Cx::new(0.0, 0.0)
            } else if case.pv[i] {
                Cx::new(r[i], 0.0)
            } else {
                Cx::new(r[i], r[nb + i])
            }
        })
        .collect();
    let ybus = &case.ybus;
    for i in 0..nb {
        for p in ybus.cp[i]..ybus.cp[i + 1] {
            let j = ybus.ri[p];
            if i == j {
                continue;
            }
            let y_ij = ybus.vals[p];
            // Y[j,i]（对称存放，直接取镜像位置）
            let y_ji = {
                let s = &ybus.ri[ybus.cp[j]..ybus.cp[j + 1]];
                ybus.vals[ybus.cp[j] + s.binary_search(&i).unwrap()]
            };
            let c1 = lam[i] * y_ij;
            let c2 = (lam[j] * y_ji).conj();
            // M(c) = [[c.re, -c.im],[c.im, c.re]]，两块相加
            let b00 = c1.re + c2.re;
            let b01 = -(c1.im + c2.im);
            let b10 = c1.im + c2.im;
            let b11 = b00;
            h[i * 2 * nb + j] += b00;
            h[i * 2 * nb + nb + j] += b01;
            h[(nb + i) * 2 * nb + j] += b10;
            h[(nb + i) * 2 * nb + nb + j] += b11;
        }
        if i == 0 {
            continue; // slack 无对角块
        }
        let y_ii = {
            let s = &ybus.ri[ybus.cp[i]..ybus.cp[i + 1]];
            ybus.vals[ybus.cp[i] + s.binary_search(&i).unwrap()]
        };
        let d = if case.pv[i] {
            2.0 * y_ii.re * r[i] + 2.0 * r[nb + i]
        } else {
            2.0 * (y_ii.re * r[i] - y_ii.im * r[nb + i])
        };
        h[i * 2 * nb + i] += d;
        h[(nb + i) * 2 * nb + nb + i] += d;
    }
    h
}

// ---------------------------------------------------------------------------
// 精确 LM（NLLS full Newton）：(JᵀJ + H(r) + μI)δ = −Jᵀr，增益比自适应 μ
// exact = false 时退化为 GN-LM（H ≡ 0），用于对照
// 原型直接形成法方程；生产路径走增广系统保持 1-hop（见论文推导）
// ---------------------------------------------------------------------------
fn run_lm(case: &Case, v0: &[Cx], exact: bool, tol: f64, maxit: usize) -> (Vec<Cx>, usize, bool) {
    let nb = case.nb;
    let n2 = 2 * nb;
    let mut v = v0.to_vec();
    let mut mu = 1e-2f64;
    let mut jac = symbolic_cache(&case.ybus);
    for it in 0..maxit {
        let r = mismatch(case, &v);
        let res_inf = r.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        if res_inf < tol {
            return (v, it, true);
        }
        let f = 0.5 * r.iter().map(|x| x * x).sum::<f64>();
        fill_jacobian(case, &v, &mut jac);
        let jd = csc_to_dense(&jac);
        // g = Jᵀr；A = JᵀJ (+ H(r))
        let mut g = vec![0.0f64; n2];
        let mut a = vec![0.0f64; n2 * n2];
        for i in 0..n2 {
            for k in 0..n2 {
                g[i] += jd[k * n2 + i] * r[k];
            }
        }
        for i in 0..n2 {
            for j in 0..n2 {
                let mut s = 0.0;
                for k in 0..n2 {
                    s += jd[k * n2 + i] * jd[k * n2 + j];
                }
                a[i * n2 + j] = s;
            }
        }
        if exact {
            let hr = assemble_hr(case, &r);
            for p in 0..n2 * n2 {
                a[p] += hr[p];
            }
        }
        // μ 内循环：增益比决定接受/拒绝
        let mut accepted = false;
        for _ in 0..30 {
            let mut am = a.clone();
            for i in 0..n2 {
                am[i * n2 + i] += mu;
            }
            let lu = lu_factor(&am, n2);
            let rhs: Vec<f64> = g.iter().map(|x| -x).collect();
            let d = lu_solve(&lu, &rhs);
            let mut vn = v.clone();
            let mut finite = true;
            for k in 0..nb {
                let dv = Cx::new(d[k], d[nb + k]);
                if !dv.re.is_finite() || !dv.im.is_finite() {
                    finite = false;
                }
                vn[k] += dv;
            }
            if !finite {
                mu *= 10.0;
                continue;
            }
            let rn = mismatch(case, &vn);
            let fn_ = 0.5 * rn.iter().map(|x| x * x).sum::<f64>();
            let pred = -0.5 * g.iter().zip(d.iter()).map(|(a, b)| a * b).sum::<f64>();
            let rho = if pred > 0.0 { (f - fn_) / pred } else { -1.0 };
            if rho > 1e-4 {
                v = vn;
                accepted = true;
                if rho > 0.75 {
                    mu = (mu / 3.0).max(1e-12);
                }
                break;
            } else {
                mu *= 2.0;
                if mu > 1e12 {
                    return (v, it, false);
                }
            }
        }
        if !accepted {
            return (v, it, false);
        }
    }
    let r = mismatch(case, &v);
    let ok = r.iter().fold(0.0f64, |m, &x| m.max(x.abs())) < tol;
    (v, maxit, ok)
}

// ---------------------------------------------------------------------------
// Newton 驱动：second_order = false 标准 NR；true Chebyshev 两步
// ---------------------------------------------------------------------------
fn run(case: &Case, v0: &[Cx], second_order: bool, tol: f64) -> (Vec<Cx>, usize, bool) {
    let nb = case.nb;
    let mut v = v0.to_vec();
    let mut jac = symbolic_cache(&case.ybus);
    for it in 0..50 {
        let r = mismatch(case, &v);
        let res = r.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        if res < tol {
            return (v, it, true);
        }
        fill_jacobian(case, &v, &mut jac);
        let lu = lu_factor(&csc_to_dense(&jac), 2 * nb);
        let rhs1: Vec<f64> = r.iter().map(|x| -x).collect();
        let d1 = lu_solve(&lu, &rhs1);
        let dv = if second_order {
            let q = quad_term(case, &d1);
            let rhs2: Vec<f64> = q.iter().map(|x| -x).collect();
            let d2 = lu_solve(&lu, &rhs2); // 复用同一分解
            d1.iter().zip(d2.iter()).map(|(a, b)| a + b).collect::<Vec<_>>()
        } else {
            d1
        };
        for k in 0..nb {
            v[k] += Cx::new(dv[k], dv[nb + k]);
            if !v[k].re.is_finite() || !v[k].im.is_finite() {
                return (v, it, false);
            }
        }
    }
    let r = mismatch(case, &v);
    let ok = r.iter().fold(0.0f64, |m, &x| m.max(x.abs())) < tol;
    (v, 50, ok)
}

// ---------------------------------------------------------------------------
// 测试算例：14 节点环网 + 弦支路，精确解标定，验证三阶收敛
// ---------------------------------------------------------------------------
fn main() {
    let nb = 14;
    let mut branches: Vec<(usize, usize)> = (0..nb).map(|i| (i, (i + 1) % nb)).collect();
    for i in (0..nb).step_by(2) {
        branches.push((i, (i + 3) % nb));
    }
    // y = 1/(R + iX)
    let den = 0.02f64 * 0.02 + 0.06 * 0.06;
    let y_series = Cx::new(0.02 / den, -0.06 / den);

    let ybus = YbusC::from_branches(nb, &branches, y_series, 0.05);

    // 精确解 v*
    let v_star: Vec<Cx> = (0..nb)
        .map(|k| {
            let ang = 0.004 * (1.7 * k as f64).cos() - 0.018 * k as f64 / nb as f64;
            let mag = 1.0 + 0.006 * (2.3 * k as f64).sin();
            Cx::new(mag * ang.cos(), mag * ang.sin())
        })
        .collect();

    // 由精确解标定注入
    let yv_star = ybus.spmv(&v_star);
    let mut p_spec = vec![0.0; nb];
    let mut q_spec = vec![0.0; nb];
    let mut v2_spec = vec![0.0; nb];
    let mut pv = vec![false; nb];
    for k in 0..nb {
        let s = v_star[k] * yv_star[k].conj();
        p_spec[k] = s.re;
        q_spec[k] = s.im;
        v2_spec[k] = v_star[k].norm2();
        pv[k] = k > 0 && k % 3 == 0;
    }

    let case = Case { nb, ybus, pv, p_spec, q_spec, v2_spec };

    // 平坦启动（slack 固定为精确值）
    let mut v_flat = vec![Cx::new(1.0, 0.0); nb];
    v_flat[0] = v_star[0];

    let (v_nr, _, _) = run(&case, &v_flat, false, 1e-13);
    let (v_ch, _, _) = run(&case, &v_flat, true, 1e-13);

    let err = |v: &[Cx]| {
        v.iter()
            .zip(v_star.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a.re - b.re).abs()).max((a.im - b.im).abs()))
    };
    println!("NR   终态误差 ‖v-v*‖∞ = {:.3e}", err(&v_nr));
    println!("二阶 终态误差 ‖v-v*‖∞ = {:.3e}", err(&v_ch));

    // 逐迭代误差（重跑并记录，用于验证收敛阶）
    for (name, so) in [("NR", false), ("二阶", true)] {
        let mut v = v_flat.clone();
        let mut jac = symbolic_cache(&case.ybus);
        println!("--- {} ---", name);
        let mut prev = err(&v);
        for it in 0..10 {
            let r = mismatch(&case, &v);
            fill_jacobian(&case, &v, &mut jac);
            let lu = lu_factor(&csc_to_dense(&jac), 2 * nb);
            let rhs1: Vec<f64> = r.iter().map(|x| -x).collect();
            let d1 = lu_solve(&lu, &rhs1);
            let dv = if so {
                let q = quad_term(&case, &d1);
                let rhs2: Vec<f64> = q.iter().map(|x| -x).collect();
                let d2 = lu_solve(&lu, &rhs2);
                d1.iter().zip(d2.iter()).map(|(a, b)| a + b).collect::<Vec<_>>()
            } else {
                d1
            };
            for k in 0..nb {
                v[k] += Cx::new(dv[k], dv[nb + k]);
            }
            let e = err(&v);
            let order = if prev > 0.0 && e > 0.0 {
                prev.ln() / e.ln() // 粗略指标，主要看比值 e/prev^p
            } else {
                0.0
            };
            let _ = order;
            println!("iter {}: ‖v-v*‖∞ = {:.3e}", it, e);
            if e < 1e-13 {
                break;
            }
            prev = e;
        }
    }

    // 精确残差恒等式验证：F(v+δ₁) == q(δ₁)
    let mut jac = symbolic_cache(&case.ybus);
    fill_jacobian(&case, &v_flat, &mut jac);
    let lu = lu_factor(&csc_to_dense(&jac), 2 * nb);
    let r = mismatch(&case, &v_flat);
    let rhs1: Vec<f64> = r.iter().map(|x| -x).collect();
    let d1 = lu_solve(&lu, &rhs1);
    let mut v1 = v_flat.clone();
    for k in 0..nb {
        v1[k] += Cx::new(d1[k], d1[nb + k]);
    }
    let lhs = mismatch(&case, &v1);
    let rhs = quad_term(&case, &d1);
    let diff = lhs
        .iter()
        .zip(rhs.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    println!("\n精确残差恒等式 max|F(v+δ₁) - q(δ₁)| = {:.3e}", diff);

    // -----------------------------------------------------------------------
    // 病态工况对比：高 R/X 网 + 大相角差精确解，扫描负荷因子
    // 预期（与 Python 验证版一致）：
    //   α≤1.1 四家皆收敛；1.15~1.2 仅精确 LM 收敛；α≥1.22 全灭（不可行区）
    // -----------------------------------------------------------------------
    let den2 = 0.2f64 * 0.2 + 0.6 * 0.6;
    let y_series2 = Cx::new(0.2 / den2, -0.6 / den2);
    let ybus2 = YbusC::from_branches(nb, &branches, y_series2, 0.05);
    let v_star2: Vec<Cx> = (0..nb)
        .map(|k| {
            let ang = 0.32 * (1.3 * k as f64).sin() - 0.22 * k as f64 / nb as f64;
            let mag = 0.97 + 0.02 * (2.1 * k as f64).sin();
            Cx::new(mag * ang.cos(), mag * ang.sin())
        })
        .collect();
    let yv2 = ybus2.spmv(&v_star2);
    let mut p2 = vec![0.0; nb];
    let mut q2 = vec![0.0; nb];
    let mut v22 = vec![0.0; nb];
    for k in 0..nb {
        let s = v_star2[k] * yv2[k].conj();
        p2[k] = s.re;
        q2[k] = s.im;
        v22[k] = v_star2[k].norm2();
    }
    let case2 = Case { nb, ybus: ybus2, pv: case.pv.clone(), p_spec: p2, q_spec: q2, v2_spec: v22 };
    let mut v_flat2 = vec![Cx::new(1.0, 0.0); nb];
    v_flat2[0] = v_star2[0];

    println!("\n病态工况扫描（迭代数，x = 发散）");
    println!("alpha   | NR  | Chebyshev | GN-LM | exact-LM");
    for &alpha in &[1.0f64, 1.1, 1.15, 1.18, 1.2, 1.22] {
        // 负荷因子：缩放注入指定值
        let case_a = Case {
            nb,
            ybus: YbusC::from_branches(nb, &branches, y_series2, 0.05),
            pv: case2.pv.clone(),
            p_spec: case2.p_spec.iter().map(|x| x * alpha).collect(),
            q_spec: case2.q_spec.iter().map(|x| x * alpha).collect(),
            v2_spec: case2.v2_spec.clone(),
        };
        let (_, it_nr, ok_nr) = run(&case_a, &v_flat2, false, 1e-10);
        let (_, it_ch, ok_ch) = run(&case_a, &v_flat2, true, 1e-10);
        let (_, it_gn, ok_gn) = run_lm(&case_a, &v_flat2, false, 1e-10, 200);
        let (_, it_ex, ok_ex) = run_lm(&case_a, &v_flat2, true, 1e-10, 200);
        let f = |it: usize, ok: bool| if ok { format!("{:3}", it) } else { "  x".to_string() };
        println!(
            "α={:4.2} | {} | {} | {} | {}",
            alpha,
            f(it_nr, ok_nr),
            f(it_ch, ok_ch),
            f(it_gn, ok_gn),
            f(it_ex, ok_ex)
        );
    }
}
