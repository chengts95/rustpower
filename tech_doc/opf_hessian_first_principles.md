# OPF Lagrangian Hessian: Rectangular Assembly with a Universal Polar Pull-Back

**Status:** corrected & finite-difference-validated. This note records the mathematical
basis of the rewritten `src/new_opf` Hessian. It supersedes the *double-applied* curvature
correction of `final_report.md` and completes the branch pull-back of `branch_hessian_v3.md`
(which omitted its curvature term, §4 there).

**Method (Route A), in one line.** Every scalar Lagrangian term $P(x)$ has polar Hessian

$$
\boxed{\;H^{\mathrm{polar}} \;=\; \mathcal M^\top H^{\mathrm{rect}} \mathcal M \;+\; \Delta[P]\;}
$$

where $H^{\mathrm{rect}}$ is a **fixed rectangular operator** (a constant network form
contracted with state/multipliers), $\mathcal M$ is the **universal per-node** polar→rect
transform, and $\Delta[P]$ is a **universal diagonal curvature operator** that depends only
on the term's own gradient. Nodes and branches share the *same* $\mathcal M$ and the *same*
$\Delta$; only $H^{\mathrm{rect}}$ and the gradient change. The whole point: $\mathcal M$ is
generic and can be integrated onto a fixed $H^{\mathrm{rect}}$, and the curvature is a single
clean diagonal term — no per-case bookkeeping.

> The shipped kernel (`v3_numeric_scalar.rs`) evaluates the algebraically-fused equivalent
> (one direct-polar pass) for speed; the modular $\mathcal M^\top H^{\mathrm{rect}}\mathcal M+\Delta$
> form below is the derivation it realizes. FD (§5) validates the two agree.

---

## 1. Notation

Bus voltages $V_i = v_i e^{j\theta_i}$, rectangular parts $e_i=v_i\cos\theta_i$,
$f_i=v_i\sin\theta_i$. State $x=[\boldsymbol\theta;\mathbf v;\mathbf P_g;\mathbf Q_g]$. The
Lagrangian whose Hessian we assemble is

$$
\mathcal L(x) = \sigma f(x) + \boldsymbol\lambda^\top g(x) + \boldsymbol\mu^\top h(x),
$$

with $\sigma$ the cost multiplier, $g$ the $2n_b$ power balances
($\lambda=[\lambda_P;\lambda_Q]$), $h$ the $2n_l$ branch apparent-power limits
($\mu=[\mu_f;\mu_t]$). $I=Y_{bus}V$.

The polar→rect coordinate map $m:\ (\theta_i,v_i)\mapsto(e_i,f_i)$ has the per-node Jacobian

$$
\mathcal M_i \;=\; \frac{\partial(e_i,f_i)}{\partial(\theta_i,v_i)}
= \begin{bmatrix} -f_i & \cos\theta_i \\ \ \ e_i & \sin\theta_i \end{bmatrix},
$$

and $\mathcal M = \operatorname{blkdiag}(\mathcal M_1,\dots,\mathcal M_{n_b})$ (identity on the
$P_g,Q_g$ slots). This single operator is reused for every term.

---

## 2. The universal pull-back

### 2.1 Transform identity

For a scalar $P(x)=G\big(m(p)\big)$ with $p$ polar and $z=(e,f)$ rectangular, the chain rule
gives the exact decomposition

$$
\frac{\partial^2 P}{\partial p_a\,\partial p_b}
= \underbrace{\sum_{k,l}\frac{\partial^2 G}{\partial z_k\partial z_l}\,\mathcal M_{ka}\mathcal M_{lb}}_{(\mathcal M^\top H^{\mathrm{rect}}\mathcal M)_{ab}}
\;+\; \underbrace{\sum_k \frac{\partial G}{\partial z_k}\,\frac{\partial^2 m_k}{\partial p_a\partial p_b}}_{\Delta[P]_{ab}} .
$$

Part A is the linear pull-back of the rectangular Hessian; Part B is the curvature of the
(nonlinear) coordinate map contracted with the rectangular gradient. **Part B is present
exactly because $m$ is nonlinear** ($\partial^2 m\neq 0$), and is the term that the earlier
branch blueprint dropped.

### 2.2 The universal curvature operator $\Delta$

Because $m$ is per-node, $\partial^2 m_k/\partial p^2$ is block-diagonal. With
$\partial^2 e_i/\partial\theta_i^2=-e_i,\ \partial^2 e_i/\partial\theta_i\partial v_i=-\sin\theta_i$,
$\partial^2 f_i/\partial\theta_i^2=-f_i,\ \partial^2 f_i/\partial\theta_i\partial v_i=\cos\theta_i$
(all $\partial^2/\partial v_i^2=0$), Part B collapses, for each bus $i$ the term touches, to a
$2\times2$ block in $(\theta_i,v_i)$ written **purely from $P$'s own polar gradient**
$g_{\theta_i}=\partial P/\partial\theta_i,\ g_{v_i}=\partial P/\partial v_i$:

$$
\Delta[P]_i = \begin{bmatrix} -v_i\,g_{v_i} & g_{\theta_i}/v_i \\[2pt] g_{\theta_i}/v_i & 0 \end{bmatrix}.
$$

*Derivation of the $(\theta_i,\theta_i)$ entry:* $-(g^z_{e_i}e_i+g^z_{f_i}f_i)$, and since
$g_{v_i}=g^z_{e_i}\cos\theta_i+g^z_{f_i}\sin\theta_i=(g^z_{e_i}e_i+g^z_{f_i}f_i)/v_i$ this is
$-v_i g_{v_i}$. The $(\theta_i,v_i)$ entry $-g^z_{e_i}\sin\theta_i+g^z_{f_i}\cos\theta_i
=(-g^z_{e_i}f_i+g^z_{f_i}e_i)/v_i=g_{\theta_i}/v_i$, using
$g_{\theta_i}=-g^z_{e_i}f_i+g^z_{f_i}e_i$. $\blacksquare$

This is the corrected $\Delta_{polar}$ of `final_report.md` — that formula was **right**. It is
**term-agnostic**: the node block uses it with the power-balance gradient; *each* branch uses
the *same* operator with that branch's gradient.

### 2.3 The double-count, precisely

The previous implementation computed Pass 1 as the *fused* direct-polar quantity (already
$\mathcal M^\top H^{\mathrm{rect}}\mathcal M+\Delta$) and then **added $\Delta$ again** (Pass 4),
fed with the **full** Lagrangian gradient $\nabla(\lambda^\top g+\mu^\top h)$ instead of the
node term's own gradient. Two stacked errors. Route A is correct iff $\Delta[P]$ is applied
**once per term, with that term's own gradient** — which is what `math_verify.rs::verify_hessian`
does for the node block.

---

## 3. Node power-balance block

### 3.1 Fused multiplier (single complex pass)

The conventional real Hessian is $\operatorname{Re}G(\lambda_P)+\operatorname{Im}G(\lambda_Q)$
with $G$ $\mathbb R$-linear in $\lambda$. With the fused multiplier
$\lambda^{\mathbb C}_i=\lambda_{P,i}-j\lambda_{Q,i}$,

$$
\operatorname{Re}G(\lambda_P)+\operatorname{Im}G(\lambda_Q)=\operatorname{Re}G(\lambda^{\mathbb C}),
$$

(*proof:* coefficientwise $\operatorname{Re}(c)\lambda_P+\operatorname{Im}(c)\lambda_Q
=\operatorname{Re}(c(\lambda_P-j\lambda_Q))$), so one complex pass replaces two real passes.

### 3.2 Rectangular operator $H^{\mathrm{rect}}_{\text{node}}$

With $W=\operatorname{diag}(\lambda^{\mathbb C})\,\overline{Y_{bus}}$ (sparsity $=Y_{bus}$), the
rectangular node Hessian in $[\,\mathbf e;\mathbf f\,]$ is the fixed network form

$$
H^{\mathrm{rect}}_{\text{node}}=\begin{bmatrix} \operatorname{Re}(W+W^\top) & \operatorname{Im}(W-W^\top)\\ -\operatorname{Im}(W-W^\top) & \operatorname{Re}(W+W^\top)\end{bmatrix}.
$$

### 3.3 Polar block

$H^{\mathrm{polar}}_{\text{node}}=\mathcal M^\top H^{\mathrm{rect}}_{\text{node}}\mathcal M+\Delta[\,\lambda^\top g\,]$, with the node gradient
$g^{\text{rect}}_i=\lambda^{\mathbb C}_i\overline{I_i}+\big(Y_{bus}\overline{\lambda^v}\big)_i$,
$\lambda^v_i=\lambda^{\mathbb C}_iV_i$, fed into the universal $\Delta$ of §2.2.

---

## 4. Branch flow block (with its Part B restored)

### 4.1 Local power and its rectangular Hessian

For branch $l:f\to t$, $I_f=Y_{ff}V_f+Y_{ft}V_t$, so

$$
S_f = V_f\overline{I_f}=\overline{Y_{ff}}\,|V_f|^2+\overline{Y_{ft}}\,V_f\overline{V_t},
$$

which is **quadratic in the rectangular coordinates** $z=[e_f,f_f,e_t,f_t]$. Hence with
$P=\operatorname{Re}S_f,\ Q=\operatorname{Im}S_f$ (each quadratic in $z$), the rectangular
Hessian of $\mu\,|S_f|^2=\mu(P^2+Q^2)$ is

$$
H^{\mathrm{rect}}_{\text{br}}=2\mu\big(\nabla_z P\,\nabla_z P^\top+\nabla_z Q\,\nabla_z Q^\top\big)
+2\mu P\,\mathcal H_G+2\mu Q\,\mathcal H_B,
$$

where $\mathcal H_G=\nabla_z^2P,\ \mathcal H_B=\nabla_z^2Q$ are **constant** $4\times4$ network
operators (independent of state, because $P,Q$ are quadratic). The two ends of the branch use
$(\overline{Y_{ff}},\overline{Y_{ft}})$ and $(\overline{Y_{tt}},\overline{Y_{tf}})$.

### 4.2 Polar block — **including the curvature term**

$$
H^{\mathrm{polar}}_{\text{br}}
= \mathcal M^\top H^{\mathrm{rect}}_{\text{br}}\,\mathcal M \;+\; \Delta[\,\mu|S_f|^2\,].
$$

The first term is the rotation pull-back ($\mathcal M$ restricted to buses $f,t$;
equivalently `branch_hessian_v3.md`'s $\operatorname{Re}\{M_{\mathbb C}^H\mathcal H_h M_{\mathbb C}\}$).
The **second term is the branch curvature that `branch_hessian_v3.md` §4 omitted** — it is the
*same* universal $\Delta$ of §2.2, evaluated at buses $f,t$ with the branch penalty's own polar
gradient $\partial(\mu|S_f|^2)/\partial(\theta_i,v_i)$. Without it the branch block is wrong on
the diagonal (the FD discrepancy of §5).

### 4.3 A structural check

Since the angle dependence of $S_f$ enters only through $e^{j(\theta_f-\theta_t)}$, the angle
block of each branch is antisymmetric, $\begin{bmatrix}c&-c\\-c&c\end{bmatrix}$, giving the
diagnostic identity $\big[H\big]^{\theta\theta}_{ii}\big|_{\text{br}}=-\sum_{j\sim i}\big[H\big]^{\theta\theta}_{ij}\big|_{\text{br}}$.
Route A with the §4.2 curvature term satisfies it; the legacy matrix-product path violates it
on the diagonal.

---

## 5. Validation: finite differences are the gold standard

Correctness is judged against the central 4-point finite difference of $\mathcal L$, **not**
against the legacy `opf_hessfcn` (V1) — V1 is itself the buggy reference.

- **IEEE 118, all nonzeros:** the assembly matches FD to the FD noise floor (worst ratio
  $0.57$ at $\text{ATOL}=0.5,\text{RTOL}=5\times10^{-4}$); the legacy path's worst entry is
  $\approx80\times$ — a real **branch-Hessian diagonal error** in `d2Sbr_dV2.rs` (its $F-D-E$
  diagonal correction), invisible to OPF convergence because the optimum is set by the
  gradient and constraints, not the Hessian.
- **PEGASE 9241, single-branch isolation:** full-Lagrangian FD is roundoff-dominated at scale,
  so activate one branch via a one-hot $\mu$ ($O(1)$ penalty, clean FD): the worst-differing
  branch ($f{=}5480,t{=}959$) matches FD to ratio $7\times10^{-4}$; the legacy path is off by
  $\approx245\times$.

**FD caveats.** Use $\lambda=0$ to isolate the branch term, or a one-hot $\mu$ to isolate a
branch, or global cancellation ruins the estimate; judge with a combined tolerance
$\text{ATOL}+\text{RTOL}\,|H|$ (relative-only error is meaningless for near-zero entries).

---

## 6. Code & tests

- `src/new_opf/v3_numeric_scalar.rs` — fused realization of §3–§4 (one direct-polar pass; no
  intermediate matrices; scatters into the precomputed `y_to_lxx`/`br_to_lxx` offsets). It
  carries Part A and the universal $\Delta$ together in closed form, per term — **never the
  global double-applied correction.**
- `src/new_opf/math_verify.rs::verify_hessian` — the explicit Route-A reference for the node
  block ($\mathcal M^\top H^{\mathrm{rect}}\mathcal M+\Delta$).
- `src/new_opf/mod.rs` — `diag_hessian_breakdown_ieee118` (FD gate over all nonzeros);
  `test_exact_hessian_comparison_pegase9241` (one-hot single-branch FD); `run_pegase9241_v3_convergence` (`#[ignore]`).

---

## 7. Open item (not a Hessian issue)

With the FD-correct Hessian, IEEE 39 / 118 reach the analytic optima; PEGASE 9241 still stalls,
but the signature is `feas` $\approx5.5\times10^{-10}$ (feasible) with `grad`/`comp` frozen and
$\alpha_p\to10^{-9}$ as a branch-limit slack hits its bound — interior-point step control at
active bounds. **The curvature is settled; the remaining work is on the solver.**
