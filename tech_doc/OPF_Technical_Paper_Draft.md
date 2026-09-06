# Symbolic Structure and Branch-Local Curvature for Direct KKT Assembly in AC Optimal Power Flow

**Technical paper draft — 5 September 2026**  
**Implementation audited:** RustPower `symbolic-kkt-lm`, commit `adf1458c72604e02f3e15f9b7c4060b21fc13769` plus the local branch-Hessian diagonal-insertion fix in `src/basic/d2sbr_dv2.rs`.  
**Status:** mathematical description and reproducible preliminary experiments; further validation needed for a submission-ready performance study. This manuscript concerns AC-OPF independently of the LM power-flow work.

## Abstract

An interior-point AC optimal power flow solver repeatedly evaluates network derivatives and assembles a reduced Karush–Kuhn–Tucker (KKT) system. This paper describes an implementation that derives the sparse KKT structure from network connectivity, generator incidence, and fixed-variable constraints. The mathematical observation is that the Hessian and eliminated-slack contribution of each apparent-power constraint have support on only the four voltage variables at its branch endpoints. Their sum can therefore be accumulated into four coefficient channels over the structural bus-admittance pattern. Combined with direct complex-valued differentiation, this permits numerical KKT assembly without constructing a separate Hessian or performing a general sparse Jacobian-product assembly. Successive implementations additionally reuse nonlinear and bound-constraint Jacobian structures. Matched-input experiments on IEEE39 and IEEE118 reproduce pandapower objectives to relative differences below $1.1\times10^{-8}$ for the latest path, with independently checked power balances and limits. Within RustPower, the latest implementation reduces complete solver-call time by approximately **1.15× and 1.70×** relative to the externally checked V4 direct-Hessian path. A sparse-translation defect in the legacy baseline was corrected and that baseline was independently revalidated before the timing comparison. The comparison with pandapower also shows substantial solver-level acceleration, but includes differences in language, linear solver, and dual initialization. Remaining work includes workspace reuse, complete dual-result recovery, converter alignment, and broader robustness and scaling experiments.

**Index terms:** AC optimal power flow; interior-point method; symbolic sparsity; direct KKT assembly; exact derivatives; sparse numerical computation.

## 1. Introduction and contribution

AC-OPF combines nonlinear network equalities with voltage, generation, and thermal constraints. Its repeated derivative evaluations offer a different opportunity from accelerating a power-flow Jacobian alone: an OPF implementation must also assemble the Lagrangian Hessian, incorporate barrier contributions, and couple primal variables to equality multipliers.

The contribution examined here is a structure-derived formulation of that assembly. The implementation uses analytical derivatives and the support of each physical constraint to determine where numerical coefficients belong. Preallocation follows from this derivation. The central claim is therefore more specific than avoiding memory allocation: the reduced Newton operator can be evaluated through local branch contributions and topology-derived column layouts, avoiding general-purpose sparse matrix products and concatenations in the relevant paths.

Three contributions form the proposed paper:

1. A support argument for assembling exact branch curvature and slack-elimination terms within the network graph, together with a topology-derived KKT CSC layout.
2. A sequence of implementations that isolates direct Hessian evaluation, symbolic KKT filling, fused branch assembly, and persistent constraint-Jacobian structures.
3. A reproducible comparison using the same numerical network, costs, bounds, and primal starting point in RustPower and pandapower, accompanied by independent feasibility and solution checks.

This is an extension from PF derivative assembly to the constrained AC-OPF Newton system. It does not require the LM power-flow method as a component. A final novelty claim needs a focused literature comparison and a check against the author's submitted PF manuscript, which was unavailable for this audit. Standard complex derivative identities and primal-dual interior-point equations are established foundations [1,2]; the paper should identify precisely which structural construction and assembly organization are new.

## 2. Optimization model and reduced Newton system

### 2.1 Model implemented

The current numerical model uses polar decision variables

$$
x=[\theta^T,v^T,p_g^T,q_g^T]^T,\qquad n_x=2n_b+2n_g,
$$

where generation is in per unit and $V_i=v_i e^{\mathrm{j}\theta_i}$. For base power $S_b$, the supported objective is separable quadratic active-generation cost:

$$
f(x)=\sum_{a=1}^{n_g}\left[c_{2a}(S_b p_{ga})^2+c_{1a}S_b p_{ga}+c_{0a}\right].
$$

Power balance is

$$
g(x)=\begin{bmatrix}\Re r(x)\\\Im r(x)\end{bmatrix}=0,
\quad r(x)=\operatorname{diag}(V)\overline{YV}-C_g(p_g+\mathrm{j}q_g)+s_d.
$$

For branch $\ell=(f,t)$, the two thermal constraints are

$$
h_{\ell f}=|V_f\overline{(Y_fV)_\ell}|^2-\bar s_\ell^2\le0,
\quad h_{\ell t}=|V_t\overline{(Y_tV)_\ell}|^2-\bar s_\ell^2\le0.
$$

Variable bounds cover voltage magnitudes and active/reactive generation. Equal lower and upper bounds become linear equalities, including the reference angle. The audit uses finite thermal limits on every branch. Reactive-power costs, general piecewise-linear objectives, and arbitrary additional linear constraints are outside this harness and must not be implied by the numerical results.

The filenames containing `rect` do **not** imply a rectangular-coordinate OPF: the inspected V4 and V5 paths still solve for $\theta$ and $v$, using complex scalars to evaluate their derivatives.

### 2.2 Elimination of inequality slack directions

Let $g$ include the linear equalities and $h$ include the bound inequalities. Introduce $z>0$ with $h+z=0$, equality multipliers $\lambda$, inequality multipliers $\mu>0$, and objective scaling $c_m$. Using row-oriented Jacobians $J_g,J_h$, define

$$
L_x=c_m\nabla f+J_g^T\lambda+J_h^T\mu,
\qquad W=\nabla^2_{xx}(c_mf+\lambda^Tg+\mu^Th).
$$

With $Z=\operatorname{diag}(z)$ and $D=\operatorname{diag}(\mu/z)$, eliminating slack and inequality-multiplier directions yields

$$
\underbrace{\begin{bmatrix}M&J_g^T\\J_g&0\end{bmatrix}}_K
\begin{bmatrix}\Delta x\\\Delta\lambda\end{bmatrix}
=-\begin{bmatrix}L_x+J_h^TZ^{-1}(\mu\odot h+\gamma\mathbf1)\\g\end{bmatrix},
\quad M=W+J_h^TDJ_h.
$$

The remaining directions are recovered by

$$
\Delta z=-h-z-J_h\Delta x,\qquad
\Delta\mu=-\mu+Z^{-1}(\gamma\mathbf1-\mu\odot\Delta z).
$$

The implementation stores transposed constraint Jacobians as `dg` and `dh`. Its centering parameter is $\gamma=0.1(z^T\mu)/n_h$, with fraction-to-boundary factor $0.99995$. The audit uses $c_m=10^{-4}$ and $10^{-6}$ for each stopping tolerance. Feasibility and stationarity use scaled infinity norms; complementarity uses the summed gap. Thus a feasibility tolerance of $10^{-6}$ does not itself promise an unscaled maximum power mismatch below $10^{-6}$ p.u.

The baseline explicitly forms the sparse product corresponding to $J_h^TDJ_h$. The optimized paths derive and accumulate the nonlinear portion locally, while bound inequalities contribute only diagonal terms.

## 3. Structure-derived direct assembly

### 3.1 Branch-local exact curvature

For a branch end, write

$$
S=a v_f^2+T,\qquad T=bV_f\overline{V_t},
\quad a=\overline{Y_{ff}},\quad b=\overline{Y_{ft}}.
$$

Only $u=[\theta_f,\theta_t,v_f,v_t]^T$ enters this expression. Its first derivative vector is

$$
d=\frac{\partial S}{\partial u}
=[\mathrm{j}T,-\mathrm{j}T,2av_f+T/v_f,T/v_t]^T.
$$

For $h=|S|^2-\bar s^2$ and $Q_{ij}=\partial^2S/\partial u_i\partial u_j$,

$$
\eta_i=\frac{\partial h}{\partial u_i}=2\Re(\overline S d_i),
\quad H_{ij}=2\Re(\overline{d_i}d_j+\overline S Q_{ij}).
$$

If $E_\ell$ selects the four endpoint variables, this constraint contributes

$$
E_\ell^TB_\ell E_\ell,\qquad
B_\ell=\mu_\ell H_\ell+\frac{\mu_\ell}{z_\ell}\eta_\ell\eta_\ell^T
$$

to $M$. The function `branch_end_hess_v4` computes this exact $4\times4$ block. Both branch ends are included, with the receiving-end variable order permuted back to the common endpoint order. This retains second derivatives; it is not a Gauss–Newton approximation.

### 3.2 Structural-support proposition

**Proposition.** Assume a fixed network of two-terminal branches, a structural admittance pattern containing every endpoint pair and bus diagonal, separable generation costs, and individual variable bounds. The voltage block of $M$ has a structural superpattern consisting of four copies of the bus-admittance graph.

**Argument.** Each branch thermal constraint depends on two complex voltages, so both its Hessian and its weighted gradient outer product have support only on the two diagonal endpoint blocks and their two cross blocks. Summing parallel branches changes coefficients without introducing a new endpoint pair. Weighted power-balance curvature consists of self terms and terms $V_i\overline{V_j}$ for network neighbors, with the same support property. Individual bound penalties add only diagonals, and separable generation costs add generation diagonals. Consequently no general graph expansion from sparse Jacobian multiplication is required for this particular operator.

This is a structural statement. Numerical cancellations in $Y$ must not remove an edge needed by a branch constraint. Phase-shifting transformers may make admittance **values** nonsymmetric even though the stored endpoint pattern is symmetric. More general controls, coupled constraints, and multi-terminal devices require an extended support construction.

### 3.3 Symbolic CSC construction and numerical filling

`KKTSymbolicV5::from_parts` enumerates KKT rows in sorted block order from each admittance column's neighbors, the generators attached to each bus, and the fixed-variable list. The production wrappers currently use the natural bus order. Utilities for a PQ/PV/reference permutation exist, but the inspected wrappers do not apply that permutation; no ordering speedup is attributed to it here.

For the current structural assumptions, let $m=\operatorname{nnz}(Y)$ and $n_f$ be the number of fixed variables. The stored KKT dimension is $4n_b+2n_g+n_f$, and its enumerated structural entry count is

$$
\operatorname{nnz}_{\mathrm{stored}}(K)=12m+6n_g+2n_f.
$$

This includes structural zeros, such as the reactive-generation diagonal. The layout construction follows directly from four voltage Hessian blocks and two copies of the equality Jacobian. It avoids a generic COO-to-CSC conversion for the KKT skeleton itself. Ancillary setup still builds caches and performs index searches; the entire setup is not search-free.

V5.2 computes node/equality coefficients directly into KKT storage and adds branch blocks through precomputed destinations. V5.3 first accumulates branch blocks into four real coefficient channels for each stored $Y$ entry, then gathers those channels while filling KKT columns. The scratch array occupies $32m$ bytes for four `f64` channels. Numerical work is linear in $m+n_\ell+n_g$ for this fixed-support assembly, excluding factorization. There is no blanket claim of improved asymptotic complexity over every sparse baseline: the demonstrated benefit is removing generic products, intermediate structures, and repeated structural processing.

The final column writes are independent in principle. The preceding branch accumulation still has shared endpoints and is sequential. Parallelizing it requires a reduction strategy; merely adding a parallel iterator does not make the complete algorithm race-free.

### 3.4 Implemented stages

| Path | Implemented change | Work still present |
|---|---|---|
| V1 | Matrix-based derivative evaluation and general sparse barrier-product/KKT assembly | Intermediate sparse products and matrices |
| V4 | Direct complex-scalar exact curvature; branch barrier terms merged locally | Separate Hessian, general KKT concatenation, standard constraint evaluator |
| V5.0 | Topology-derived KKT skeleton and mapped numerical fill | Separate Hessian and constraint matrices |
| V5.2 | Direct node/equality KKT fill and branch-block scatter | Standard constraint/Jacobian construction for outer PIPS operations |
| V5.3 | Branch curvature projected into four network-pattern channels, then gathered into columns | Projection scratch allocation and standard constraint evaluator |
| V5.5 | Nonlinear constraint/Jacobian values updated in reusable patterns | Rebuilding merged nonlinear/bound Jacobian containers |
| V5.6 | Bound-constraint columns included in persistent merged Jacobians | Vector allocation, KKT copies, and repeated per-call setup |

All measured Rust paths use the same persistent KLU wrapper within a solve. It analyzes once on first use, attempts numeric refactorization subsequently, and falls back to a new numeric factorization on a reported error. Symbolic reuse is therefore a shared Rust baseline feature, not a benefit introduced only by V5.6.

## 4. Experimental method

### 4.1 Matched-input comparison

The exporter intercepts the normal PIPS call made by `pandapower.runopp`, records the exact primal starting vector and bounds, and exports $Y$, $Y_f$, $Y_t$, load powers, generator incidence, branch ratings, and polynomial costs from that internal model. The Rust harness constructs `OPFData` directly from these arrays and asserts agreement of its generated bounds with the exported bounds. Both systems use a zero reference angle, apparent-power limits, and the same numerical tolerances. Pandapower preprocessing choices, including its generator/ext-grid voltage-bound handling, are preserved in the common model.

Every Rust version receives the same $x_0$. Initial dual states differ: Rust uses $z_k=\max(1,-h_k)$ and $\mu_k=1/z_k$; this installed PYPOWER implementation leaves $\mu_k=1$ for these initial slacks. Thus the external comparison does not force identical iteration trajectories. All seven Rust paths nevertheless take 14 iterations on each matched case, allowing an equal-iteration within-Rust comparison. Derivative equivalence still needs its own check; it cannot be inferred from the iteration count.

The Python verifier independently reconstructs complex voltages, bus mismatches, both branch-end apparent powers, bound violations, and generation cost from every Rust output. It checks all 35 recorded solves per case. Acceptance thresholds are relative objective difference $<10^{-6}$, power-balance infinity norm $<10^{-5}$ p.u., branch-rating exceedance $<10^{-7}$ p.u., and bound exceedance $<10^{-7}$ in the corresponding variable units. These are explicit cross-solver audit thresholds, distinct from the scaled internal stopping tests.

### 4.2 Timing boundaries and environment

Measurements use one warm-up followed by five solves on logical CPU 0 of an Intel Core Ultra 9 288V, with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` for Python. Versions are Rust 1.98.0, pandapower 3.5.4, NumPy 2.5.2, and SciPy 1.18.1. Rust is compiled in release mode with `klu_dyn`. SciPy's installed sparse direct backend is SuperLU; scikit-umfpack is absent. Raw samples and environment metadata accompany this draft.

Pandapower's **PIPS core** timing covers its PIPS call with already constructed network admittances and callbacks. The separately recorded **runopp API** timing includes conversion and result processing. Rust's timing includes the solver wrapper, its symbolic setup, KLU initialization, and, for optimized paths, `NewOPFData` construction and a model clone. Input-file parsing, conversion diagnostics, final independent validation, and JSON serialization are excluded. Each sample creates a fresh solver; these are repeated cold solver calls after a process warm-up, not cached time-series solves. Existing Rust timing-log output to redirected stderr remains inside the wrapper timing.

A compatibility alias restores SciPy's removed `.H` property to `conjugate().transpose()` because pandapower's OPF derivative code still uses it. This changes no derivative formulas and is confined to the audit process. The original installed packages are not edited.

Cross-language speed ratios describe the measured complete numerical implementations. They combine assembly, sparse backend, initialization, and interpreter effects. The corrected within-Rust comparisons give the stronger evidence for the assembly contribution. Both V1 and V4 are checked against the external Hessian reference below.

## 5. Results

### 5.1 Numerical agreement

| Case | Method | Objective | PIPS iterations | Power-balance infinity norm (p.u.) |
|---|---|---:|---:|---:|
| case39 | pandapower | 41872.304316432 | 15 | 2.626e-07 |
| case39 | Rust V5.6 | 41872.303897215 | 14 | 4.591e-07 |
| case118 | pandapower | 129704.739185125 | 20 | 1.719e-06 |
| case118 | Rust V5.6 | 129704.739764507 | 14 | 1.857e-07 |

All listed runs converged. For V5.6, both cases have zero positive branch-rating or variable-bound violations in the independent calculation. The maximum differences from pandapower's generation dispatch are case39: 0.00669 MW and 0.01094 Mvar; case118: 0.03109 MW and 0.17566 Mvar. These differences are consistent with nearby terminated numerical solutions; equality of objective alone would not establish equality of dispatch. The raw artifact reports both voltage-angle and voltage-magnitude differences as well.

Pandapower's unscaled IEEE118 mismatch exceeds $10^{-6}$ p.u. while satisfying its internal stopping condition, illustrating why scaled solver tolerances must be distinguished from independently measured raw residuals. No global-optimality certificate is claimed for this nonconvex model. Complete independent stationarity/complementarity verification of Rust results remains a follow-up because its public result currently omits the bound duals and final slack vector.

### 5.2 Independent Hessian comparison

The harness also evaluates the objective-scaled Lagrangian Hessian at a deterministic perturbation of the shared starting state, using nonuniform equality and positive inequality multipliers. The reference is pandapower's `opf_hessfcn`; the Rust comparison uses V4 with the barrier-gradient outer-product term disabled so that both return the same mathematical Hessian.

| Case | Path vs pandapower | Maximum absolute Hessian difference | Maximum scaled difference |
|---|---|---:|---:|
| case39 | V1 | 1.455e-11 | 5.768e-15 |
| case39 | V4 | 1.091e-11 | 2.304e-14 |
| case118 | V1 | 5.457e-12 | 1.264e-14 |
| case118 | V4 | 5.457e-12 | 1.816e-14 |

The scaled error is the maximum entrywise quantity $|a-b|/(1+\max(|a|,|b|))$, over the union of stored entries. After correction, both V1 and V4 agree with pandapower to approximately floating-point rounding accuracy in these probes. The initial V1 discrepancy was caused by `d2Sbr_dV2::subtract_diags`, which skipped a diagonal correction when the branch-derived sparse matrix did not already store that position. SciPy sparse subtraction inserts the missing entry. The error was in the sparse translation of branch curvature, not the copied bus-injection derivative formulas. A two-bus analytical regression reproduced the missing opposite-endpoint angle curvature before the fix and passes after inserting the required diagonals. The corrected V1 is therefore retained as the conventional baseline. Earlier measurements are archived in `opf_audit/before_diagonal_fix/`; the tables here use fresh Rust runs after correction. Two probe states do not replace randomized derivative and Newton-direction validation.

### 5.3 Complete solver-call timing

| Method | IEEE39 median [min, max] (ms) | IEEE118 median [min, max] (ms) |
|---|---:|---:|
| pandapower PIPS core | 151.045 [149.861, 153.504] | 295.494 [292.291, 297.041] |
| pandapower runopp API | 162.867 [160.756, 165.693] | 308.427 [304.701, 309.888] |
| V1 | 2.048 [2.030, 2.147] | 7.785 [7.648, 7.849] |
| V4 | 0.972 [0.955, 0.982] | 4.432 [4.387, 4.690] |
| V5.0 | 0.795 [0.760, 0.821] | 3.724 [3.652, 3.850] |
| V5.2 | 0.730 [0.721, 0.797] | 3.366 [3.332, 3.385] |
| V5.3 | 0.871 [0.824, 0.909] | 3.425 [3.391, 3.526] |
| V5.5 | 0.984 [0.979, 1.021] | 2.790 [2.741, 2.866] |
| V5.6 | 0.845 [0.834, 0.919] | 2.599 [2.581, 2.639] |

The latest V5.6 path is approximately **178.8× and 113.7×** faster than the measured pandapower PIPS core on IEEE39 and IEEE118. These are local implementation-level results under the boundaries above. The corresponding within-Rust V1/V5.6 ratios are **2.43× and 2.99×**, with equal iteration counts, after validating the corrected V1 derivatives. Relative to the externally checked V4 path, V5.6 reduces solver-call time by **1.15× and 1.70×**. V5.2 is faster than V5.6 on IEEE39 in this small sample: removing more iteration work does not guarantee lower total time when extra setup is significant.

### 5.4 Assembly ablation on IEEE118

| Path | Hessian / fused fill (ms) | G/H region (ms) | KKT region (ms) | First solve region (ms) | Later solve regions (ms) |
|---|---:|---:|---:|---:|---:|
| V1 | 2.916 | 1.161 | 1.669 | 0.627 | 1.142 |
| V4 | 0.224 | 1.105 | 0.985 | 0.670 | 1.088 |
| V5.0 | 0.224 | 1.125 | 0.287 | 0.630 | 1.097 |
| V5.2 | 0.268 | 1.037 | 0.053 | 0.616 | 1.020 |
| V5.3 | 0.282 | 1.055 | 0.056 | 0.624 | 1.052 |
| V5.5 | 0.263 | 0.272 | 0.053 | 0.611 | 1.010 |
| V5.6 | 0.261 | 0.159 | 0.052 | 0.623 | 1.006 |

Stage entries are median accumulated times over a solve. “Hessian/fused fill” measures different amounts of work across generations: from V5.2 onward it also includes direct equality/KKT coefficient filling. It should be interpreted jointly with KKT work, not as an isolated Hessian microbenchmark. G/H timing also includes objective-gradient and Lagrangian-gradient updates. Setup and other outer-loop work account for the remaining total, and independently computed medians need not add exactly.

The main reductions are direct scalar curvature evaluation in V4, removal of KKT concatenation through V5.0/V5.2, and the persistent G/H structures of V5.5/V5.6. V5.3 is not faster than V5.2 in this serial small-network run; its different accumulation organization should be evaluated separately for larger cases and parallel execution.

The fields named `solve_sym` and `solve_num` in the implementation are **not** pure symbolic and numeric factorization timings. The first accumulates the entire first solve region; the second accumulates later solve regions. They also cover associated copies, right-hand-side work, and direction recovery. They cannot establish that factorization has reached a hardware or theoretical lower bound.

### 5.5 Conversion differences and unsuccessful trials

The separate native-converter audit found the following maximum differences from the exported pandapower model:

| Case | $\max\lvert\Delta Y_{ij}\rvert$ (p.u.) | Maximum lower-voltage-bound difference | Maximum upper-voltage-bound difference | Maximum rating difference (p.u.) |
|---|---:|---:|---:|---:|
| case39 | 5.859e-14 | 4.200e-02 | 7.800e-02 | 8.882e-16 |
| case118 | 4.013e-01 | 9.500e-02 | 2.500e-02 | 0.000e+00 |

Bus counts, generator ordering, branch ordering, and load vectors agree in these checks. IEEE118 additionally has an admittance discrepancy that needs element-by-element localization; this audit does not assign it to a specific transformer formula without that check. Native-builder cost defaults also require explicit cost-table application. The older repository OPF tests therefore must not be compared numerically with a fresh `runopp` result solely because the case names match. The matched-input experiment isolates and validates the numerical solver independently of that conversion work.

An additional flat-start IEEE300 attempt in pandapower failed numerically after 40 iterations under the same tolerance settings. Its failure is saved in `opf_audit/case300_failure.json`. No matched Rust trial or speed ratio is reported for IEEE300. This is an unsuccessful pilot, not evidence that either solver is generally more robust. Trying a common PF-derived starting point is an appropriate next experiment.

The historical PEGASE9241 timings in `docs/design/opf_design_doc.md` are not included as fresh experimental results. That report contains varying iteration counts and a nonconverged V5.5 run, and labels the first-solve timer as symbolic factorization. A new large-case matched-input experiment is required before using those observations in a paper's performance tables.

## 6. Further improvements

### 6.1 Reuse numerical workspaces and remove redundant setup

`assemble_kkt_v5_3` allocates voltage-related vectors, a fixed-variable mask, and the four-channel branch buffer each iteration. `V55Evaluator::update`, also used by V5.6, creates voltage and power vectors. `solve_kkt_fused_timed` copies KKT values and CSC index arrays, allocates the RHS and direction vectors, and performs binary searches for bound-penalty diagonals. Consequently “persistent sparse structure” is supported, while “zero allocation throughout the solve” is not yet supported.

A persistent workspace can own these arrays, cache the fixed-variable mask and diagonal destinations, and update values in place. The solver interface could accept immutable CSC structure and mutable numerical data without copying indices each iteration. `NewOPFData::new` constructs an older symbolic cache even when newer wrappers build their own caches; eliminating this duplicated setup is especially relevant to IEEE39. Measure allocation counts and setup separately before and after these changes. Repeated network solves should additionally reuse valid symbolic objects across calls, with explicit invalidation on topology, generator-incidence, or constraint-structure changes.

### 6.2 Consolidate the interior-point loop and expose complete results

The versioned module organization is documented in [the module guide](../src/new_opf/README.md). Inputs, outputs, assembly, evaluation, and ECS adapters now have explicit boundaries. The optimized PIPS drivers have moved to `src/new_opf/interior_point/`, while the baseline remains in `src/opf/`. Four driver implementations still contain similar initialization, stopping, step, and result-packing logic. A shared driver with interchangeable evaluators and assembly workspaces would reduce duplicated code while retaining the ablation paths. Regression checks should verify complete Newton directions and termination behavior, not merely objectives in a broad interval.

The returned `mu_lower` and `mu_upper` arrays are currently filled with zeros. Recover actual bound multipliers and expose final slacks, fixed-variable multipliers, and all stopping measures. Define whether multipliers are returned in objective-scaled or physical units; the current internal equality multipliers reflect `cost_mult`, so price interpretation also needs the appropriate objective/base-power conversion. This is necessary for independent KKT-residual checks and meaningful economic outputs.

The linear solve currently uses `unwrap()` in the PIPS driver. Return a structured numerical failure with the last iterate instead. Add finite-value checks and retain residual histories so that a failed large-case run provides useful diagnostics.

### 6.3 Resolve model-conversion gaps

First align voltage-bound rules and localize the measured IEEE118 admittance difference. The native line-limit builder also does not incorporate `parallel`, derating, or `max_loading_percent` into its line rating formula, although those factors can matter for general networks. Transformer tap-side and magnetizing-branch conventions warrant targeted comparisons. Cases with zero/unlimited ratings need special attention: the current constraint path retains all branch rows and can insert infinite limits, whereas pandapower excludes unconstrained branch rows. A supported finite constraint subset should be formed before symbolic analysis.

Tests should cover inactive elements, noncontiguous IDs, parallel circuits, phase shifts, multiple generators per bus, fixed generation, and topology changes. Cost lookup should follow element identities across filtering, rather than assume every table position is preserved. These tasks concern model equivalence and user-facing OPF coverage; they do not invalidate the matched numerical-input assembly results.

### 6.4 Improve the linear solve with measured evidence

Use the existing feature-gated KLU probes to distinguish symbolic analysis, first numeric factorization, refactorization, fallback factorization, and triangular solution. Compare ordering, scaling, pivot quality, and normalized linear-system residuals on actual OPF KKT matrices. A symmetric-indefinite backend could reduce work, but it needs suitable pivoting or regularization and reliable numerical behavior. The LM QDLDL path cannot simply be presumed suitable for a general nonconvex OPF KKT matrix. Neither GPU acceleration nor iterative solvers should be presented as an established improvement before they are measured.

### 6.5 Extend the mathematical and empirical validation

Retain the corrected V1 baseline and its external derivative check. For a submission, add randomized-state comparisons of Hessians and complete reduced KKT directions, including taps and parallel branches; assert scale-aware tolerances in tests that currently only print diagnostic differences. Add finite-difference directional checks independent of the legacy derivative path. After the diagonal fix, the broader release library test run reports **102 passed, 3 ignored**, with 12 benchmark/performance tests filtered out. This includes the new analytical branch-curvature regression and existing OPF derivative and symbolic-structure checks, but is not a robustness survey.

Evaluate both cold setup and cached repeated solves on larger networks, including PEGASE9241, with shared finite-constraint models and starting points. Record converged and failed runs, raw feasibility, stationarity, complementarity, memory, factor fill, and timing distributions. Use fixed-iterate microbenchmarks to isolate assembly from nonlinear iteration counts. Study column parallelism together with a deterministic branch-reduction scheme. The present serial data supports direct assembly and persistent structures; it does not yet support claims of linear multicore scaling or universal fastest performance.

## 7. Conclusion

The implemented acceleration comes from evaluating the constrained Newton operator according to its physical and mathematical support: exact local branch curvature and barrier outer products are combined before assembly, the KKT structure is derived from network connectivity, and later versions retain constraint-Jacobian structures. Matched-input experiments establish close numerical agreement with pandapower and a measurable within-Rust reduction in complete solve time. The next useful advances are reusable workspaces, complete dual outputs, conversion alignment, and larger controlled experiments. Those extensions can strengthen the paper without changing the exact-curvature basis of the present method.

## References and implementation traceability

[1] R. D. Zimmerman, *AC Power Flows, Generalized OPF Costs and their Derivatives using Complex Matrix Notation*, MATPOWER Technical Note 2, revision 7, 2019. [Technical note](https://matpower.org/docs/TN2-OPF-Derivatives.pdf).

[2] MATPOWER, *MIPS: primal-dual interior-point nonlinear programming solver*. [Official documentation](https://matpower.org/documentation/mips/functions/mips.html).

[3] pandapower, *Optimization with PYPOWER*, version 3.5.4 documentation. [OPF interface and options](https://pandapower.readthedocs.io/en/stable/opf/pypower_run.html). The actual comparator is the installed `pandapower.pypower.pips` and `pipsopf_solver` source inspected for this audit.

| Paper component | Repository implementation |
|---|---|
| Model, bounds, initial point | `src/opf/problem.rs`, `src/opf/builder.rs` |
| Baseline derivatives | `src/opf/cost.rs`, `constraints.rs`, `hessian.rs` |
| Reduced system and PIPS drivers | `src/opf/pips.rs`, `src/new_opf/interior_point/` |
| Exact branch scalar curvature | `src/new_opf/assembly/v4/curvature.rs::branch_end_hess_v4` |
| KKT symbolic layout | `src/new_opf/assembly/v5/symbolic.rs::KKTSymbolicV5` |
| Fused assembly | `src/new_opf/assembly/v5/scatter.rs`, `src/new_opf/assembly/v5/partitioned.rs` |
| Persistent nonlinear/merged Jacobians | `src/new_opf/evaluation/v5/nonlinear.rs`, `src/new_opf/evaluation/v5/merged.rs` |
| Version selection and setup | `src/new_opf/configurations.rs`, `src/new_opf/model.rs` |
| KLU reuse and optional probes | `src/basic/solver/klu.rs` |
| Reproducible external comparison | `performance/audit_opf_pandapower.py`, `performance/audit_opf.rs` |
| Raw data and verification | `tech_doc/opf_audit/` |

## Appendix A. Reproduction

Run from the `symbolic-kkt-lm` worktree with an installed Python environment containing the versions above. Replace `PYTHON` and `CARGO_TARGET_DIR` with local paths. The supplied worktree includes the diagonal-insertion fix; reproduce these results with that fix present.

```bash
export PYTHON=/home/cts/workspace/rustpower/.venv/bin/python
export CARGO_TARGET_DIR=/home/cts/workspace/rustpower/target
cargo build --release --features klu_dyn --example audit_opf

taskset -c 0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PYTHON" performance/audit_opf_pandapower.py

taskset -c 0 "$CARGO_TARGET_DIR/release/examples/audit_opf" \
  tech_doc/opf_audit/case39_input.json 5 \
  > tech_doc/opf_audit/case39_rust.jsonl 2> /tmp/opf39-timing.log
taskset -c 0 "$CARGO_TARGET_DIR/release/examples/audit_opf" \
  tech_doc/opf_audit/case118_input.json 5 \
  > tech_doc/opf_audit/case118_rust.jsonl 2> /tmp/opf118-timing.log
"$PYTHON" performance/audit_opf_pandapower.py --compare

cargo test --release --lib --features klu_dyn opf:: \
  -- --nocapture --skip bench --test-threads=1
```

Each `*_input.json` has a SHA-256 digest in its `*_pandapower.json` reference. The `*_rust.jsonl` files retain each solution vector and stage timings; `*_comparison.json` records independently recomputed feasibility and agreement. The conversion diagnostics and independent Hessian probes are outside the numerical timing region. `*_derivatives.json` contains the Hessian comparisons; the corresponding probe states and reference matrices are stored in `*_input.json`. The supplied tables are one five-repeat local experiment, not confidence intervals or a completed scalability study.
