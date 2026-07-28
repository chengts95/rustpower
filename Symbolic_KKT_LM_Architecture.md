# Symbolic-Pattern KKT/LM Assembly Architecture

**Design document — from mathematical principles to top-level implementation logic**
Scope: exact Levenberg–Marquardt with an assembled polar Hessian, sharing one Ybus-derived symbolic pattern with the production Jacobian (`JacobianPattern2`) and the OPF KKT path — the LM (1,1) block and the OPF interior-point (1,1) block are the same machine. The Chebyshev two-step is **out of scope**: its exactness is a rectangular-coordinate property (Section 1.2); if ever revived, it enters through a coordinate transform, not this pattern. Parallelism is out of scope for this document; the architecture is designed so that it becomes a later addition without structural changes.

---

## 1. Mathematical Foundations

### 1.1 Quadratic structure in rectangular coordinates (background)

*This subsection motivates the (out-of-scope) rect Chebyshev path and the `ext_ref` prototype. The LM path itself is polar (Section 1.4); only consequence 3 — every block lives on the Ybus graph — is load-bearing here.*

Let the complex bus voltage be `v = e + i f` and the network admittance matrix `Y = G + iB`. The complex power injection is

```
s(v) = v ⊙ conj(Y·v)
```

Every entry of the power-flow mismatch vector `F(v)` is a **quadratic form** in `(e, f)`:

- PQ bus: `P_k = Re{s_k} − P_k^spec`, `Q_k = Im{s_k} − Q_k^spec`
- PV bus: `P_k = Re{s_k} − P_k^spec`, `V²_k = |v_k|² − |V_k^spec|²`
- Slack bus: eliminated from the reduced system; its effect enters as linear and constant terms through the precomputed injection current `Yv` (the `scalc` channel)

Consequences:

1. The Hessian of each equation, `H_k = ∇²F_k`, is a **constant matrix** built only from `G, B`.
2. The Taylor expansion terminates exactly at second order — no truncation error:

```
F(v + δ) = F(v) + J(v)·δ + q(δ),    q(δ) = δ ⊙ conj(Y·δ)
```

3. The sparsity graph of every Hessian equals the Ybus graph (1-hop neighborhood per bus).

### 1.2 Second-order Newton (Chebyshev two-step) — out of scope

**Dropped.** The exact truncation below is a rectangular-coordinate property: in polar form `F` contains `sin/cos` and is not quadratic, so `q(δ₁)` ceases to be the exact residual and the cubic-convergence monitor dies. If a second-order step is ever wanted, the plan is a coordinate transform into rect, not a second pattern family. The prototype remains in `ext_ref/second_order_pf`; its only surviving role here is the numerical evidence that exact LM dominates in the ill-conditioned window (Section 1.3).

```
J·δ₁ = −F(v)          one LU factorization
q  = q(δ₁)            one Ybus SpMV + elementwise product (stencil, matrix never formed)
J·δ₂ = −q             reuse of the same factorization (back-substitution only)
v ← v + δ₁ + δ₂
```

### 1.3 Exact Levenberg–Marquardt

Merit function `f(v) = ½‖F(v)‖²`. Its exact Hessian:

```
∇²f = JᵀJ + Σ_k r_k·∇²F_k = JᵀJ + H(r),    H(r): assembled with multipliers λ := residual r
```

Newton step with damping: `(JᵀJ + H(r) + μI)·δ = −Jᵀr`, with gain-ratio adaptive μ.

**Why the exact Hessian matters.** Near the loadability boundary, `J` is nearly singular along the fold direction `w` (`Jw ≈ 0`). Along `w` the true merit function is a quartic curve (`F` changes as `t²·Q(w,w)`); the Gauss–Newton model is a parabola whose minimum explodes as `1/‖Jw‖²`. The dropped term supplies exactly the missing curvature:

```
wᵀ·H(r)·w = rᵀ·Q(w,w)
```

Gauss–Newton LM stalls (μ → ∞, step → 0); exact LM walks through. Verified numerically: a loading window exists (α ≈ 1.15–1.2 on the ill-conditioned test case) where NR, Chebyshev, and GN-LM all diverge while exact LM converges. Beyond feasibility, exact LM halts at a least-squares stationary point — a built-in infeasibility diagnostic.

### 1.4 Hessian block formulas (polar, reduced system)

The LM path is **polar** — states `[θ (all active); |V| (PQ only)]`, equations `[P (all active); Q (PQ only)]`, matching the production Jacobian (`JacobianPattern2`) and the OPF heritage (`d2sbus_dv2.rs`, `v4_numeric_rect.rs`). With complex multipliers `λ_k = rP_k + i·rQ_k` (PV buses: `λ_k = rP_k`, no Q row), the four polar quadrants of `H(r)` are the MATPOWER TN2 blocks, evaluated per Ybus edge with rectangular-complex arithmetic:

```
C_ij = λV_i · conj(Y_ij·V_j)
E_ij = conj(V_i) · (conj(Y_ii)·V_i·λ_i − dλ_i)         (i = j;  dλ = Yᴴ·(λ⊙V))
E_ij = conj(V_i) · conj(Y_ji) · V_j·λ_j                 (i ≠ j)
F_ij = C_ij − λV_i·conj(Ibus_i)                         (i = j),   F_ij = C_ij  (i ≠ j)

Gaa_ij = Re{E_ij + F_ij}                 Gva_ij = Re{ j·(E_ij − F_ij)/|V_i| }
Gvv_ij = Re{(C_ij + C_ji)/(|V_i||V_j|)}  Gav    = Gvaᵀ
```

All four quadrants live on the Ybus 1-hop graph and are assembled in a single graph pass — cost equivalent to one Jacobian assembly. The rectangular-coordinate variant (2×2 blocks, `H_ij = M(λ_i·Y_ij) + M(conj(λ_j·Y_ji))`) is retained in `ext_ref/second_order_pf` for reference; both were verified against finite differences (≈ 2e-9, exact symmetry).

**Fill discipline (from the v4 lesson).** The numeric kernel writes the four quadrants in two passes with no `i == j` branch in the hot loop: one off-diagonal pass over the graph edges writes every coupling entry; one per-bus diagonal pass writes every diagonal entry (via `diag_off`). The v4 prototype's inline `if i == j` is extracted into the diagonal pass.

### 1.5 The augmented system and its KKT identity

To avoid forming `JᵀJ` (2-hop fill), introduce the auxiliary residual `s = Jδ + r`. The LM step is the KKT system of the QP `min ½sᵀs + ½δᵀ(μI + H(r))δ  s.t.  s = Jδ + r`:

```
┌ μI + H(r)    Jᵀ ┐ ┌ δ ┐   ┌  0 ┐
│                 │ │   │ = │    │
└ J            −I ┘ └ s ┘   └ −r ┘
```

Properties: symmetric indefinite, quasi-definite; every block lives on the Ybus 1-hop graph; `JᵀJ` never materializes. This is structurally identical to the OPF interior-point KKT system — the (1,1) slot holds a multiplier-weighted Hessian in both (residuals here, dual variables in OPF), the off-diagonal holds `J`, the (2,2) block is diagonal. One assembly engine serves both.

### 1.6 Bus-type permutation and the reduced system

Buses are ordered `PQ → PV → slack`. The reduced system removes slack rows/columns entirely; in the polar convention PV buses lose the `Q` equation and the `|V|` column, so states and equations are both `[n_act | n_pq]`-partitioned and the system stays square. Because row indices within a Ybus column are sorted, each column splits into contiguous segments:

```
[0, pq_end)          PQ neighbors → full block writes
[pq_end, active_end) PV neighbors → P-row blocks only
[active_end, …)      slack        → never allocated, never touched
```

The type permutation is therefore also a degree-ascending ordering: degenerate rows sink to the bottom of each segment, which coincides with good elimination practice. All of this is baked into the symbolic phase; the numeric phase contains no type logic.

**Retention conventions.** The PF/LM path uses the *reduced* system described above: slack eliminated (its physics enters through the `scalc`/`yv` channel), PV buses contributing `P` rows and `θ` columns only. The OPF path uses *full retention*: every row and column is kept (slack buses included, no PV compression), in line with the existing `v5_kkt` convention. Full retention is the degenerate case of the same machinery — `n_pq = n_active = n_bus`, every segment resolving to the whole Ybus column. The layer structure, the block descriptors, and the base-plus-offset addressing are identical; only the segment lengths differ. One engine, two conventions.

---

## 2. Core Principle: Symbolic/Numeric Separation

### 2.1 The pattern is the Ybus offset itself

The design invariant: **the fill loop reads the Ybus CSC and writes result arrays at positions derived by start-plus-offset arithmetic only.** No slot maps, no per-edge index tables, no runtime search. The symbolic phase emits exactly five derived arrays:

| Array | Meaning |
|---|---|
| `pq_ends`, `active_ends` | segment boundaries within each column |
| `diag_ptrs` | offset of the diagonal edge per column (absolute in Ybus values, as in `JacobianPattern2`); its column-local form `diag_off[k] = diag_ptrs[k] − y_col_ptrs[k]` is block-independent |
| `y_trans` | offset of the mirror edge `(j,i)` per nnz `(i,j)` |
| segment-start tables | per-block, per-column starts = `base` + prefix sums of cache-derived segment lengths (Section 3) |

Two address families follow from the invariant and are shared by **every** block — they are derived, never tabulated per block:

- **Column starts.** Every block replicates each Ybus column as one or more contiguous segments whose lengths are functions of `pq_ends` / `active_ends`. Hence `col_starts[k] = base + Σ_{j<k} seg_len(j)`: a single prefix sum over cache-derived lengths, computed once in the symbolic phase. A column start is never searched for and never stored per edge.
- **Diagonal slots.** The diagonal edge sits at a fixed *local* offset within Ybus column `k`: `diag_off[k] = diag_ptrs[k] − y_col_ptrs[k]`. Because each block's column `k` replicates the Ybus row order (or its PQ/active prefix), **the same local offset addresses the diagonal in every block**: `diag_pos(k) = base + col_starts[k] + diag_off[k]` — and, for the second-segment quadrants (`va`, `vv`), at `+ active_ends[k]`. One integer per bus, derived from the Ybus structure plus the block's `base`, serves J, Jᵀ, H(r), and the LM μ-shift alike.

This is the addressing already proven in `fill_jacobian_v3` (`j11_starts[k] + (diag_ptrs[k] − y_start)`, one `diag_offset` shared by all four J quadrants), promoted here from a Jacobian-local trick to the architecture-wide convention.

### 2.2 Transpose writes via `y_trans`

`J` and `Jᵀ` share the traversal and the intermediate products. While processing edge `p = (i, j)`, the kernel writes the `J` block at `p`-based positions and the `Jᵀ` block at positions derived from `y_trans[p]`. For blocks shaped exactly like the active Ybus CSC, the transposed position **is** `y_trans[p]` itself. No transpose logic exists at runtime.

### 2.3 Write-once coverage

Every slot of every output array is written exactly once per fill: coupling terms in the main edge loop, diagonal corrections (injection-current terms, `|V|²` rows, `H(r)` diagonal) in a second per-bus pass. No zeroing pass is required; no slot is written twice. This invariant is what makes later column-level parallelism trivially safe.

---

## 3. Top-Level Data Architecture

Functional decomposition, four layers. Coordinate system (polar J vs rect J/H) affects only segment-length formulas and kernel formulas — never the layer structure.

### 3.1 Layer 0 — `YbusAnalysisCache`

Already exists (rustpower). Provides: `col_ptrs`, `row_indices`, `pq_ends`, `active_ends`, `diag_ptrs`, `y_trans`. Single source of truth for every derived quantity below.

### 3.2 Layer 1 — Block descriptors (`BlockDesc`)

Each block matrix is identified by **one integer**: its base offset in the global value array. Everything else is derived from Layer 0.

**Key structural fact (polar, reduced system): J, H(r) and Jᵀ share one column pattern.** Every column of bus `k` is `[active neighbours][PQ neighbours, shifted by n_act]`:

- **J**: θ col = `[J11 | J21]`, |V| col = `[J12 | J22]` — the layout of `JacobianPattern2` itself;
- **H(r)**: θ col = `[aa | va]`, |V| col = `[av | vv]` — the polar Hessian quadrants of Section 1.4;
- **Jᵀ**: P col = `[θ rows | |V| rows]`, Q col likewise (its Q columns are full-degree for PQ buses; see Section 4.3).

The symbolic phase therefore builds the pattern **once** (`graph`) and places the blocks by base offsets:

```rust
pub struct BlockDesc {
    pub base: usize,            // the single integer marking the block's start
    pub col_starts: Vec<usize>, // col_starts[k] = base + prefix-sum of per-column lengths
    pub row_indices: Vec<usize>,// global row numbers (flat, in emission order)
    pub n_cols: usize,
    pub nnz: usize,
}

pub struct KktPattern {
    pub cache: YbusAnalysisCache,
    pub graph: BlockDesc,  // shared column pattern of J, H(r) and Jᵀ (base 0)
    pub j_base: usize,     // = 0
    pub h_base: usize,     // = graph.nnz
    pub jt_base: usize,    // = 2·graph.nnz
    pub d_base: usize,     // = 3·graph.nnz; −I entry of equation i at d_base + i
    pub nnz_total: usize,  // = d_base + (n_act + n_pq)
}
```

The pattern stores **no diagonal table and no per-edge map**. With `diag_off[k] = diag_ptrs[k] − y_col_ptrs[k]` from Layer 0, any leading-segment diagonal is `base + col_starts[col] + diag_off[bus]`; the second-segment quadrants (`va`, `vv`) follow by adding the leading segment length `active_ends[bus]` — which is all `apply_mu_delta` ever touches (Section 4.4).

Per-column length is `active_end + pq_end` for every column of every block (θ columns of all active buses, |V| columns of PQ buses). The full-retention OPF convention (Section 1.6) replaces both cuts by the whole column length — same code path, different cuts.

### 3.3 Layer 2 — Storage views (two, one set of kernels)

- **Block-independent view**: each block its own CSC. Used by the CG/Craig route (`H` factored alone, `J`/`Jᵀ` as stencil operators) and by any block-parallel schedule.
- **Flat view**: one global CSC for direct solvers. Column layout: δ-columns contain `[H segment | J segment]`, s-columns contain `[Jᵀ segment | −I]`; segment boundaries are cache-derived. Global row indices and column pointers are assembled once in the symbolic phase.

The fill kernels never know which view they are writing — they receive a base pointer and a starts table. View selection is a symbolic-phase decision with zero numeric-phase cost.

### 3.4 Layer 3 — Partitioned fill

```rust
pub fn fill_kkt(pat: &KktPattern, input: &FillInput, values: &mut [f64]) {
    let (j_vals,  rest)   = values.split_at_mut(pat.graph.nnz);
    let (h_vals,  rest)   = rest.split_at_mut(pat.graph.nnz);
    let (jt_vals, d_vals) = rest.split_at_mut(pat.graph.nnz);

    fill_j(&pat.cache,  &pat.graph, input, j_vals);   // existing v3 kernel, unchanged
    fill_h(&pat.cache,  &pat.graph, input, h_vals);   // H(r) kernel: off-diag pass + diag pass
    fill_jt(&pat.cache, &pat.graph, input, jt_vals);  // transposed-write kernel
    fill_neg_i(pat.d_base, d_vals);                   // constant block
    // μ handled separately: apply_mu_delta touches only the aa/vv diagonal slots
}
```

Each kernel writes a disjoint slice — sequential now, spawnable later without structural change. Within a kernel, columns also write disjoint segments (raw-pointer style already used in v3), which is the second, finer parallel axis reserved for later.

---

## 4. Fill Kernels

### 4.1 `fill_j`

Existing v3 kernel. Per bus column, contiguous quadrant slices via `jslice`, coupling terms from `Yvnorm`-scaled products, diagonal corrections (`S_calc` terms, `inv_vmag` scalings) applied afterward through `diag_ptrs`.

### 4.2 `fill_h`

Polar quadrants per Section 1.4, in **two passes with no `i == j` branch in the hot loop** (the v4 prototype's inline diagonal branch is extracted):

1. **Off-diagonal pass** — one sweep over the Ybus edges. For edge `p = (i, j)`, `i ≠ j`: compute `C_ij, E_ij, F_ij` (mirror coefficient `C_ji` via `y_trans[p]`) and write `Gaa, Gva, Gav, Gvv` at `p`-based positions in the four quadrant segments.
2. **Diagonal pass** — one per-bus sweep. Compute `E_ii, F_ii` from the precomputed `dλ` and `Ibus` and write the four diagonal slots via `col_starts[k] + diag_off[k]` (+ `active_ends[k]` for the second segment).

O(1) complex FMAs per nnz; per-bus quantities (`λV`, `dλ`, `Ibus`, `inv_vmag`) precomputed once per fill.

### 4.3 `fill_jt`

Shares the traversal of `fill_j` (or runs standalone with the same intermediates): values identical to the `J` block, written at `y_trans`-derived positions. Q-quadrant compressed segments handled by the same prefix property (PQ neighbors are the column prefix).

### 4.4 `apply_mu_delta`

μ updates (LM inner loop) write only the state-diagonal slots: `dμ` added at the `aa` diagonal of every θ column (`h_base + col_starts[k] + diag_off[k]`) and at the `vv` diagonal of every |V| column (`+ active_ends[k]`), all addresses derived from the block's `base` and the Ybus structure — no μ-specific table. The main fill is never re-run for a μ change.

---

## 5. Safety Model (`unsafe` policy)

Deliberate and sanctioned:

- Raw-pointer slice writes (`jslice`-style) and `get_unchecked*` are the default in all numeric-phase kernels.
- Justification by invariant, not by convention: write-once coverage + disjoint segments + symbolic-phase-validated offsets. All invariants are machine-checked by the verification protocol (Section 7) on every structural change.
- Safe wrappers only at API boundaries (length checks against `pat.nnz`), never inside loops.

---

## 6. Implementation Plan (sequential, no parallelism)

| Phase | Deliverable | Gate |
|---|---|---|
| 0 ✅ | `YbusAnalysisCache` + shared `graph` BlockDesc + `KktPattern::build` (polar) | starts tables match hand-computed values on 3-bus and 14-bus fixtures; `base + col_starts[k] + diag_off[k]` matches a direct search of every diagonal quadrant; `graph` identical to `JacobianPattern2` entry-by-entry |
| 1 ✅ | `fill_h` (polar, off-diag pass + diag pass) + `fill_jt` kernels | finite-difference checks: J vs FD ≤ 1e-8; H(r) vs residual-weighted J-difference ≤ 1e-8; H exactly symmetric; Jᵀ reconstruction == Jᵀ exactly |
| 2 ✅ | `fill_kkt` dispatcher + flat-view global CSC assembly | global CSC passes solver-format validation; block/flat views agree entry-by-entry |
| 3 | Exact-LM driver (dense factorization prototype) | convergence window reproduced (α ∈ [1.15, 1.2]: only exact LM converges); infeasibility stall beyond nose |
| 4 | Wall-clock benchmark suite: symbolic fill vs dynamic-assembly reference implementation | assembly time, allocation count, pattern-rebuild count (PV↔PQ switching cost) reported per iteration |

Each gate is a hard stop: no phase starts with the previous gate red.

---

## 7. Verification Protocol (standing)

1. **Finite-difference harness** for every kernel against the analytic mismatch function (J) and against residual-weighted J-differences (H).
2. **Structural assertions**: exact symmetry of `H(r)`; exact transpose relation of `Jᵀ`; write-once coverage audit (debug-only counter instrumentation).
3. **Algorithmic oracles**: LM convergence window table (α ∈ [1.15, 1.2]: only exact LM converges); least-squares stall beyond the nose (infeasibility diagnostic).

---

## 8. Parked Items (post-paper)

- Chebyshev two-step (rect) via coordinate transform — prototype in `ext_ref/second_order_pf`; the exact-residual monitor does not survive polar conversion (Section 1.2).
- Craig/CG matrix-free solve of the augmented system (research question: iteration counts on power-grid graphs, preconditioner choice).
- Augmented 1-hop LDLᵀ path with constrained ordering (s-variables last).
- μ strategy refinement (Nielsen update, Gershgorin lower bound from `H(r)` diagonal blocks).
- Column-level and block-level parallelism (architecture already reserves both axes).
