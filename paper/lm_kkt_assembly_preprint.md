# Symbolic-Once Construction of the Augmented KKT System for Levenberg–Marquardt Power Flow

*Preprint skeleton — working draft, 2026-09-02. Author line and affiliations to be added.*

---

## Abstract

The Levenberg–Marquardt (LM) method extends Newton–Raphson power flow into
ill-conditioned and infeasible operating regions, but each iteration assembles
and factors a symmetric indefinite system whose dimension is twice the number
of state variables. This note studies the assembly side of that cost. We show
that when the bus admittance matrix is stored in compressed sparse column form
with buses grouped by type, the sparsity pattern of the augmented LM system is
determined entirely by the column offsets of the admittance matrix: every
output column begins at an address that is computed, in closed form, at the
start of its own construction loop. The pattern is therefore built once, and
each numerical iteration writes values directly into their final compressed
positions, with no coordinate-list intermediate and no symbolic multiplication.
A structural-symmetry argument — which remains valid in the presence of phase
shifters, where the admittance matrix is not numerically symmetric — allows the
upper triangle of the augmented system to be filled row by row with strictly
sequential writes. On the 9241-bus PEGASE system, assembly per iteration is
12–42 times faster than coordinate-list and normal-equation baselines that use
the same numerical kernels; storage of the system matrix is reduced by nearly
one half. Damping-parameter retries modify only the diagonal entries and never
trigger reassembly.

*Keywords:* power flow, Levenberg–Marquardt, augmented system, sparse matrix
assembly, quasi-definite matrix, symbolic factorization

---

## 1. Introduction

Newton–Raphson (NR) power flow diverges or stalls when the Jacobian becomes
nearly singular — under heavy loading, at nose points, or in systems with
extreme branch parameters [3, 4]. The Levenberg–Marquardt method [1, 2]
replaces the Newton step by a damped least-squares step and continues to return
a meaningful point — a least-squares minimizer of the mismatch — even when no
exact solution exists. This property has made LM-type methods a standard
remedy for ill-conditioned power flow since the work of Iwamoto and Tamura
[3] and Tripathy et al. [4].

The price of LM is linear-algebraic. Each iteration solves

$$
(J^{T} J + \mu I)\,\delta = -J^{T} r
\tag{1}
$$

and every change of the damping parameter $\mu$ requires another
factorization. Two costs follow: assembling the system matrix, and factoring
it. For large systems the second is usually assumed to dominate. We show that
this assumption holds only if the first is done properly: assembled through
coordinate (COO) lists, the system matrix of a 9241-bus case costs 17–34 ms
per iteration, while a direct construction reduces this to about 1 ms.

The contribution of this note is a construction with the following
properties:

1. The sparsity pattern of the augmented LM system is derived once, in closed
   form, from the column offsets of a type-ordered bus admittance matrix.
   No symbolic matrix multiplication, merge, or sort is involved.
2. Numerical assembly writes each Jacobian and transposed-Jacobian entry
   directly into its final compressed position; there is no intermediate
   storage and no copy pass.
3. A structural symmetry of the admittance matrix (which holds even with
   phase-shifting transformers) permits a row-oriented construction of the
   upper triangle, halving storage and eliminating scattered writes.
4. Damping retries update $n$ diagonal entries in place; the factors' Symbolic
   phase and the assembly pattern are reused unchanged.

Section 2 develops the LM power-flow step and the augmented system we solve.
Section 3 gives the construction. Section 4 describes the alternative
assembly paths used as baselines. Section 5 reports preliminary measurements.
Section 6 concludes.

---

## 2. Least-Squares Power Flow and the Augmented System

### 2.1 Power-flow mismatches in polar form

Let $Y \in \mathbb{C}^{n_b \times n_b}$ be the bus admittance matrix and
$v \in \mathbb{C}^{n_b}$ the bus voltages, $v_k = |V_k| e^{j\theta_k}$. The
complex injection at bus $k$ is

$$
S_k(v) = v_k \, \overline{(Yv)_k},
\tag{2}
$$

and the mismatches against the specified injections are

$$
r_k^{P} = \operatorname{Re} S_k(v) - P_k^{spec}, \qquad
r_k^{Q} = \operatorname{Im} S_k(v) - Q_k^{spec}.
\tag{3}
$$

With buses grouped as PQ (indices $1..n_{pq}$), PV ($n_{pq}+1 .. n_a$,
$n_a = n_{pq} + n_{pv}$), and slack, the state vector and residual are

$$
x = \begin{bmatrix} \theta_{1..n_a} \\ |V|_{1..n_{pq}} \end{bmatrix},
\qquad
r(x) = \begin{bmatrix} r^{P}_{1..n_a} \\ r^{Q}_{1..n_{pq}} \end{bmatrix},
\qquad n = n_a + n_{pq}.
\tag{4}
$$

The Jacobian $J = \partial r / \partial x$ has the usual $2\times 2$ block
structure $(J_{11}, J_{12}; J_{21}, J_{22})$ of active/reactive power
against angle/magnitude. Its entries are rational combinations of the
quantities $Y_{kj} v_j$ and $Y_{kj} \hat{v}_j$, where
$\hat{v}_k = v_k / |V_k|$; evaluating them in complex arithmetic directly
from $Y$ and $v$ avoids all trigonometric recomputation. For example, with
$e_k + j f_k = v_k$ and $(a_{kj}, b_{kj})$ the real and imaginary parts of
$Y_{kj} v_j$,

$$
J_{11}[k,j] = f_k a_{kj} - e_k b_{kj} \quad (j \neq k),
\qquad
J_{11}[k,k] = f_k a_{kk} - e_k b_{kk} - \operatorname{Im} S_k ,
\tag{5}
$$

with analogous formulas for the other three blocks. The details are standard
and omitted.

### 2.2 The damped Gauss–Newton step

We minimize

$$
f(x) = \tfrac{1}{2} \| r(x) \|^2 .
\tag{6}
$$

The Gauss–Newton step with Levenberg–Marquardt damping solves (1), where
$\mu > 0$ is adapted by the gain ratio

$$
\rho = \frac{f(x) - f(x + \delta)}{-\tfrac{1}{2}\, g^{T} \delta},
\qquad g = J^{T} r ,
\tag{7}
$$

using Nielsen's rules [5]: a step with $\rho > 10^{-4}$ is accepted, and
$\rho > 0.75$ decreases $\mu$ by a factor of three; a rejected step doubles
$\mu$; a non-finite iterate multiplies $\mu$ by ten. Trial updates are taken
in polar coordinates: angles by addition, magnitudes by addition on PQ buses
only.

### 2.3 The augmented formulation

Forming $J^T J$ squares the condition number and, for direct methods, costs a
symbolic and numeric sparse multiplication per iteration. We instead solve the
equivalent augmented system

$$
\underbrace{\begin{bmatrix} \mu I & J^{T} \\ J & -I \end{bmatrix}}_{K}
\begin{bmatrix} \delta \\ \lambda \end{bmatrix}
=
\begin{bmatrix} 0 \\ -r \end{bmatrix}.
\tag{8}
$$

Block elimination gives $\lambda = J\delta + r$ and recovers (1) exactly, so
(8) introduces no approximation. The cost is a larger system — dimension
$2n$ — that is symmetric and sparse.

### 2.4 Quasi-definiteness and pivot-free factorization

The matrix $K$ of (8) has the form

$$
K = \begin{bmatrix} A_{11} & A_{12}^{T} \\ A_{12} & -A_{22} \end{bmatrix},
\qquad A_{11} = \mu I \succ 0, \quad A_{22} = I \succ 0 ,
\tag{9}
$$

and is therefore *quasi-definite* in the sense of Vanderbei [6]. Every
symmetric permutation of $K$ admits an $LDL^{T}$ factorization with diagonal
$D$, and the factorization is stable without pivoting. Two practical
consequences follow:

* any fill-reducing ordering computed once from the pattern of $K$ can be
  reused for every value of $\mu$; and
* a failed numeric factorization (a zero pivot) can only mean that the
  quasi-definiteness was lost, i.e. $\mu$ is effectively zero — the damping
  rule then increases $\mu$, which restores the property.

The inertia of $K$ is $(n, n)$, which also provides a cheap certificate that
the factorization behaved as expected.

### 2.5 What changes when $\mu$ changes

Only the $(1,1)$ block of $K$ depends on $\mu$, and only on its diagonal. In
the layout of Section 3 the $\mu$-entries occupy one known position per
column, so a damping retry is $n$ scalar writes followed by a numeric
refactorization; neither the pattern nor the off-diagonal values are touched.

---

## 3. Symbolic Construction from the Admittance Matrix

### 3.1 Type ordering and segment partitions

Assume $Y$ is stored in CSC form $(p^{col}, i^{row}, y)$ with buses grouped
by type as in (4) and row indices sorted within each column. Because the
groups are contiguous, each column is cut into at most three segments —
rows in PQ, rows in PV, rows beyond — and the two cut positions are found by
one binary search each:

$$
\pi_k = \#\{ i : Y_{ik} \neq 0,\ i < n_{pq} \},
\qquad
\alpha_k = \#\{ i : Y_{ik} \neq 0,\ i < n_a \}.
\tag{10}
$$

The position of the diagonal entry within column $k$,

$$
d_k = \operatorname{rank}(k \mid i^{row}[p^{col}_k .. p^{col}_{k+1}]),
\tag{11}
$$

is found by the same search. The vectors $\pi$, $\alpha$, $d$ are computed
once for a given topology.

### 3.2 The reduced Jacobian pattern is a prefix of the admittance pattern

Consider a column of $J$ corresponding to state variable $\theta_j$. Its
nonzero rows are exactly the buses $i$ with $Y_{ij} \neq 0$ that remain in
the reduced system: active rows for the $P$-block, PQ rows for the
$Q$-block. Because rows are sorted, these are the *prefixes* of length
$\alpha_j$ and $\pi_j$ of column $j$ of $Y$. The same holds, with the roles
exchanged, for a magnitude column. Hence every column of the reduced
Jacobian replicates a Ybus column pattern up to a cut, and its values are
computable entry-by-entry from (5) with loop-invariant quantities hoisted.

It follows that the CSC arrays of the augmented matrix (8) can be written
down in closed form. A state column $c \le n$ contains one diagonal entry
(the damping slot) followed by the $c$-th Jacobian column shifted by $n$; a
residual column $n + c$ contains the $c$-th Jacobian *row* (equivalently, a
column of $J^T$) followed by one $-I$ entry. The start address of each output
column is a running sum of segment lengths, each obtained from two Ybus
offsets via (10). Construction of each column therefore begins by computing
its own write address — no auxiliary index tables, no per-branch lists, no
runtime searches.

### 3.3 Structural symmetry and phase shifters

The argument above is column-oriented: it fills $J$ by columns and copies a
transpose. Two of the three measured assembly costs — the transpose pass and
the subsequent copy — are scattered-write passes over the full nonzero count.
Removing them requires computing the rows of $J$ directly, i.e. walking the
*rows* of $Y$.

Phase-shifting transformers make $Y$ numerically unsymmetric
($Y_{kj} \neq \overline{Y_{jk}}$ in general), so a column walk cannot be
reused as a row walk for *values*. The *pattern* of $Y$, however, is
symmetric: a branch contributes an off-diagonal entry in both directions,
with or without a phase shifter. Consequently the partition vectors of
Section 3.1 apply unchanged to the row walk, and in particular the diagonal
of row $k$ sits at the same rank $d_k$ within the row as within the column.
A row-oriented kernel can therefore compute the entries of $J$ row $r$ —
which are the values of column $n+r$ of $K$ — in storage order, writing
strictly sequentially. The diagonal corrections of (5) are applied at the
same rank $d_k$ as in the column-oriented code, so the two constructions
produce bitwise identical values.

### 3.4 Triangle-only storage and solver pairing

$K$ is symmetric, and an $LDL^T$ factorization reads only one triangle.
Storing only the upper triangle of (8) reduces the nonzero count from
$2\, \mathrm{nnz}(J) + 2n$ to $\mathrm{nnz}(J) + 2n$ — nearly one half for
realistic systems, where $\mathrm{nnz}(J) \gg n$.

Whether a given backend accepts triangle-only input depends on *which*
triangle it reads. QDLDL [9] takes the upper triangle in the original
ordering and derives its elimination tree from it, so the triangle-only
layout can be handed over directly. The classical LDL of SuiteSparse [7]
reads the upper triangle of the *permuted* matrix $P K P^{T}$, with $P$
unknown to the caller; it must therefore receive the full symmetric pattern,
as must LU-based solvers. The layout is thus a property of the solver
backend, and the construction code selects it at build time. All three
combinations were verified to produce identical iterates.

### 3.5 Damping retries

With the layouts above, the damping slot of state column $c$ is the single
entry at the column start (full layout) or the lone entry of the column
(triangle-only layout). A $\mu$-retry writes $n$ scalars and calls the
numeric phase of the factorization; Sections 3.2–3.4 are not revisited.
This matters in practice because ill-conditioned cases perform several
$\mu$-retries per accepted step.

---

## 4. Alternative Assembly Paths (Ablation Design)

To isolate the effect of the construction strategy, four alternatives were
implemented and run under the same damping policy, the same numerical
kernels for $J$, and the same $LDL^T$ backend:

| path | description |
|---|---|
| NE-COO | normal equations: COO assembly of $J$, conversion to CSC, then sparse $J^T J$ (pattern and values) every iteration |
| AUG-FS | the full $2 n_b$ Jacobian (all buses, including slack and PV quadrants) assembled, then sliced and restacked as a coordinate list of $K$ |
| AUG-COO | coordinate-list assembly of $K$ with sort/convert on every damping retry |
| AUG-SDF | the construction of Sections 3.1–3.2, full symmetric storage |
| AUG-SDF-TRIU | the row-oriented construction of Section 3.3, triangle-only storage |

The first three represent common practice: coordinate lists, whole-matrix
assembly followed by slicing, and explicit normal equations. Because all
paths share the Jacobian evaluation kernels, measured differences
underestimate what a from-scratch textbook implementation would show; the
comparison is deliberately conservative in favour of the baselines.

---

## 5. Preliminary Results

Test systems: IEEE 39, IEEE 118, and PEGASE 9241 [11]. All runs use a
tolerance of $10^{-8}$ on the infinity norm of the mismatch, a flat-plus-PV
start, and the same damping schedule. Times are single-run wall times in
release mode on a desktop x86-64 machine; the PEGASE rows report the range
of two runs.

| system | path | iterations | wall | assembly / iteration | factor+solve / call |
|---|---|---|---|---|---|
| IEEE 39 | AUG-SDF | 4 | 292 µs | 4.3 µs | 60.5 µs |
| | AUG-SDF-TRIU | 4 | 174 µs | 2.2 µs | 36.5 µs |
| | AUG-FS | 4 | 577 µs | 9.7 µs + 30.0 µs slicing | 95.8 µs |
| | AUG-COO | 4 | 483 µs | 21.4 µs | 90.4 µs |
| | NE-COO | 4 | 561 µs | 50.6 µs | — |
| IEEE 118 | AUG-SDF | 5 | 566 µs | 8.4 µs | 93.9 µs |
| | AUG-SDF-TRIU | 5 | 499 µs | 5.9 µs | 83.8 µs |
| | AUG-FS | 5 | 2.41 ms | 26.2 µs + 111.1 µs | 284.8 µs |
| | AUG-COO | 5 | 2.14 ms | 93.4 µs | 281.9 µs |
| | NE-COO | 5 | 2.74 ms | 193.5 µs | — |
| PEGASE 9241 | AUG-SDF | 11 | 181–202 ms | 1.35–1.47 ms | 13.9–15.6 ms |
| | AUG-SDF-TRIU | 11 | 174–212 ms | 0.80–0.97 ms | 13.9–17.0 ms |
| | AUG-FS | 11 | 969 ms | 5.08 ms + 21.8 ms | 56.5 ms |
| | AUG-COO | 11 | 837 ms | 17.2 ms | 54.2 ms |
| | NE-COO | 11 | 1.08 s | 34.2 ms | — |

Observations:

1. Direct construction is 12–42 times faster per iteration than the
   coordinate-list and normal-equation paths, and the gap widens with system
   size. The symbolic phase is executed once per topology in the direct
   paths, against once per iteration or once per damping retry in the
   baselines.
2. The row-oriented triangle-only variant removes the transpose pass and the
   copy pass; assembly drops by a further 40% on the largest system and
   storage of $K$ is nearly halved.
3. Above a few thousand buses the factorization dominates the wall clock
   (about 14 ms per call at 9241 buses, against about 1 ms of assembly).
   Further gains must come from the factorization side, not from assembly.

Correctness. The row-oriented triangle-only fill agrees with the
column-oriented reference kernel bitwise on IEEE 39. Iteration counts and
final iterates agree with the full-storage path on all systems (maximum
voltage difference $6.2 \times 10^{-15}$ on IEEE 39), and with the
production Newton–Raphson solver at a tolerance of $10^{-12}$.

---

## 6. Concluding Remarks

The augmented LM system of power flow can be assembled for close to nothing
once the admittance matrix carries a type-grouped ordering: the pattern is
closed-form in the admittance column offsets, the numeric phase is a single
sequential-write pass, and damping retries touch only the diagonal. The
remaining cost is the sparse $LDL^T$ factorization, which on the largest
system studied accounts for over 90% of the wall time. Current work applies
the same symbolic-once construction to the full KKT system of optimal power
flow, and investigates GPU $LDL^T$ factorization, for which the one-shot
symbolic phase is a prerequisite. Benchmarks on the classical ill-conditioned
systems of [4] (11, 13, and 43 buses) are in preparation; in infeasible
regions the method returns a least-squares point, which these systems
exhibit at base loading.

---

## References (to be completed)

[1] K. Levenberg, "A method for the solution of certain non-linear problems
    in least squares," *Quart. Appl. Math.*, 1944.

[2] D. W. Marquardt, "An algorithm for least-squares estimation of nonlinear
    parameters," *J. SIAM*, 1963.

[3] S. Iwamoto and Y. Tamura, "A load flow calculation method for
    ill-conditioned power systems," *IEEE Trans. Power Appar. Syst.*, 1981.

[4] S. C. Tripathy, G. Durga Prasad, O. P. Malik, and G. S. Hope,
    "Load-flow solutions for ill-conditioned power systems by a Newton-like
    method," *IEEE Trans. Power Appar. Syst.*, vol. PAS-101, 1982.
    DOI: 10.1109/TPAS.1982.317050.

[5] H. B. Nielsen, "Damping parameter in Marquardt's method," Technical
    University of Denmark, IMM-REP-1999-05.

[6] R. J. Vanderbei, "Symmetric quasidefinite matrices," *SIAM J. Optim.*,
    vol. 5, no. 1, 1995.

[7] T. A. Davis, "Algorithm 849: A concise sparse Cholesky factorization
    package" / LDL User Guide, SuiteSparse. *(exact citation to be fixed)*

[8] P. R. Amestoy, T. A. Davis, and I. S. Duff, "An approximate minimum
    degree ordering algorithm," *SIAM J. Matrix Anal. Appl.*, 1996.

[9] B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, "OSQP: An
    operator splitting solver for quadratic programs," *Math. Prog. Comp.*,
    2020. *(QDLDL factorization)*

[10] W. F. Tinney and J. W. Walker, "Direct solutions of sparse network
     equations by optimally ordered triangular factorization," *Proc. IEEE*,
     1967.

[11] S. Fliscounakis, P. Panciatici, F. Capitanescu, and L. Wehenkel,
     "Contingency ranking with respect to overloads in very large power
     systems taking into account uncertainty, preventive, and corrective
     actions," *IEEE Trans. Power Syst.*, 2013. *(PEGASE test cases)*

---

*Implementation note for the arXiv version: all measurements and the
reference implementation are in the rustpower repository, branch
`symbolic-kkt-lm`; the mapping from the paths of Section 4 to source files is
recorded in `LM_GN_Assembly_Worklog.md`. Reproduction commands are listed
there.*
