# AC-OPF module guide

The conventional baseline remains in `src/opf`. This directory organizes its
optimized counterparts by **what they compute**, then by implementation version.
All named configurations consume the same model and return the same result type.

## Start here

1. `model.rs`: numerical inputs and packed variable order.
2. `configurations.rs`: the assembly/evaluation combination for each experiment.
3. `solution.rs`: solver options, result units, and diagnostics.
4. `assembly/v4/curvature.rs`: readable local branch-curvature equations.
5. `assembly/v5/symbolic.rs`: where each KKT coefficient belongs.
6. `assembly/v5/partitioned.rs`: how those coefficients are filled numerically.

```rust,ignore
use rustpower::new_opf::{Configuration, NewOPFData, PipsOpt};
use rustpower::new_opf::model::OPFData;

// Build the numerical model once, including the intended costs and constraints.
let model: OPFData = /* model supplied by the application */;
let initial = model.warm_x0();
let data = NewOPFData::new(model);
let result = Configuration::V5_6.solve(
    &data,
    initial,
    PipsOpt { cost_mult: 1e-4, ..Default::default() },
);
```

The initial vector is explicit, so comparisons can use exactly the same initial
point. `Configuration::solve` obtains bounds from the model. It creates fresh
strategy workspaces and a fresh linear solver for each call. It does not yet
provide a persistent solver across separate calls.

## Directory map

```text
new_opf/
  model.rs                   numerical model and existing shared cache
  solution.rs                common options, result, and timing types
  configurations.rs          named combinations and solve entry points
  assembly/
    v1/                      adapter to the reference Hessian
    mapped/                  earlier mapped prototype and its cache
    v3/                      numeric, fused, and scalar Hessian experiments
    v4/curvature.rs           exact scalar curvature and local barrier terms
    v5/symbolic.rs            topology-derived KKT layout and mappings
    v5/scatter.rs             V5.2 direct fill with branch scatter
    v5/partitioned.rs         V5.3 branch projection and column fill
  evaluation/
    v1/                      adapter to reference constraints/Jacobians
    v5/nonlinear.rs           V5.5 nonlinear direct fill
    v5/merged.rs              V5.6 direct fill including bound constraints
  interior_point/
    mod.rs                   shared execution entry points
    fused.rs                 existing V5.2/V5.3 driver
    nonlinear.rs             existing V5.5 driver
    merged.rs                existing V5.6 driver
    linear_system.rs         common fused KKT solve and direction recovery
  adapters/ecs/
    components.rs            OPF ECS components
    translate.rs             source tables to OPF ECS limits
    results.rs               solution-to-ECS mapping
  verification.rs            independent mathematical reference utilities
  tests.rs                   cross-version integration tests
```

`mapped` is retained under its descriptive name rather than assigning an
unverified historical version number. The V3 kernels remain available for
experiments; this reorganization does not invent a new complete V3 solver.

## Configurations

| Configuration | Assembly | Constraint evaluation |
|---|---|---|
| `V1` | Reference Hessian and KKT construction | V1 |
| `V4` | Scalar curvature, conventional KKT construction | V1 |
| `V5_0` | V4 curvature with symbolic KKT fill | V1 |
| `V5_2` | Direct KKT fill with branch scatter | V1 |
| `V5_3` | Branch projection and partitioned fill | V1 |
| `V5_5` | V5.3 | Direct nonlinear values/Jacobians |
| `V5_6` | V5.3 | Direct nonlinear values/Jacobians and persistent bounds |

There are no duplicate V3/V4 evaluation folders: those configurations use V1's
evaluator. `Configuration::strategies()` exposes these combinations for reporting.
`Configuration::default()` is V5.6. The historical `pips()` function still selects
V4, preserving existing callers' behavior.

## Numerical input contract

`model::OPFData` reexports the existing reference type; it is not a competing copy
of the model. `NewOPFData` adds the existing mapped symbolic cache. Network-table
conversion remains in `io::pandapower`, outside numerical assembly/evaluation.

| Input | Shape / order | Meaning / unit |
|---|---|---|
| `base_mva` | Scalar | Power base in MVA |
| `ybus` | buses × buses, CSC | Bus admittance in per unit |
| `yf`, `yt` | branches × buses, CSC | Current operators at both branch ends |
| `f_buses`, `t_buses` | One index per branch | Zero-based model bus indices |
| `s_load` | One complex value per bus | Net fixed consumption, P + jQ, per unit |
| `gen_bus`, `cg` | Generator indices; buses × generators | Generator incidence |
| `cf`, `ct` | buses × branches | Endpoint incidence; transpose of PYPOWER's connection convention |
| `vm_min`, `vm_max` | Bus order | Voltage magnitude bounds, per unit |
| `pg_min/max`, `qg_min/max` | Generator order | Generation bounds, per unit |
| `rate_a` | Branch order | Apparent-power rating, per unit |
| `cost_coeffs` | Generator order, `[c2,c1,c0]` | Polynomial evaluated at generation in MW |
| `ref_bus` | One model bus index | Angle fixed to zero radians |
| Initial vector | `[theta, Vm, Pg, Qg]` | Radians, per unit, per unit, per unit |

Use `nx`, `va_range`, `vm_range`, `pg_range`, and `qg_range` from the model instead
of independently calculating offsets. Generator cost rows must be mapped to the
same generator order. The existing converter's default costs are not a substitute
for loading the intended objective.

The model still carries redundant sparse representations for compatibility with
the reference paths. This move does not introduce a new validation builder or
change converter semantics. The external audit currently covers finite branch
ratings; unlimited-branch filtering and complete converter alignment remain
separate follow-ups documented in the technical draft.

## Numerical output contract

Every configuration returns `PipsResult`:

- `x`: last iterate in the same packed model order, even if convergence fails.
- `f`: objective with `cost_mult` removed.
- `converged`, `iterations`, `message`: termination information.
- `lam_eq`, `mu_ineq`: nonlinear equality and branch-flow multipliers.
- `timing`: accumulated computation regions.

The current reference API returns zero placeholders for `mu_lower`/`mu_upper`,
and nonlinear multipliers retain objective scaling. The timing field `solve_sym`
measures the entire first solve region, not isolated symbolic factorization.
These semantics are preserved, not silently changed by the reorganization.
`NewOPFData::write_results` remains available, but its ECS implementation now
lives under `adapters/ecs/results.rs`.

## Boundaries and compatibility

Local derivative formulas do not read pandapower tables. Symbolic analysis owns
patterns and mappings. Evaluation computes constraints and transposed Jacobians.
Assembly computes curvature or writes KKT values directly; V5 is not forced to
construct an intermediate Hessian to fit a common interface.

The optimized interior-point drivers have moved out of the reference folder.
They share the existing bound/stopping helpers and fused linear-system helper.
Their separate iteration loops are deliberately retained in this structural
change. Consolidating those loops requires a subsequent workspace/evaluator
interface and should be checked independently from file moves.

Old `new_opf::v5_kkt`, `new_opf::pips`, `new_opf::problem`, and related module paths
remain aliases to the new locations. The reference `opf::pips` also reexports its
former optimized driver entry points. These aliases contain no copied algorithms.
Internal qualified calls and the OPF audit example use the new module paths.

## Validation

Run release library tests with `klu_dyn`, then build `examples/audit_opf.rs` and
compare every configuration's objective, iteration count, and solution vector
against the existing matched-input audit. The new configuration-dispatch test
also compares named selections with the preserved solver functions. The
reorganization should not change the numerical results.

Verified in this worktree: 103 release library tests passed; all seven solver
configurations produced exactly the same IEEE39/IEEE118 solution vectors,
objectives, and iteration counts as the saved pre-reorganization results.
