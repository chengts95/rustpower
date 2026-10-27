# Experimental AMD addon with an unchanged legacy KLU path

The AMD experiment is isolated in `rustpower-amd` and explicitly selected solver
structures. The original `rustpower-sol-klu` crate, `basic::solver::KLUSolver`,
`DefaultSolver`, native `NewtonSolver`, `NewtonCache`/`newton_pf` and default ECS
run systems match the branch's baseline source. `Solve` is unchanged. Enabling
an experimental feature makes new APIs available; existing constructors keep
the ordinary KLU analysis path.

## Ownership and dependencies

- `rustpower-sol-klu` owns the original common/Symbolic/Numeric pointers and its
  original analysis/factor/refactor/solve/reset API. It has no AMD dependency.
- `rustpower-amd` keeps its pure Rust ordering APIs by default. Optional `klu`
  adds `klu::KluOrdering` and the Rust Symbolic adapter, depending on the legacy
  KLU crate. Optional `c_amd` additionally builds `vendor/amd` through `cc`.
- Production C kernel, C Symbolic helper, headers, license and provenance reside
  under `crates/rustpower-amd/vendor/amd`. Active C measurement/prototype sources
  live under `crates/rustpower-amd/experiments`; they are excluded from publishing.
  Historical ignored result snapshots remain preserved at their recorded paths.
- `KluOrdering` owns exact Ybus keys, policy/backend and statistics independently
  of the legacy KLU object. Preparation validates the same structural
  preconditions, invalidates both KLU caches on changes, and installs Symbolic.

## Explicit native solver

Build with `klu_ybus_amd` for a C-default addon, or `klu_ybus_amd_rust` for a
Rust-default addon. Both feature combinations preserve ordinary constructors.

```rust,ignore
use rustpower::prelude::AmdNewtonSolver;
let mut solver = AmdNewtonSolver::new();
// setup_context / setup_from_nodes and result APIs are inherited from the
// contained legacy NewtonSolver. solve() prepares AMD, then delegates to it.
```

The new type contains a legacy native solver plus `KluOrdering`. It performs
preparation before delegating the entire numerical solve. The original linear
solver retains factor/refactor/solve behavior. The old generic NR loop has no
AMD preparation hook or extra timing fields.

## Explicit ECS plugin and Python interfaces

Rust ECS users install `rustpower::amd::AmdOrderingPlugin` on the power-flow app.
It adds one preparation system after Newton-cache setup and before the existing
solver system set. Existing NR/DCPF systems are unchanged. Custom-solver mode
bypasses this addon. The optional `AmdOrderingState` resource owns its cache.

```python
from rustpower.solver import NewtonSolver, AmdNewtonSolver
ordinary = NewtonSolver()          # original path
experimental = AmdNewtonSolver()  # explicit addon
experimental.set_ybus_ordering_backend("rust")  # c / rust / scalar
print(experimental.ybus_ordering_stats())
# setup_context and solve retain the existing argument/result conventions.
```

For `PowerGrid`, `set_ybus_ordering_backend("c" | "rust")` explicitly installs
and configures the plugin. `"scalar"` disables it. A normal grid has no AMD state
and reports scalar/zero preparations. Shared Python argument adapters are generated
for the two distinct native classes; the original class's method set is retained.

`scripts/bench_py.py` selects the experimental native class only when
`RUSTPOWER_YBUS_BACKEND` is explicitly provided. With the variable unset it uses
the old class and ordinary grid. The original clocks still cover cold
`setup_context` (Ybus permutation) + `solve` + `get_scalc`, and cached grid
`solve` + `post_process`.

## Cache lifecycle and fallback

Keys include exact Ybus column offsets/row indices, npq/npv, BTF and ordering. A changed key
clears KLU Numeric/Symbolic and tells the addon caller to invalidate J cache.
Numeric-only updates reuse Symbolic. Context checks the common-object owner and
a non-null Symbolic before accepting a key hit. Legacy reset/re-setup therefore
forces preparation even when the input arrays are identical. Backend/policy
changes clear both KLU caches through the addon, leaving common settings intact.

Both seeded paths operate on a quotient graph with PQ weight 2 and PV weight 1,
and emit a full scalar permutation. They validate sorted, symmetric active
structure, active diagonals and one connected active component. They directly
install the proven one-block Symbolic with the configured BTF flag; general BTF
is skipped on the optimized path. Unsupported structures/settings retain ordinary
KLU analysis. PQ/PV/Slack assembly layout is unchanged.

## Reproduction and reports

`cargo test --offline -p rustpower-sol-klu` checks the original crate.
`cargo test --offline -p rustpower-amd --features c_amd` checks addon context,
reset/owner binding, policy changes and fallback. Root `tests/ybus_ordering.rs`
checks explicit opt-in, original three-field KLU construction, native/ECS
convergence, C/Rust permutations/fill/voltages, switching and topology changes.
`scripts/test_ybus_backends.py` checks both Python classes and explicit grid opt-in.

Historical performance reports retain their measured implementation and scope.
The old NR timing fields have been removed; the integration trace now reports
J preparation as unavailable (`null`) and labels the measured preparation sum
with `net_preparation_includes_j=false`. Full solve and bus-permutation clocks
remain available without modifying the old numerical path.

Compiler experiments are in [AMD_CODEGEN_RESULTS.md](experiments/amd-ybus/AMD_CODEGEN_RESULTS.md).
For LTO, `cargo rustc --lib --crate-type cdylib` selects a single compatible
output; the mixed rlib/cdylib build otherwise omits final LTO flags. Default
compiler settings remain intact.

## Isolation validation (2026-10-27)

Legacy KLU crate and the six original native/NR/ECS source paths were compared
against HEAD and are unchanged. The old three-public-field KLU construction
compiles in the integration test. Rust library tests passed 87 with 2 ignored
both with ordinary KLU and with the addon enabled; 7 addon integration tests,
4 C/Rust context tests, original KLU tests and all 58 Python tests passed.
The Rust-only `rustpower-amd/klu` configuration also passed its 3 context tests.
All examples compile. Final release was rebuilt after the shared eligibility
helper was deduplicated; the 7 integration tests and Python backend tests were
rerun on the final source.

A release extension was loaded through an explicit PYTHONPATH overlay; the
installed virtualenv package was preserved. The original `scripts/bench_py.py`
clocks cover Ybus permutation in `setup_context`, all Newton iterations including
first numerical factorization, and `get_scalc`. Each row below used 30 cold and
40 hot samples; backend order was scalar/C/Rust followed by Rust/C/scalar.
The scalar comparison explicitly disables the addon on the new wrapper.
System dynamic KLU and ordinary release flags were used. No CPU pinning was set.

| 9241 PEGASE cold mean (ms) | Round 1 | Round 2 | Combined mean |
|---|---:|---:|---:|
| Scalar | 22.392 | 19.523 | 20.957 |
| C seeded | 17.589 | 17.446 | 17.517 |
| Rust seeded | 17.856 | 17.646 | 17.751 |

Combined cold means improve 16.4% for C and 15.3% for Rust. Scalar round drift
shows the measurement variability. Core hot means are 11.917/12.337/12.299 ms
(scalar/C/Rust), and grid plus postprocess means are 12.324/12.613/12.744 ms.
The experiment retains a cold-path benefit and shows a small hot-path cost.
All backends converged in 6 iterations; C/Rust reported one successful
preparation and scalar reported zero on both native and grid objects.

Raw records, logs, driver and pre-refactor backup are preserved at
`/home/cts/Documents/Codex/2026-10-26/task/amd-isolation/`.
