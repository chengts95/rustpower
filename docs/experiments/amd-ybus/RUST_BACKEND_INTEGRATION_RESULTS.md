# Rust seeded AMD in the native and ECS solve paths

The opt-in `klu_ybus_amd_rust` feature connects `rustpower-amd` to the existing
KLU Ybus preparation boundary and selects Rust by default. The C reference is
retained and selectable. Ordinary `klu_ybus_amd` builds continue to select C.

The Rust adapter calls `YbusPattern::analyze` and the checked
`OrderingWorkspace::order`, then creates a one-block KLU Symbolic using public
KLU allocation/free functions. It uses the same P/Q, BTF policy, structural rank,
fill estimates and numerical factor/refactor/solve as the C adapter. It generates
quotient adjacency directly from Ybus; J numerical assembly remains unchanged.

## End-to-end comparison, 2026-10-27

One release extension, features `klu_ybus_amd_rust,python`, the same dynamically
linked system KLU for all paths, CPU0. Static KLU libraries were unavailable in
this environment. The virtual environment's installed extension was unchanged;
PYTHONPATH selected the freshly built extension. Governor `powersave` and EPP
`performance` were identical before and after; no power settings were changed.

The driver calls the original `scripts/bench_py.py::run_benchmark`. Cold clocks
include `setup_context` (including Ybus permutation), complete `solve` with first
factorization, and `get_scalc`. Cached core clocks include initial-voltage reset,
`solve` and `get_scalc`. Grid clocks include `solve` and `post_process`. Input
network conversion and initial partition-vector construction remain outside the
original clocks. Each cold sample creates a new solver. Backend selection is
outside each clock. Cached solves start from the same initial voltage.

39, 1354 and 9241 were measured. Two rounds: scalar/C/Rust, then Rust/C/scalar;
50 cold and 100 cached samples per case/backend/round. Values below are means
within each round, matching the original script, not medians.

### 9241, milliseconds

| Round | Backend | Cold | Cached core | Cached Grid + postprocess |
|---|---|---:|---:|---:|
| 1 | Ordinary KLU | 23.544 | 12.197 | 12.755 |
| 1 | C seeded | 17.776 | 12.349 | 12.960 |
| 1 | Rust seeded | 17.853 | 12.476 | 12.641 |
| 2 | Rust seeded | 18.043 | 12.395 | 12.973 |
| 2 | C seeded | 17.678 | 12.550 | 13.026 |
| 2 | Ordinary KLU | 19.368 | 12.122 | 12.732 |

Rust cold time was +0.43% and +2.07% relative to C. Cached core differences changed
sign between rounds (+1.03% and -1.23%). Ordinary KLU cold measurements varied
substantially between rounds; these samples do not isolate a stable total
speedup percentage against that baseline. No C-vs-Rust kernel-only inference is
made from these full solve clocks. This integration uses the checked Rust kernel.

Every measured native and grid seeded path reported `optimized=true`, the
requested backend, and exactly one Symbolic preparation on the reused solver.
All measured paths converged; 9241 used six NR iterations. Raw JSON, output logs,
source/extension SHA256 and build metadata are retained in the task workspace's
`rust-amd-integration` output directory.

## Validation

- Rust feature: 87 root library tests passed, two existing tests ignored.
- Native/ECS integration: six tests passed, including C/Rust P/Q, Symbolic
  estimates, actual L/U and voltage comparisons for 39/9241 under BTF0/1,
  runtime backend switching, and cache invalidation/reuse.
- Original C-only integration: five tests passed.
- KLU crate: Rust-enabled and C-only suites each passed five tests; feature-off
  suite passed two tests. Rust AMD crate tests and doctest passed.
- Python: all 57 regression tests passed, including Native/ECS three-backend
  selection, repeated switches, numerical equivalence and invalid-name handling.

## Usage

```sh
cargo build --release --features klu_ybus_amd_rust,python
# With the newly built extension installed or selected via PYTHONPATH:
RUSTPOWER_YBUS_BACKEND=rust python scripts/bench_py.py
# Select c or scalar with the same environment variable for comparison.
```

Python Native/ECS solvers also accept
`set_ybus_ordering_backend("rust" | "c" | "scalar")` and expose
`ybus_ordering_stats()`. Default builds remain unchanged. No commits or pushes
were made for this integration.
