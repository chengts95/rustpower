# Preserved C seeded AMD and Rust comparison

Persistent worktree: `/home/cts/workspace/rustpower-amd-supervariable`.
Branch: `codex/ybus-amd-supervariable`, base main1ad70a7. No commit or push.
The original `/home/cts/workspace/rustpower` worktree remains separate.

## Feature boundary

- Root `klu_ybus_amd` enables the retained C seeded ordering and concrete KLU
  power-flow preparation. Disabled by default; enabling it also enables KLU.
- Lower `rustpower-sol-klu/ybus_amd` controls the C helper, private Ybus cache,
  preparation methods, and optional `cc` build dependency.
- Plain `klu_dyn`/`klu`: original KLU analysis; no seeded helper compilation.
- `Solve` trait is restored verbatim. Generic `newton_pf<Solver: Solve>` has no
  Ybus preparation method/call. Native/ECS/DCPF call sites are cfg-gated.
- `rustpower-amd` is the separate experimental native Rust comparison, not a
  replacement selected automatically by the production solver.

## Retained sources

- `crates/rustpower-sol-klu/vendor/amd/amd_super2.c`: our seeded C AMD core.
- `crates/rustpower-sol-klu/vendor/amd/ybus_order.c`: quotient graph and KLU adapter.
- `crates/rustpower-sol-klu/vendor/amd/amd_internal.h`, `LICENSE.txt`: upstream
  definitions and BSD3 license; original attribution retained.
- `crates/rustpower-sol-klu/src/ybus.rs`: validation/cache/policy.
- `src/basic/ybus_ordering.rs`: feature-specific concrete preparation boundary.
- `crates/rustpower-amd`: checked/unsafe scalar and seeded Rust experiment,
  upstream Rust-port provenance and C oracle tests.
- `examples/bench_integrated_ybus.rs`, `examples/bench_rust_amd.rs`: actual NR
  and C/Rust comparisons; `tests/ybus_ordering.rs`: lifecycle/entry regressions.

The C AMD core is BSD3; the adapter uses existing KLU public APIs. Vendoring AMD
and feature gating do not assert any change to KLU's own licensing.

## Data and reproducibility

The current C seeded versus Rust seeded comparison is
`C_SEEDED_VS_RUST_SEEDED.md`; the common dataset is
`rust-amd-results/cpu0-20261026-151026`,200 samples per implementation/case.
It records source hashes/snapshots, executable hash, power/CPU checks and raw runs.
The earlier150850 dataset is marked preliminary because C seeded eligibility
validation was missing from its timed scope.

The reported9241 difference7.36% is validation+graph+AMD+Symbolic TOTAL
preparation, not isolated AMD core speed. Different Symbolic adapters remain a
known measurement boundary. See the dedicated report rather than attributing the
whole gap to Rust AMD.

```sh
cargo build --release --features klu_ybus_amd
cargo test --offline --features klu_dyn --lib
cargo test --offline --features klu_ybus_amd --lib --test ybus_ordering
cargo test --offline -p rustpower-sol-klu --features ybus_amd
cargo test --offline -p rustpower-amd
BENCH_CPU=0 REPEATS=100 docs/experiments/amd-ybus/run-rust-amd-bench.sh
```

All new source, restoration material, data and continuation notes live in
persistent directories. A separate full dirty-source/data archive and Git patch
are also saved under `/home/cts/Documents/Codex/2026-10-24/task/rustpower-amd-preservation`.
