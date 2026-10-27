# Experimental native Rust AMD

The default ordering library uses Rust only. Optional `klu` and `c_amd` features
provide the experimental KLU addon and the retained C reference, respectively. Adapted from Richard Lincoln's
`amd`0.2.2 Rust port, retaining its BSD3 license and AMD attribution. It is not a
new independently authored AMD algorithm. See PROVENANCE.md and LICENSE.

Public APIs:

- `order`: ordinary scalar AMD, checked indexing.
- `unsafe order_unchecked`: the same scalar algorithm, unchecked elimination
  workspace indexing. CSC preparation and postorder remain checked Rust.
- `order_ybus`: exact supported Jacobian structure, with initial PQ size2/PV
  size1 groups, checked indexing.
- `unsafe order_ybus_unchecked`: same seeded algorithm, unchecked workspace.

The unsafe APIs are experimental; documented index invariants remain caller
obligations. They are not used by the production solver. Checked and unchecked
kernels share code through a compile-time const policy, avoiding runtime dispatch.

Changes to the imported port:64-bit hash compatible with long C AMD; signed-index
flag limit; saturating small-n dense threshold; preprocessing assertion uses
cleaned nnz; optional seed initialization, group-aware dense/empty handling.
Elimination, absorption and ordering steps remain the imported algorithm.

The seeded route builds quotient adjacency directly from permuted Ybus; it does
not materialize J adjacency first. Full scalar dimension is retained in work and
permutation arrays. Structural eligibility checks conservatively require sorted,
duplicate-free columns, active diagonals, symmetric connected active graph and
at least one slack slot. Seeded permutations need not equal ordinary scalar AMD;
they must match the corresponding C seeded reference for this experiment.

Default builds use system C AMD only as a test oracle; `c_amd` explicitly
compiles the retained seeded C kernel. Tests compare
permutations/inverses/fill estimates across144 structure/control combinations and
empty/jumbled matrices. The root example `bench_rust_amd` compares both Rust modes
and C on39/118/9241, checks the complete NR solve, and measures alternating runs.
All benchmark paths use BTF0 and the same KLU numerical engine. C seeded preparation includes exactly the same Rust structural eligibility check as Rust seeded preparation; raw C helper time alone would omit that validation and is not a fair language comparison.

Run from repository root:

    cargo test --offline -p rustpower-amd
    BENCH_CPU=0 REPEATS=100 docs/experiments/amd-ybus/run-rust-amd-bench.sh

The benchmark is an experiment, not a production integration. It supplies
precomputed P/Q to `klu_l_analyze_given` with BTF disabled (C seeded reference
constructs the same one-block Symbolic directly), sets the AMD L estimate, and
uses original KLU numerical factor/refactor/solve. It builds the full numerical
Jacobian cache exactly once for every route; ordinary scalar ordering consumes
that cache, seeded ordering consumes Ybus. `cold_us` includes J cache preparation,
ordering, Symbolic creation, numerical work buffers and all NR iterations; it
starts from already-permuted Ybus and excludes network loading, input mat cloning
and bus permutation. `reuse_us` resets original V and repeats NR with caches.
`prepare_symbolic_us` is the fair total ordering+Symbolic measurement. C seeded
ordering-only timing is deliberately null because its helper also constructs
Symbolic; no split is inferred.

## Specialized interface

The new `powerflow::{YbusPattern, OrderingWorkspace, Ordering}` API shares one
immutable structural analysis between reduced J generation and ordering, and
encapsulates mutable quotient adjacency. See [POWERFLOW_API.md](POWERFLOW_API.md)
for the code/mathematics mapping and current encapsulation boundary.
Earlier performance records predate this preparation refactor.

## Optional KLU addon

`klu::KluOrdering` owns preparation/cache policy separately from the original
`rustpower-sol-klu::KLUSolver`. The dependency points from this experimental
crate to the unchanged KLU crate. Feature `klu` provides the Rust adapter;
`c_amd` additionally compiles `vendor/amd` and exposes the C reference backend.
The root application opts in through its distinct `AmdNewtonSolver` or
`AmdOrderingPlugin`. Active C experiment sources live in `experiments/` and are
excluded from packaged releases. Their original license notices are retained.
