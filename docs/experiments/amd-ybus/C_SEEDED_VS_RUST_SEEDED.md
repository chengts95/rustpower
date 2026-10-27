# Modified C seeded AMD vs Rust seeded AMD

This report selects the requested two implementations from the already recorded, valid same-batch dataset `cpu0-20261026-151026`. No benchmark was rerun and no algorithm or production API was changed for this report. The C reference here is OUR MODIFIED seeded C AMD, not system scalar AMD.

Both receive the same permuted Ybus, PQ2/PV1 groups and scalar variable IDs, dense10/aggressive true,64-bit indices, BTF0. Same release binary, opt-level3 (C cc::Build inherits release OPT_LEVEL; no CFLAGS overrides recorded), CPU0 P core, performance profile and EPP, unchanged before/after power snapshots. Two rounds,5 warmups/round,100 alternating samples/path/case/round:200 samples per table cell.

| case | C seeded preparation μs | Rust unsafe seeded preparation μs | Rust change | C first NR ms | Rust first NR ms | Rust change |
|---|---:|---:|---:|---:|---:|---:|
| 39 | 4.749 | 4.771 | +0.45% | 0.020038 | 0.019821 | -1.09% |
| 118 | 27.689 | 29.012 | +4.78% | 0.082691 | 0.083169 | +0.58% |
| 9241 | 2167.968 | 2327.575 | +7.36% | 16.318714 | 16.494155 | +1.08% |

Preparation is structural eligibility validation + quotient graph/workspace preparation + AMD + KLU Symbolic creation. It is the available comparable TOTAL preparation boundary, not isolated AMD core time. Both include exactly the same Rust structural eligibility check. Rust-only ordering (validation+graph+kernel+permutation conversion, excluding Symbolic) is4.423/27.892/2164.317us; C helper combines ordering and Symbolic and has no separate ordering/core timer. Do NOT subtract medians of these scopes to infer C kernel speed or attribute the7.36% total gap entirely to translation.

Adapter difference: C directly constructs one-block Symbolic in rustpower_ybus_analyze; Rust emits P/Q then installs them through klu_l_analyze_given with BTF0 and supplies the AMD L estimate. Both use the original KLU numeric factor/refactor/solve. First-NR measurement includes one numerical J cache build, all preparation/workspaces and complete NR from the same V; already-permuted network loading/cloning/bus permutation are outside scope.

Every Rust seeded permutation is checked elementwise against our modified C seeded reference. Actual L/U counts each are336/748/115817; same iteration counts4/5/6, all converged; max voltage error against the numerical reference4.5356643598081007e-13. C and Rust seeded produce identical checked numerical results.9241 reused NR C11.509580ms versus Rust11.503385ms; no meaningful reused performance difference established.

Conclusion: current Rust unsafe SEEDED preparation is+0.45%/+4.78%/+7.36% versus our C SEEDED reference on39/118/9241.9241 full first NR is+1.08%. This is a measurable current pipeline overhead; pure AMD elimination-core overhead remains unseparated because no core timers exist.

C source: vendor/amd/amd_super2.c (renamed/seeded SuiteSparse AMD3.3.4), ybus_order.c adapter. Rust source: rustpower-amd adapted from amd0.2.2 with matching seeded initialization and const-generic unsafe workspace. Base repository commit1ad70a7a5834dd6d93c5c967115beb63e853f91b; all experimental modifications uncommitted. Full source identities in seeded-source-sha256.json, C snapshots in c-seeded-reference, Rust snapshot in rustpower-amd-source. Core hashes match measured snapshot; a subsequent returned Info.nz_a_plus_at metadata-only correction is documented in RUST_AMD_RESULTS.md and does not enter timing or Symbolic setup.

Reproduce existing comparison:

```sh
BENCH_CPU=0 REPEATS=100 docs/experiments/amd-ybus/run-rust-amd-bench.sh
```

Select c_seeded and rust_unchecked_seeded from summary.json. Binary SHA256bf7594cf7701624351ae2611b7ad3e5dbecf60d7d4d3fe86dd7cb07aa5f88b2f. Original C AMD is an additional oracle in that script and is not the denominator in this report.
