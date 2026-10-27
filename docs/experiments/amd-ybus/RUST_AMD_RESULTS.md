# Native Rust AMD comparison

Experimental subcrate: `crates/rustpower-amd`, adapted from BSD3 `amd`0.2.2. Checked and unsafe elimination workspace modes share a const-generic kernel; preparation and postorder retain checked indexing. The library has no C/FFI dependency. Production solver dispatch does not use the Rust experiment.

CPU0 P core, same release binary/options, power profile/EPP performance (governor label powersave), all before/after snapshots unchanged. BTF0 for all paths, default dense10/aggressive true.2 rounds,5 warmups per round,100 alternating samples per path/case/round:200 per cell,3600 recorded cold+reuse pairs. CPU verified before/after each pair.

Ordinary Rust permutations and inverse permutations match original C AMD, seeded Rust permutations match corresponding C seeded helper. Seeded and ordinary permutations are not claimed equal. All paths converge in4/5/6 iterations; max complex voltage difference4.5356643598081007e-13. Fill below is actual L/U per factor, not estimate.

| case | path | order μs | ordering+Symbolic μs | cold μs | reuse μs | L/U each |
|---|---|---:|---:|---:|---:|---:|
| 118 | c_scalar | 31.640 | 32.700 | 87.563 | 33.191 | 768/768 |
| 118 | c_seeded | — | 27.689 | 82.691 | 32.882 | 748/748 |
| 118 | rust_checked_scalar | 33.600 | 34.756 | 89.629 | 33.287 | 768/768 |
| 118 | rust_checked_seeded | 28.872 | 30.000 | 83.942 | 32.626 | 748/748 |
| 118 | rust_unchecked_scalar | 31.863 | 32.971 | 87.430 | 33.206 | 768/768 |
| 118 | rust_unchecked_seeded | 27.892 | 29.012 | 83.169 | 32.690 | 748/748 |
| 39 | c_scalar | 6.498 | 6.864 | 21.916 | 10.522 | 338/338 |
| 39 | c_seeded | — | 4.749 | 20.038 | 10.473 | 336/336 |
| 39 | rust_checked_scalar | 6.929 | 7.224 | 22.322 | 10.494 | 338/338 |
| 39 | rust_checked_seeded | 4.736 | 5.085 | 19.923 | 10.343 | 336/336 |
| 39 | rust_unchecked_scalar | 6.314 | 6.631 | 21.884 | 10.555 | 338/338 |
| 39 | rust_unchecked_seeded | 4.423 | 4.771 | 19.821 | 10.513 | 336/336 |
| 9241 | c_scalar | 2810.823 | 2968.343 | 16922.585 | 11289.930 | 116171/116171 |
| 9241 | c_seeded | — | 2167.968 | 16318.714 | 11509.579 | 115817/115817 |
| 9241 | rust_checked_scalar | 3004.827 | 3172.657 | 17101.611 | 11267.677 | 116171/116171 |
| 9241 | rust_checked_seeded | 2240.367 | 2400.250 | 16579.217 | 11475.778 | 115817/115817 |
| 9241 | rust_unchecked_scalar | 2898.231 | 3069.695 | 17027.445 | 11363.158 | 116171/116171 |
| 9241 | rust_unchecked_seeded | 2164.317 | 2327.575 | 16494.155 | 11503.385 | 115817/115817 |

Cold timing includes one numerical J symbolic cache build, ordering preparation, KLU Symbolic allocation, numerical workspaces and complete NR iterations. Inputs are already-permuted Ybus; loading, mat cloning and bus permutation are excluded. Ordinary ordering consumes the numerical J cache, while seeded ordering derives quotient adjacency from Ybus; no second full J graph is built. All seeds include the same structural eligibility validation, including the C reference. C seeded ordering-only time is null because the helper combines ordering and Symbolic; no artificial split is reported.

All candidates use unchanged KLU numerical factor/refactor/solve. Scalar and Rust seeded routes install supplied P/Q with klu_l_analyze_given(BTF0), using their AMD L estimate. The C seeded helper directly constructs equivalent single-block Symbolic. Numeric permutations match, but Symbolic adapter construction is not identical; ordering-only and total-preparation timings are distinguished.

For9241, Rust unsafe ordinary ordering2.898ms versus C2.811ms(+3.11%); Rust unsafe seeded2.164ms versus Rust unsafe ordinary2.898ms(-25.32%). Full cold17.027->16.494ms(-3.13%); reused11.363->11.503ms(+1.23%). This is no evidence of a reused-solve gain. C seeded total preparation2.168ms versus Rust seeded2.328ms; Rust seeded still has preparation overhead to optimize. No claim that unsafe Rust automatically beats C.

Tests: Rust crate checked/unsafe versus system C oracle across144 structure/control combinations plus empty/jumbled/invalid input checks; debug and release pass. Debug actual39/118/9241 six-path run passes. Default non-KLU87 pass; feature-off KLU87 pass/2 pre-existing ignored; feature-on KLU87 pass/2 pre-existing ignored and4 integration pass. Generic Solve trait restored verbatim; specialized C entry is opt-in klu_ybus_amd at concrete power-flow call sites. Direct generic newton_pf and ordinary klu_dyn/klu keep original behavior.

After timing, returned seeded Info.nz_a_plus_at metadata was corrected from quotient adjacency count to expanded J off-diagonal count. It is unused by kernel, Symbolic adapter or benchmark assertions/timings; checked/release oracle tests were rerun afterward. Source snapshot records that metadata-only correction.

The earlier cpu0-20261026-150850 dataset omitted eligibility validation on C seeded and is superseded for language timing comparison.

Reproduce:

```sh
cargo test --offline -p rustpower-amd
BENCH_CPU=0 REPEATS=100 docs/experiments/amd-ybus/run-rust-amd-bench.sh
```

No commits or pushes. Original main worktree remains clean.
