# Exact scalar AMD from Ybus: initial optimization prototype

New research path only. Accepted weighted/pregrouped C and Rust paths remain available and do not enable sharing. Production Solve trait/J assembly/PQ-PV-Slack layout unchanged. New implementation: crates/rustpower-amd/src/exact.rs.

Three same-binary alternatives:

- original_scalar: existing unsafe Rust scalar AMD, supplied the already available complete J pattern.
- direct_input: directly generate scalar AMD mutable input from the analyzed Ybus layout, skip intermediate J input CSC/A+A' count and construction. Singleton Nv=1, scalar IDs/N/controls unchanged; no premerging.
- first_touch: same direct input plus exact first-touch payload sharing and own-twin mass-elimination shortcut. Accepted old weighted algorithm is a different path, not the timing denominator.

| case | original scalar μs | direct input μs | shared first touch μs |
|---|---:|---:|---:|
| 39 | 3.896 | 3.013 | 3.766 |
| 118 | 10.834 | 8.847 | 11.399 |
| 9241 | 3171.016 | 2680.567 | 2732.313 |

Final dataset cpu0-20261026-181544: CPU0 P core asserted around every sample; performance power profile/EPP, before/after snapshots unchanged; same release opt3 binary, trace disabled.2rounds*100samples per path/case=200 per cell,1800 records. Five warmups/round. Same original input and controls dense10/aggressive true. Common Ybus structural analysis and numerical-J pattern availability are outside timers for all routes; each timer includes its own ordering input/workspace allocation, core and postorder. This is ordering-only, not NR or KLU timing.

9241: direct input3.171016->2.680567ms(-15.47%); first touch2.732313ms(-13.83% versus original, +1.93% versus direct). Shared kernel is not yet a net improvement over direct input. Initial per-pair Vec allocation prototype was slower; final uses one continuous payload pool with typed offset/count descriptors. Do not compare absolute values between the preliminary runs with different power/thermal histories; use each run's alternating control.

Final first-touch counters9241: see summary.json. The total avoided suffix checks includes own-twin skips. Shared suffix traversal still scans the extra excluded self ID once while generating a bus payload, so net saved examined entries subtract the payload count from suffix_scans_avoided. These counters do not count all dynamic adjacency scans in AMD. Each scalar still receives its own survivor writes, bucket/hash/parent updates in original order. No false claim of eliminating those writes.

Boundary: current prototype still materializes AMD's own full scalar mutable adjacency arena. It avoids intermediate J CSC and generic input preparation, but is NOT a completed lazy Seed/Dynamic arena. Input view validation is conservatively limited by the existing YbusPattern contract (connected, symmetric, sorted, diagonal-present). A general/disconnected-input implementation remains future work; unsupported inputs are not silently accepted.

Exactness acceptance: actual39/118/9241 pivot streams, first8 candidate states, each variable-update ordered list/degree/hash/mass-elimination parent, supervariable merge events and final permutation match original scalar path exactly under trace. Final permutation/inverse, L estimate and LU flop estimate asserted for every measured sample. Bucket reinsertion and postorder intermediate operations are not separately traced; later pivots and final output provide additional checks, not a full formal bisimulation proof.

96 mixed-type sparse/dense/control scenarios (192 direct/shared comparisons) compare direct/shared outputs to checked scalar oracle, including all-PV/all-PQ, original/aggressive off, dense threshold controls; debug trace checks and C-oracle unit tests pass. Full literal ordinary C reference equality for actual inputs is inherited from saved scalar-reference correspondence tests; new traces compare with the unchanged Rust scalar control.

No numerical KLU/NR timing claim. Same permutation/factor estimates do not substitute for an unperformed end-to-end integration benchmark.

Reproduce:

```sh
cargo run --offline --features experimental_amd_trace --example bench_first_touch -- 0 /path/to/empty-smoke.jsonl
cargo test --offline -p rustpower-amd --features trace
BENCH_CPU=0 REPEATS=100 docs/experiments/amd-ybus/run-first-touch.sh
```

Sources and final records saved persistently; no commits/pushes.
