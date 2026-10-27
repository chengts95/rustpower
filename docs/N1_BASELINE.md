# N−1 implementation and performance checkpoint

This checkpoint preserves the fixed-partition Rust N−1 implementation before
experiments with dispatch queues. Changes to queues, transport chunks or worker
execution should retain scenario decisions and show a meaningful measured gain.

## Implementation

- Optional `experimental_n1` feature; shared baseline, AC/DC Symbolic and branch SoA.
- Worker-private numerical buffers, KLU Common/Numeric and sparse outage patches.
- Shared base-AC initialization; per-round DCPF retries and selected Detailed output.
- Tarjan bridge classification with parallel-device identity and Slack-side counts.
- `prepare_n1_for(world, faults)`, `load_faults`, borrowed `N1Study`, `execute`,
  explicit-subset `run`, and `details` for Rust callers. World/App remain caller-owned.
- Bounded Crossbeam task/result queues, scoped worker threads and a collector.
- Existing portable/AVX2 branch kernels and compact voltage/branch constraint checks.
- Legacy load-table default percentages needed by the case-import path.

## Measured baseline

Four P cores, CPU 0–3; governor=powersave, EPP=performance, Turbo enabled.
Basic ACPF convergence, empty constraint plan, base-AC start, bridge screening,
batch=8, StatusOnly, max_iter=30, tolerance=1e-10. Execution includes study.run
classification, dispatch, thread/App/workspace startup, solve, collection and join.
One-time shared preparation and input loading are recorded separately.

| Buses | Candidates / actual PF | Repetitions per configuration | 1 worker ms | 2 workers ms | 4 workers ms | 4-worker efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 20 / 19 | 201 | 0.242 | 0.194 | 0.281 | 21.5% |
| 30 | 41 / 38 | 101 | 0.530 | 0.355 | 0.371 | 35.7% |
| 57 | 80 / 79 | 101 | 2.776 | 1.693 | 1.365 | 50.8% |
| 89 | 210 / 194 | 51 | 8.413 | 4.651 | 2.779 | 75.7% |
| 300 | 411 / 322 | 21 | 45.371 | 24.001 | 13.899 | 81.6% |
| 1354 | 1991 / 1430 | 3 | 800.974 | 409.649 | 230.846 | 86.7% |
| 2869 | 4582 / 3804 | 3 | 5866.724 | 3000.912 | 1684.674 | 87.1% |
| 6515 | 9037 / 6474 | 1 | 25512.806 | 13231.136 | 7722.467 | 82.6% |
| 9241 | 16049 / 14384 | 1 | 92753.212 | 48054.057 | 28259.399 | 82.1% |

Parallel benefit is already present at 30 buses: 2 workers achieve 1.492× speedup.
At 14 buses, batch=1 gives 0.296/0.216/0.312 ms for 1/2/4 workers over 201 batches;
fixed execution costs remain significant. Scenario decisions match across workers.

With line and bus-voltage constraints enabled, 9241 previously measured 29.322 s
on 4 workers; 1665 islands, 10 numerical failures, 485 constraint violations.
Keep this monitoring policy distinct from the basic ACPF table above.

Details: [usage](N1_USAGE.md), [scalability](N1_SCALABILITY.md),
[6515 RTE](N1_6515_RTE.md), [serial cost estimate](N1_SERIAL_COST_ESTIMATE.md).
Raw JSON, input hashes, measured executable and reproduction scripts remain in
the task's `n1-scalability/`, `n1-6515/`, and `n1-acpf-estimate/` directories.

## Dispatch observations

For 6515, 6474 PF tasks produce 810 task batches and two result batches. Each
worker set also receives 810 group signals in total. Queues
move 24-byte Vec handles; OutageTask is 248 bytes on this build. Task payloads
total 1.53 MiB. The code additionally clones each solved voltage locally,
totalling approximately 644 MiB across the run. These are byte-count observations;
queue waiting, scheduling and copying costs require separate timing.

Transport chunk size and compute/postprocessing batch size can be evaluated
independently. The current checkpoint retains the measured queue implementation.
