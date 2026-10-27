# N−1 studies: fault sets, screening, and detailed reruns

Enable `experimental_n1` for independent line and two-winding transformer outages
with a fixed PQ/PV/Slack partition. Each selected Entity defines one scenario; the
source World retains its base state. Initialization supports a shared converged
base AC voltage or per-scenario DCPF. Q-limit enforcement is a future extension.

See [Parallel execution design](design/n1_parallel_execution.md) for the producer,
worker, and collector model, memory budgets, result retention, and extension paths.

## Rust entry points

Initialize the source World, attach the required `N1Watch` components, and prepare
the shared snapshot:

```rust,ignore
use rustpower::n1::*;

let faults = N1FaultSet::out_of_service([line_entity, transformer_entity]);
let prepared = prepare_n1_for(grid.world_mut(), faults)?;
let study = prepared.study()?;
let options = N1RunOptions {
    config: N1Config { workers: 4, batch_size: 8, ..Default::default() },
    ..Default::default()
};
let mut failed = Vec::new();
let report = study.execute(options, |batch| {
    failed.extend(batch.events.into_iter()
        .filter(|event| event.outcome == N1Outcome::NumericalFailure)
        .map(|event| event.outage_entity));
})?;
```

- `prepare_n1_for(world, faults)` binds a selection. Duplicate Entities keep their first occurrence.
- `prepare_n1(world)` prepares a baseline. An unbound `study.execute` selects all initially active branches.
- `prepared.load_faults(faults)` validates and replaces the selection atomically. An explicit empty set executes zero scenarios.
- `study.run(&faults, options, collect)` executes a selection; `all_branch_outages()` constructs the complete active-branch set.
- `study.details(&faults, options, collect)` emits status and configured detail payloads for every selected numerical scenario.

`PreparedN1` owns the snapshot; `N1Study` borrows it. Prepare a new snapshot after
changing the source model. `study()` checks the unchanged baseline against its
monitoring rules and returns an error identifying an existing constraint violation.

## Monitoring and result modes

Attach rules to source branch or bus Entities. Preparation compiles them into a
shared immutable `WatchPlan`:

```rust,ignore
world.entity_mut(line_entity).insert(N1Watch {
    rules: vec![WatchRule {
        rule_id: 1,
        condition: WatchCondition::BranchLoadingAbove(100.0),
    }],
});
world.entity_mut(bus_entity).insert(N1Watch {
    rules: vec![WatchRule {
        rule_id: 2,
        condition: WatchCondition::BusVoltageOutsideLimits,
    }],
});
```

Bus bounds come from `VmLimit<PerUnit>` on that Entity. Branch rules also support
current thresholds. `SolveFailed` monitors numerical failures; `Always` is a
diagnostic rule whose hits are separate from security-violation counts.

| Entry or mode | Passing scenarios | Rejected scenarios | Detail payload |
| --- | --- | --- | --- |
| Default `StatusOnly` | Update local aggregates | Emit status | Stop at the first violated constraint |
| Configured `Detailed` | Emit when a rule matches | Emit when a rule matches | All configured rule hits |
| `study.details` | Emit each selected numerical status | Emit each selected numerical status | Configured rule hits |

`N1Event` carries scenario and outage identities, source Entity, solver stage,
outcome, iteration count, residual, error, linear-solver statistics, and `WatchHits`.
Hits carry the rule ID, source Entity, target, and scalar value. Multiple rules on
a branch share one `BranchFlowResult`. Detail evaluation follows configured targets.

Numerical outcomes are `Passed`, `ConstraintViolation`, and `NumericalFailure`.
Predispatch topology rejections are returned in `N1StudyReport.islands`.
`processed`, `passed`, `violated`, `numerical_failed`, and `islanded` provide totals.
Passing scenarios in the default mode only update worker-local `N1Summary`;
worker summaries are merged when the run finishes.

Monitoring borrows the worker's current voltage before the next scenario overwrites
it. Each worker reads the shared AC start and copies it into its private writable
iteration buffer. Per-scenario result-voltage allocations and copies are avoided.
The public `N1Worker` solve methods still return owned voltage snapshots for callers
that explicitly retain a solution.

## Failure selections and reruns

Reuse the same study for a DCPF-initialized retry or detailed inspection:

```rust,ignore
let retry = N1FaultSet::out_of_service(failed);
let retry_report = study.run(&retry, N1RunOptions {
    initialization: Some(N1Initialization::DcPerCase),
    emit_all: true,
    ..options
}, |batch| { /* consume every retry status */ })?;
study.details(&retry, options, |batch| { /* consume configured details */ })?;
```

`BaseAc` solves the unchanged baseline during preparation and initializes every
scenario from that converged voltage. `DcPerCase` solves the patched scenario's DC
system before Newton. Both paths reuse the prepared AC/DC analyses; every worker
owns its writable buffers and KLU Numeric/Common objects.

An unusable refactor/solve, including non-finite output, triggers a full factorization
of the current matrix and a solve retry with the original RHS. Follow-up assessment
and requeue orchestration are described in the design document.

## Scheduling, topology, and memory

The calling thread produces tasks. Each of P scoped worker threads owns an App;
a separate collector invokes `FnMut(N1EventBatch) + Send`. Both Crossbeam bounded
MPMC queues hold up to 2P batches. Workers receive groups, solve and monitor each
scenario, and send nonempty event batches. Delivery follows completion order;
scenario and device identities associate results with their inputs.

The collector callback can aggregate, forward, or write batches. The current
collector is a single thread. Independent downstream batch sizes and multiple
collectors are extension designs.

Queue slots are preallocated. Task Vecs and nonempty event batches still allocate
per group. Optional task-block recycling and Auto/Full/Budget policies are planned;
the numerical workspaces already persist within a worker's run.

Defaults: `workers=1`, `batch_size=8`, `max_iter=30`, `tolerance=1e-10`.
Batch size is 1 through 32. `max_batch_bytes` defaults to 64 MiB and bounds the
estimated task/event/detail scratch for a group. Worker numerical buffers, KLU
Numeric, stacks, and retained consumer output have separate memory costs.

Preparation runs one Tarjan DFS over the branch multigraph, retaining bridge
information, subtree intervals, and Slack counts. A bridge-outage query is O(1),
with parallel branches distinguished by device identity. High-level studies reject
newly separated partitions lacking a fixed Slack before numerical dispatch and
record them as islands. Each Ybus patch addresses absolute CSC slots; a worker
restores the preceding patch before applying the next scenario.

Entities trace results to source components and names. Persistent or cross-process
results also use stable device type/ID and a baseline identity. The retry example
checks that a recorded catalog index still identifies the expected device.

## Two Rust examples

```sh
# CASE accepts 39, 9241, or a pandapower CSV ZIP path.
# Use a model whose baseline satisfies the selected monitoring rules.
CASE=/path/network.zip

# Screening: write rejected numerical statuses; print aggregate counts.
cargo run --release --features experimental_n1 --example n1_run -- "$CASE" 4 8 status events.json base-ac

# Details: write every selected numerical status and configured rule payloads.
cargo run --release --features experimental_n1 --example n1_run -- "$CASE" 4 8 details details.json base-ac

# Extract NumericalFailure records and rerun with per-case DCPF initialization.
cargo run --release --features experimental_n1 --example n1_retry_dc -- "$CASE" events.json retry.json 4 30
```

Example monitoring covers line loading above 100% and bus voltage bounds;
transformers remain outage candidates. `n1_run` arguments are CASE, workers, batch,
status/details, output file, base-ac/dc, and optional max_iter (default 30).
`examples/common/n1.rs` shares loading and monitoring helpers. An empty retry
selection produces a message explaining that the input contains no numerical failures.
Use the same source snapshot for screening and file-based retries. Island totals
are printed separately and remain available in the Rust report.

## Source map and checks

| File | Responsibility |
| --- | --- |
| `src/n1/study.rs` | Fault selections, Entity resolution, high-level runs and reruns |
| `src/n1/execution.rs` | Worker Apps, queues, aggregates, borrowed-voltage monitoring, event batches |
| `src/n1/solver.rs`, `numeric.rs` | Shared analyses, private numerical solves, Jacobian fill |
| `src/n1/outage.rs`, `topology.rs` | Numerical patches, bridges, and partition metadata |
| `src/n1/postprocess.rs`, `watch.rs` | Shared SoA, portable/AVX2 kernels, and rules |
| `src/n1/runner.rs` | Baseline preparation and the earlier low-level base-batch API |

Six integration entry points cover study, execution, outage patches, postprocessing,
watches, and topology. They retain independent full-rebuild comparisons, cross-thread
Symbolic lifetime checks, independent stamping, kernel agreement, output-mode
agreement, and topology mapping. Traditional cache reuse is covered separately in
`tests/klu_cache_reuse.rs`.

```sh
cargo test --features experimental_n1 --lib n1::
cargo test --features experimental_n1 --test n1_study --test n1_execution --test n1_outage_patches --test n1_postprocess --test n1_watch --test n1_topology --test klu_cache_reuse
cargo test --features experimental_n1 --example n1_run
```

## Python bindings

Build with both `python` and `experimental_n1`, for example in your virtualenv:

```sh
maturin develop --release --features python,experimental_n1
```

```python
import rustpower as rp
import pandas as pd

grid = rp.PowerGrid("network.zip", qlim=False)
# Pass line/transformer handles in faults, or omit it for all active branches.
study = grid.prepare_n1()
report = study.run(workers=16, batch_size=8)
print(report.summary)

retry = study.run(report.numerical_failures, initialization="dc", workers=16)
details = study.details(report.rejected_faults, workers=16)
events = pd.DataFrame(details.events)
hits = pd.DataFrame(details.hits)
flows = pd.DataFrame(details.branch_results)
```

The default checks numerical convergence. Enable `voltage_limits=True` and/or
`line_loading_limit=100.0` during preparation for security rules. The baseline must
satisfy those rules. Temporary source-World watches used for preparation are restored;
the snapshot owns the compiled plan.

`N1Faults` retains its source grid and can be passed to a study of that grid.
Selections deduplicate handles and preserve an explicit empty set.
`report.numerical_failures` selects numerical failures; `rejected_faults` selects
numerical or constraint failures; `island_faults` separately selects topology
rejections. Foreign-grid handles or selections produce an error.

The Python study owns `PreparedN1` and a reference to its source grid. Each run
uses the Rust high-level entry, sharing patterns and analyses. High-level validation,
Entity lookup construction, and execution are included in `summary.execution_seconds`.
`summary.preparation_seconds` describes the original preparation.

`run` and `details` release the GIL for native execution. Workers and the collector
handle Rust data; Python/NumPy columns are created on access after return. Existing
studies keep their frozen snapshot when source data changes.

Passing screening scenarios contribute only to aggregates. Reported events, including
all selected numerical statuses from `details`, are retained in a Rust result container
until Python consumes or drops the report. Numeric event columns are NumPy arrays;
category/error columns are lists. `hits.event_row` links to event rows;
`hits.result_index` links to branch-result rows, with -1 for no branch payload.
`flows.event_row` also identifies the source event. Missing scalar values use NaN.
Entity columns preserve source u64 identifiers; stable device type/ID supports
persistent association. Returned arrays own their memory independently of the report.

Python checks are in `scripts/test_n1_api.py` and require NumPy and pandapower.

## Performance tools

Research binaries live in `tools/n1` and require the feature:

```sh
cargo run --release --features experimental_n1 --bin bench_n1 -- 9241 4 8 /tmp/n1-results base-ac status bridges
cargo run --release --features experimental_n1 --bin bench_n1_scale -- case14 /path/case14.zip 4 201 /tmp/n1-scale.json
cargo run --release --features experimental_n1 --bin n1_memory_layout
```

### One-versus-four-worker size sweep

```sh
cargo build --release --features experimental_n1 --bin bench_n1_scale
# Use a Python environment with pandapower to export missing CSV ZIP inputs.
python tools/n1/scan_scale.py --cpus 0-7 --out /tmp/n1-scale-4
```

The sweep covers 14/30/57/89/300/1354/2869/9241 buses with 1 and 4 PF workers.
Both counts share the same CPU set. The default 0–7 leaves scheduling capacity
for production and collection; use `--cpus 0-3` for a four-physical-core allowance
when those IDs match your machine's topology.

Repeat counts for 14/30/57/89/300 are 201/101/101/51/21. Large cases default to one
run; `--large-repeats 3` increases this. `--cases case14 case30 case300` selects a
subset, `--case-dir PATH` reuses ZIPs, and `--binary PATH` selects an executable.

The terminal reports median times, speedups, and efficiencies. `summary.csv` and
`summary.json` retain the comparison. Individual JSON files retain preparation,
first-call end-to-end time, samples, and actual affinity. Metadata records executable
and input hashes, topology, power settings, and commands. The driver checks scenario
decisions and iteration totals across worker counts. `scaling_cur_freq` snapshots
are instantaneous frequency readings. Choose a fresh output directory for each sweep.
Each benchmark process reuses its prepared baseline across repetitions and recreates
workers for every whole-batch call.

`bench_acpf_floor` and `n1_bevy_memory` remain available for cost/memory diagnostics.

## Measurement history (2026-10-27)

[performance-history.json](../tools/n1/results/performance-history.json) preserves
all series, power provenance, earlier observations, and archived experiments.
Modern case9241 runs have 16049 candidates, 14384 numerical solves, 1665 islands,
and 10 numerical failures. Execution includes the full pipeline through thread joins;
shared preparation is separate. First-call end-to-end fields begin before model loading.
Energy estimates below use execution time.

### AMD AI Max 395: default frequency and SMT

The initial 1/2/4/8/16-worker series used three repetitions per configuration.
The reported terminal lines omit affinity and binary hashes. The later 32-worker
result is approximate. This series uses its own 84.257541-second single-worker baseline.

| Workers | Execution s | Speedup | Efficiency |
| ---: | ---: | ---: | ---: |
| 1 | 84.257541 | 1.000x | 100.0% |
| 2 | 42.832640 | 1.967x | 98.4% |
| 4 | 22.997484 | 3.664x | 91.6% |
| 8 | 15.193549 | 5.546x | 69.3% |
| 16 | 8.559683 | 9.844x | 61.5% |
| 32 | ~9.500000 | 8.869x | 27.7% |

32 workers use SMT on 16 physical cores. The associated full-load discussion reports
approximately 120 W and 4.6 GHz. Sustained effective clocks for each run were not supplied.

### AMD: CPU 0–7 affinity trials

| PF workers | Allowed CPUs | Execution s | Repeats |
| ---: | --- | ---: | ---: |
| 4 | 0-7 | 23.181788 | 3 |
| 5 | 0-7 | 20.045501 | 1 |
| 6 | 0-7 | 17.999506 | 1 |
| 8 | 0-7 | 15.380010 | 1 |

A separate four-worker observation was approximately 23.3 seconds at 5.0 GHz.
The CPU allowance applies to the producer and collector too; individual workers
can migrate within it.

### AMD: fixed 3 GHz

The prior single-worker baseline is 145.29 seconds; earlier four/five-worker times
are 38.483238/about 32 seconds, with repetition counts unspecified. The latest
8–16-worker configurations each used three repetitions.

| Workers | Execution s | Speedup | Efficiency | Repeats |
| ---: | ---: | ---: | ---: | ---: |
| 1 | ~145.290000 | 1.000x | 100.0% | unspecified |
| 4 | 38.483238 | 3.775x | 94.4% | unspecified |
| 5 | ~32.000000 | 4.540x | 90.8% | unspecified |
| 8 | 19.980268 | 7.272x | 90.9% | 3 |
| 10 | 16.842420 | 8.626x | 86.3% | 3 |
| 12 | 13.859751 | 10.483x | 87.4% | 3 |
| 14 | 12.140084 | 11.968x | 85.5% | 3 |
| 16 | 11.087117 | 13.104x | 81.9% | 3 |

Reported power is about 26 W at four workers, 48–50 W across 10–12 workers, and 60 W
at 14 workers. An earlier 16-worker/11.3-second run also reported 60 W. The four-worker
26 W figure replaces the earlier 30 W estimate. Rounded 38-second and 11.3-second
observations remain in the history; the table uses the newer precise times.
The latest raw terminal lines are in [ai-max395-3ghz.json](../tools/n1/results/ai-max395-3ghz.json).

### Intel 288V: historical four-P-core sweep

BaseAc, batch=8, StatusOnly, empty rule plan, max_iter=30, tolerance=1e-10; CPUs 0–3.
This code stage precedes removal of per-case result-voltage snapshots. Per-scenario
decisions agreed across 1/2/4 workers in every case.

| Buses | Candidates/solves | Repeats | 1 worker ms | 2 workers ms | 4 workers ms | Four-worker efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 20/19 | 201 | 0.242 | 0.194 | 0.281 | 21.5% |
| 30 | 41/38 | 101 | 0.530 | 0.355 | 0.371 | 35.7% |
| 57 | 80/79 | 101 | 2.776 | 1.693 | 1.365 | 50.8% |
| 89 | 210/194 | 51 | 8.413 | 4.651 | 2.779 | 75.7% |
| 300 | 411/322 | 21 | 45.371 | 24.001 | 13.899 | 81.6% |
| 1354 | 1991/1430 | 3 | 800.974 | 409.649 | 230.846 | 86.7% |
| 2869 | 4582/3804 | 3 | 5866.724 | 3000.912 | 1684.674 | 87.1% |
| 9241 | 16049/14384 | 1 | 92753.212 | 48054.057 | 28259.399 | 82.1% |

The separate 6515 RTE run had 9037 candidates, 6474 numerical solves, 2563 islands,
2 numerical failures, and 21385 AC attempts. One complete run per 1/2/4 workers took
25.512806/13.231136/7.722467 seconds; four-worker efficiency was 82.6%.
A case14 batch=1 supplement used 201 repetitions: 1/2/4 workers took
0.295512/0.215572/0.312129 ms.

### Intel: current four-P-core versus all-eight-core end-to-end run

Commit a3a4375, release defaults, one full run per configuration, executed sequentially.
The system identifies Core Ultra 9 288V: CPUs 0–3 are P cores; 4–7 are E cores.
Governor powersave, EPP performance, turbo enabled; settings were retained.

| Configuration | Execution s | Load-to-return s | Average busy cores | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| 4 workers, CPUs 0-3 | 30.386957 | 30.486794 | 3.84 | 106.85 |
| 8 workers, CPUs 0-7 | 21.641621 | 21.740990 | 7.48 | 129.54 |

Per-scenario decisions and 43709 attempted iterations agree. Average busy cores is
child-process user+system CPU time divided by process wall time, including production
and collection. All-core end-to-end throughput increased 40.2%.
Hardware package-energy reads were unavailable. The owner reports this Lunar machine
at approximately 30 W; the energy estimates below use that observation.

### Approximate energy per batch

Power observations are treated as approximate whole-run averages in P times T.
The 48–50 W range covers the 10–12-worker group. The 16-worker 60 W estimate comes
from the earlier 11.3-second run.

| Configuration | Execution s | Approximate power W | Estimated energy J |
| --- | ---: | ---: | ---: |
| all8 current | 21.642 | 30 | 649 |
| 3GHz four workers | 38.483 | 26 | 1001 |
| 3GHz ten workers | 16.842 | 48–50 | 808–842 |
| 3GHz twelve workers | 13.860 | 48–50 | 665–693 |
| 3GHz fourteen workers | 12.140 | 60 | 728 |
| 3GHz sixteen workers latest time | 11.087 | 60 | 665 |
| high-frequency four workers affinity0-7 | 23.182 | 70–72 | 1623–1669 |

### Archives and next steps

The history JSON also retains the early 16-worker/four-P-core oversubscription test,
the four-worker/two-extra-E-core allowance, and the withdrawn SPSC experiment.
Their initialization, monitoring, and implementation stages are tagged separately.
Pre-cleanup reports and sources remain in the local task's sibling n1-cleanup-archive;
full local measurements remain in n1-scalability, n1-6515, n1-all-cores,
n1-acpf-estimate, and n1-spsc-performance. The consolidated record retains preparation
and first-call end-to-end fields.

Jacobian-only GPU offload is deferred. The owner reports about 100 microseconds for
Jacobian fill while factorization dominates. A suitable batched sparse factor/refactor/
solve backend is the next prerequisite for evaluating a complete GPU executor.
