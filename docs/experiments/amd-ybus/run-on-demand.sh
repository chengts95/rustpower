#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"
cpu="${BENCH_CPU:-0}"; reps="${REPEATS:-100}"
out="$root/docs/experiments/amd-ybus/on-demand-results/cpu${cpu}-$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$out"
taskset -c "$cpu" true
cargo build --offline --release --example bench_on_demand > "$out/build.log" 2>&1
cp Cargo.lock "$out/Cargo.lock"
sha256sum target/release/examples/bench_on_demand crates/rustpower-amd/src/on_demand.rs crates/rustpower-amd/src/lazy_core.rs > "$out/source-binary.sha256"
snapshot(){ powerprofilesctl get; cat "/sys/devices/system/cpu/cpu${cpu}/cpufreq/energy_performance_preference"; cat "/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_governor"; }
for round in 1 2; do
 snapshot > "$out/power${round}-before.txt"
 BENCH_CPU="$cpu" taskset -c "$cpu" target/release/examples/bench_on_demand "$reps" "$out/round${round}.jsonl" > "$out/round${round}.log" 2>&1
 snapshot > "$out/power${round}-after.txt"
 diff -u "$out/power${round}-before.txt" "$out/power${round}-after.txt" > "$out/power${round}.diff"
done
python3 docs/experiments/amd-ybus/summarize-on-demand.py "$out"
echo "$out"
