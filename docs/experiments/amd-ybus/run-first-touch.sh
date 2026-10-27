#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"
cpu="${BENCH_CPU:-0}"
reps="${REPEATS:-100}"
out="$root/docs/experiments/amd-ybus/first-touch-results/cpu${cpu}-$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$out"
taskset -c "$cpu" true
cargo build --offline --release --example bench_first_touch > "$out/build.log" 2>&1
cp Cargo.lock "$out/Cargo.lock"
sha256sum target/release/examples/bench_first_touch crates/rustpower-amd/src/exact.rs crates/rustpower-amd/src/amd_2.rs > "$out/source-binary.sha256"
snapshot(){
 powerprofilesctl get
 cat "/sys/devices/system/cpu/cpu${cpu}/cpufreq/energy_performance_preference"
 cat "/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_governor"
}
for round in 1 2; do
 snapshot > "$out/power${round}-before.txt"
 BENCH_CPU="$cpu" taskset -c "$cpu" target/release/examples/bench_first_touch "$reps" "$out/round${round}.jsonl" > "$out/round${round}.log" 2>&1
 snapshot > "$out/power${round}-after.txt"
 diff -u "$out/power${round}-before.txt" "$out/power${round}-after.txt" > "$out/power${round}.diff"
done
python3 docs/experiments/amd-ybus/summarize-first-touch.py "$out"
echo "$out"
