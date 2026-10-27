#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"
cpu="${BENCH_CPU:-0}"
reps="${REPEATS:-100}"
out="$root/docs/experiments/amd-ybus/rust-amd-results/cpu${cpu}-$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$out"
taskset -c "$cpu" true
if [[ ! -f Cargo.lock ]]; then
    cp docs/experiments/amd-ybus/integration.Cargo.lock Cargo.lock
fi
cargo build --offline --locked --release --features klu_ybus_amd --example bench_rust_amd > "$out/build.log" 2>&1
cp Cargo.lock "$out/Cargo.lock"
git rev-parse HEAD > "$out/base.txt"
git diff > "$out/tracked.patch"
lscpu -e=CPU,CORE,MAXMHZ > "$out/cpu-topology.txt"
cat /proc/self/status > "$out/process-status.txt"
sha256sum target/release/examples/bench_rust_amd > "$out/binary.sha256"
snapshot() {
    powerprofilesctl get
    cat "/sys/devices/system/cpu/cpu${cpu}/cpufreq/energy_performance_preference"
    cat "/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_governor"
}
for round in 1 2; do
    snapshot > "$out/power-round${round}-before.txt"
    BENCH_CPU="$cpu" taskset -c "$cpu" target/release/examples/bench_rust_amd "$reps" "$out/round${round}.jsonl" > "$out/round${round}.log" 2>&1
    snapshot > "$out/power-round${round}-after.txt"
    diff -u "$out/power-round${round}-before.txt" "$out/power-round${round}-after.txt" > "$out/power-round${round}.diff"
done
python3 docs/experiments/amd-ybus/summarize-rust-amd.py "$out"
echo "$out"
