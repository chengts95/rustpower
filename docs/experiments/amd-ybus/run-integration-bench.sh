#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$root"
cpu=${BENCH_CPU:-4}
repeats=${REPEATS:-100}
out=${OUTPUT_DIR:-"$root/docs/experiments/amd-ybus/integration-results/cpu${cpu}-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$out"
# Read-only checks: never change user.slice, governor or power profile.
taskset -c "$cpu" true
cc -O3 -fPIC -shared -Icrates/rustpower-amd/experiments/amd-ybus/bench-trace -I/usr/include/suitesparse crates/rustpower-amd/experiments/amd-ybus/bench-trace/trace.c -o crates/rustpower-amd/experiments/amd-ybus/bench-trace/librpamd_bench.so -lklu -lamd -ldl
if [ ! -f Cargo.lock ]; then cp docs/experiments/amd-ybus/integration.Cargo.lock Cargo.lock; fi
cargo build --release --offline --locked --features klu_ybus_amd,python --example bench_integrated_ybus > "$out/build.log" 2>&1
power_state() {
 powerprofilesctl get 2>/dev/null || true
 for p in /sys/devices/system/cpu/cpu${cpu}/cpufreq/{scaling_governor,energy_performance_preference}; do
  if [ -f "$p" ]; then printf '%s: ' "$p"; cat "$p"; fi
 done
}
{
 date -Is;hostname;git rev-parse HEAD;git status --short
 cat /proc/self/status | rg Cpus_allowed_list
 lscpu -e=CPU,CORE,MAXMHZ
 power_state
} > "$out/environment.txt"
for btf in 1 0; do
 for round in 1 2; do
  power_state > "$out/power-btf${btf}-round${round}-before.txt"
  BENCH_CPU="$cpu" LD_PRELOAD="$root/crates/rustpower-amd/experiments/amd-ybus/bench-trace/librpamd_bench.so" taskset -c "$cpu" target/release/examples/bench_integrated_ybus "$repeats" "$out/btf${btf}-round${round}.jsonl" "$btf" > "$out/btf${btf}-round${round}.log" 2>&1
  power_state > "$out/power-btf${btf}-round${round}-after.txt"
  diff -u "$out/power-btf${btf}-round${round}-before.txt" "$out/power-btf${btf}-round${round}-after.txt" > "$out/power-btf${btf}-round${round}-diff.txt" || { echo 'Power mode changed; do not use this round for a fixed-mode comparison.' >&2; exit 1; }
 done
 done
python3 docs/experiments/amd-ybus/summarize-integrated.py "$out" | tee "$out/summary.txt"
echo "Results: $out"
