#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$root"
out="${PROFILE_OUT:-$root/docs/experiments/amd-ybus/baseline-profile/results-$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$out"
printf '%s\n' "$out" > docs/experiments/amd-ybus/baseline-profile/latest.txt
cpu="${BENCH_CPU:-0}"
printf 'Result directory: %s\n' "$out"
export CARGO_PROFILE_RELEASE_DEBUG=1 CARGO_TARGET_DIR="$root/target/amd-profile"
cargo build --offline --release --example profile_amd > "$out/build.log" 2>&1
bin="$CARGO_TARGET_DIR/release/examples/profile_amd"
cp "$bin" "$out/profile_amd"
bin="$out/profile_amd"
# Original binary path is retained for DWARF/symbol lookup; this copy is a durable artifact.
{ uname -a; lscpu; lscpu -C; rustc -Vv; perf --version; git rev-parse HEAD; git status --porcelain | sed -n '1,12p'; env | sort | rg 'RUSTFLAGS|CARGO_PROFILE|CARGO_TARGET' || true; cat /proc/self/status | rg Cpus_allowed; cat /sys/bus/event_source/devices/cpu_{core,atom}/cpus; } > "$out/environment.txt"
perf list --details > "$out/perf-list-details.txt"
cp Cargo.toml "$out/root-Cargo.toml"
cp crates/rustpower-amd/Cargo.toml "$out/amd-Cargo.toml"
mkdir -p "$out/source"
cp -r crates/rustpower-amd/src "$out/source/amd-src"
cp examples/profile_amd.rs "$out/source/"
sha256sum "$bin" Cargo.toml examples/profile_amd.rs crates/rustpower-amd/src/*.rs > "$out/source-binary.sha256"
snapshot(){ powerprofilesctl get; cat "/sys/devices/system/cpu/cpu$cpu/cpufreq/energy_performance_preference" "/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor"; }
snapshot > "$out/power-before.txt"
# Formal wall time: no perf, no phase clocks, samples serialized only after batch.
for round in 1 2; do
 for case in 9241 39 118; do
  for mode in fresh reuse; do
   if ((round==1)); then variants=(full lazy); else variants=(lazy full); fi
   for v in "${variants[@]}"; do
    if test -s "$out/time-$round-$case-$mode-$v.json"; then continue; fi
    taskset -c "$cpu" "$bin" "$case" "$v" "$mode" 300 "$out/time-$round-$case-$mode-$v.json" > "$out/time-$round-$case-$mode-$v.log" 2>&1
   done
  done
 done
done
# Perf gate once per batch. No per-call FIFO or serialization in the enabled region.
measure(){
 local name="$1"; shift
 local ctl="$out/ctl" ack="$out/ack"
 rm -f "$ctl" "$ack"; mkfifo "$ctl" "$ack"
 local rc=0
 PERF_CTL="$ctl" PERF_ACK="$ack" taskset -c "$cpu" "$@" --delay=-1 --control="fifo:$ctl,$ack" -- "$bin" "$case" "$v" "$mode" "$reps" "$out/$name-driver.json" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?
 rm -f "$ctl" "$ack"
 return "$rc"
}
sampling_available=1
case=9241
for round in 1 2; do
 for mode in fresh reuse; do
  if ((round==1)); then variants=(full lazy); else variants=(lazy full); fi
  for v in "${variants[@]}"; do
   reps=1800; name="sample-$round-$mode-$v"
   if [[ "${PROFILE_SKIP_SAMPLING:-0}" != 1 && "$sampling_available" == 1 ]]; then
   if measure "$name" perf record -e cpu-clock:u -F 999 --call-graph "${PROFILE_CALLGRAPH:-dwarf,16384}" --sample-cpu -o "$out/$name.data"; then
   perf report -i "$out/$name.data" --stdio --no-children --percent-limit 0.1 > "$out/$name-self.txt" 2> "$out/$name-report.stderr"
   perf report -i "$out/$name.data" --stdio --children --percent-limit 0.1 > "$out/$name-inclusive.txt" 2>> "$out/$name-report.stderr"
   else
    sampling_available=0
    printf 'Sampling unavailable; see %s\n' "$out/$name.stderr" >&2
   fi
   fi
   reps=400
   measure "stat-$round-$mode-$v-basic" perf stat -x ';' -o "$out/stat-$round-$mode-$v-basic.csv" -e '{cpu_core/cycles/u,cpu_core/instructions/u,cpu_core/branches/u,cpu_core/branch-misses/u},context-switches,cpu-migrations,page-faults'
   measure "stat-$round-$mode-$v-load" perf stat -x ';' -o "$out/stat-$round-$mode-$v-load.csv" -e '{cpu_core/mem_load_retired.l1_hit/u,cpu_core/mem_load_retired.l1_miss/u,cpu_core/mem_load_retired.l2_hit/u,cpu_core/mem_load_retired.l2_miss/u}'
   measure "stat-$round-$mode-$v-last" perf stat -x ';' -o "$out/stat-$round-$mode-$v-last.csv" -e '{cpu_core/mem_load_retired.l3_hit/u,cpu_core/mem_load_retired.l3_miss/u,cpu_core/dtlb_load_misses.stlb_hit/u,cpu_core/dtlb_load_misses.walk_completed/u}'
  done
 done
done
snapshot > "$out/power-after.txt"
diff -u "$out/power-before.txt" "$out/power-after.txt" > "$out/power.diff" || true
if [[ "${PROFILE_FLAT_SAMPLING:-1}" == 1 ]]; then
 PROFILE_OUT="$out" BENCH_CPU="$cpu" bash docs/experiments/amd-ybus/baseline-profile/sample-flat.sh
fi
# Coarse clocks are compiled separately; never mix these timings with the formal binary.
cargo build --offline --release --example profile_amd --features experimental_amd_profile > "$out/build-phases.log" 2>&1
phasebin="$out/profile_amd_phases"
cp "$CARGO_TARGET_DIR/release/examples/profile_amd" "$phasebin"
sha256sum "$phasebin" > "$out/phases-binary.sha256"
for case in 9241 39 118; do
 for mode in fresh reuse; do
  taskset -c "$cpu" "$phasebin" "$case" full "$mode" 300 "$out/phases-$case-$mode.json" > "$out/phases-$case-$mode.log" 2>&1
 done
done
echo "$out"

python3 docs/experiments/amd-ybus/baseline-profile/summarize.py "$out"
python3 docs/experiments/amd-ybus/baseline-profile/analyze-samples.py "$out"
