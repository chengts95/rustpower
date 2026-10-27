#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$root"
out="${PROFILE_OUT:-$(cat docs/experiments/amd-ybus/baseline-profile/latest.txt)}"
bin="$out/profile_amd"; cpu="${BENCH_CPU:-0}"
for round in 1 2; do
 for mode in fresh reuse; do
  if ((round==1)); then variants=(full lazy); else variants=(lazy full); fi
  for v in "${variants[@]}"; do
   name="flat-$round-$mode-$v"
   ctl="$out/flatctl";ack="$out/flatack"
   rm -f "$ctl" "$ack";mkfifo "$ctl" "$ack"
   PERF_CTL="$ctl" PERF_ACK="$ack" taskset -c "$cpu" perf record -e cpu-clock:u -F 1999 --delay=-1 --control="fifo:$ctl,$ack" --sample-cpu -o "$out/$name.data" -- "$bin" 9241 "$v" "$mode" 2500 "$out/$name-driver.json" > "$out/$name.stdout" 2> "$out/$name.stderr"
   rm -f "$ctl" "$ack"
   perf report -i "$out/$name.data" --stdio --no-children --percent-limit 0.1 > "$out/$name-self.txt" 2> "$out/$name-report.stderr"
   perf annotate -i "$out/$name.data" --stdio -l --percent-type global-period > "$out/$name-annotate.txt" 2> "$out/$name-annotate.stderr"
   perf script -i "$out/$name.data" --no-inline -F period,ip,sym,symoff,dso > "$out/$name-leaf.txt" 2> "$out/$name-script.stderr"
  done
 done
done
