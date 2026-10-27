#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)";cd "$root"
out="${J_AMD_OUT:-$(cat docs/experiments/amd-j-profile/latest.txt)}";cpu="${BENCH_CPU:-0}"
for round in 1 2;do
 ctl="$out/control";ack="$out/ack";rm -f "$ctl" "$ack";mkfifo "$ctl" "$ack"
 PERF_CTL="$ctl" PERF_ACK="$ack" taskset -c "$cpu" perf record -e cpu-clock:u -F 1999 --call-graph fp --user-callchains --delay=-1 --control="fifo:$ctl,$ack" --sample-cpu -o "$out/j-amd-$round.data" -- "$out/yes/profile_j_amd" 9241 2500 "$out/j-amd-$round-driver.json" > "$out/j-amd-$round.stdout" 2> "$out/j-amd-$round.stderr"
 rm -f "$ctl" "$ack"
 perf report -i "$out/j-amd-$round.data" --stdio --no-children --percent-limit .1 > "$out/j-amd-$round-self.txt"
 perf report -i "$out/j-amd-$round.data" --stdio --children --percent-limit .1 > "$out/j-amd-$round-inclusive.txt"
 perf annotate -i "$out/j-amd-$round.data" --stdio -l --percent-type global-period > "$out/j-amd-$round-annotate.txt"
 perf script -i "$out/j-amd-$round.data" -s /usr/lib/perf/scripts/python/stackcollapse.py > "$out/j-amd-$round.folded"
 python3 docs/experiments/amd-j-profile/render-flame.py "$out/j-amd-$round.folded" "$out/j-amd-$round-flame.svg"
done
{ powerprofilesctl get;cat /sys/devices/system/cpu/cpu${cpu}/cpufreq/{energy_performance_preference,scaling_governor}; } > "$out/power-after.txt"
echo "$out"
