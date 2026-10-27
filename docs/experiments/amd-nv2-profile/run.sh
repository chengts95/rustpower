#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)";cd "$root"
out="${NV2_OUT:-$(cat docs/experiments/amd-nv2-profile/latest.txt)}";cpu="${BENCH_CPU:-0}"
snapshot(){ powerprofilesctl get;cat /sys/devices/system/cpu/cpu${cpu}/cpufreq/{energy_performance_preference,scaling_governor};}
snapshot > "$out/power-before.txt"
for kind in ordinary timed;do
 for round in 1 2;do
  if ((round==1));then variants=(scalar nv2);else variants=(nv2 scalar);fi
  for case in 9241 39 118;do
   for variant in "${variants[@]}";do
    taskset -c "$cpu" "$out/$kind/profile_nv2_amd" "$case" 400 "$out/time-$kind-$round-$case-$variant.json" "$variant"
   done
  done
 done
done
for round in 1 2;do
 if ((round==1));then variants=(scalar nv2);else variants=(nv2 scalar);fi
 for variant in "${variants[@]}";do
  name="$variant-$round";reps=2500;if [[ "$variant" == nv2 ]];then reps=6000;fi
  ctl="$out/control";ack="$out/ack";rm -f "$ctl" "$ack";mkfifo "$ctl" "$ack"
  PERF_CTL="$ctl" PERF_ACK="$ack" taskset -c "$cpu" perf record -e cpu-clock:u -F 1999 --call-graph fp --user-callchains --delay=-1 --control="fifo:$ctl,$ack" --sample-cpu -o "$out/$name.data" -- "$out/frames/profile_nv2_amd" 9241 "$reps" "$out/$name-driver.json" "$variant" > "$out/$name.stdout" 2> "$out/$name.stderr"
  rm -f "$ctl" "$ack"
  perf report -i "$out/$name.data" --stdio --no-inline --no-children --call-graph none --percent-limit .1 > "$out/$name-self.txt"
  perf report -i "$out/$name.data" --stdio --no-inline --children --call-graph none --percent-limit .1 > "$out/$name-inclusive.txt"
  perf script -i "$out/$name.data" --no-inline -s /usr/lib/perf/scripts/python/stackcollapse.py > "$out/$name.folded"
  python3 docs/experiments/amd-j-profile/render-flame.py "$out/$name.folded" "$out/$name-flame.svg" "AMD: $variant (initial PQ Nv=2 only for nv2)"
  if ((round==1));then perf annotate -i "$out/$name.data" --stdio -l --percent-type global-period > "$out/$name-annotate.txt";fi
 done
done
snapshot > "$out/power-after.txt"
diff -u "$out/power-before.txt" "$out/power-after.txt" > "$out/power.diff" || true
echo "$out"
