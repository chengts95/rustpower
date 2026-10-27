#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)";cd "$root"
out="${J_AMD_OUT:-$(cat docs/experiments/amd-j-profile/latest.txt)}";cpu="${BENCH_CPU:-0}"
taskset -c "$cpu" true
for round in 1 2; do
 for case in 9241 39 118; do
  if ((round==1)); then variants=(no yes);else variants=(yes no);fi
  for fp in "${variants[@]}";do
   taskset -c "$cpu" "$out/$fp/profile_j_amd" "$case" 400 "$out/time-$round-$case-fp-$fp.json"
  done
 done
done
J_AMD_OUT="$out" BENCH_CPU="$cpu" bash docs/experiments/amd-j-profile/collect.sh
