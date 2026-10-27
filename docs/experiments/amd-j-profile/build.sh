#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)";cd "$root"
out="${J_AMD_OUT:-$root/docs/experiments/amd-j-profile/results-$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$out";printf '%s\n' "$out" > docs/experiments/amd-j-profile/latest.txt
# Existing dependency artifacts are shared; only AMD and this driver get frame pointers.
deps="$root/target/amd-profile/release/deps"
num_traits=("$deps"/libnum_traits-*.rlib);serde_json=("$deps"/libserde_json-*.rlib)
test "${#num_traits[@]}" == 1;test "${#serde_json[@]}" == 1;test -f "$deps/librustpower.rlib"
for fp in no yes;do
 mkdir -p "$out/$fp"
 rustc --crate-name rustpower_amd --edition=2021 --crate-type rlib -C opt-level=3 -C debuginfo=2 -C overflow-checks=no -C force-frame-pointers="$fp" -L dependency="$deps" --extern "num_traits=${num_traits[0]}" crates/rustpower-amd/src/lib.rs -o "$out/$fp/librustpower_amd.rlib"
 rustc --crate-name profile_j_amd --edition=2024 -C opt-level=3 -C debuginfo=2 -C overflow-checks=no -C force-frame-pointers="$fp" -L dependency="$deps" --extern "rustpower=$deps/librustpower.rlib" --extern "rustpower_amd=$out/$fp/librustpower_amd.rlib" --extern "serde_json=${serde_json[0]}" examples/profile_j_amd.rs -o "$out/$fp/profile_j_amd"
done
sha256sum "$out"/{no,yes}/profile_j_amd crates/rustpower-amd/src/*.rs examples/profile_j_amd.rs > "$out/source-binary.sha256"
cp -r crates/rustpower-amd/src "$out/amd-source";cp examples/profile_j_amd.rs "$out/"
cp "${BASH_SOURCE[0]}" "$out/build-command.sh"
{ uname -a; lscpu;rustc -Vv;perf --version;git rev-parse HEAD;cat /sys/bus/event_source/devices/cpu_{core,atom}/cpus;powerprofilesctl get;cat /sys/devices/system/cpu/cpu0/cpufreq/{energy_performance_preference,scaling_governor}; } > "$out/environment.txt"
echo "$out"
