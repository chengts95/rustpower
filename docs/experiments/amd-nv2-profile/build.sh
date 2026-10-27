#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)";cd "$root"
out="${NV2_OUT:-$root/docs/experiments/amd-nv2-profile/results-$(date -u +%Y%m%d-%H%M%S)}";mkdir -p "$out"
printf '%s\n' "$out" > docs/experiments/amd-nv2-profile/latest.txt
mkdir -p "$out/amd-original-source";cp -r crates/rustpower-amd/src "$out/amd-original-source/";cp crates/rustpower-amd/POWERFLOW_API.md "$out/amd-original-source/"
cp -r "$out/amd-original-source" "$out/amd-timed-source"
# Only the diagnostic SOURCE COPY receives two coarse clocks; production source is untouched.
python3 - "$out" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1])/'amd-timed-source/src'
s=(p/'amd_2.rs').read_text().replace('    let un = n.to_usize().unwrap();','    let kernel_clock=std::time::Instant::now();\n    let un = n.to_usize().unwrap();',1)
last=s.rfind('\n    (\n');s=s[:last]+s[last:].replace('\n    (\n','\n    let result = (\n',1)
s=s.rsplit('\n    )\n}',1)[0]+'\n    );\n    crate::kernel_profile::record(kernel_clock.elapsed().as_secs_f64());\n    result\n}\n';(p/'amd_2.rs').write_text(s)
with (p/'lib.rs').open('a') as f:f.write('\npub mod kernel_profile;\n')
(p/'kernel_profile.rs').write_text('use std::cell::Cell;\nthread_local! {static LAST:Cell<Option<f64>>=const{Cell::new(None)};}\npub(crate) fn record(t:f64){LAST.with(|s|s.set(Some(t)));}\npub fn take()->Option<f64>{LAST.with(|s|s.take())}\n')
PY
deps="$root/target/amd-profile/release/deps";num_traits=("$deps"/libnum_traits-*.rlib);serde_json=("$deps"/libserde_json-*.rlib)
for kind in ordinary frames timed;do
 fp=no;source=amd-original-source;args=()
 if [[ "$kind" == frames ]];then fp=yes;fi
 if [[ "$kind" == timed ]];then source=amd-timed-source;args=(--cfg kernel_timing);fi
 mkdir -p "$out/$kind"
 rustc --crate-name rustpower_amd --edition=2021 --crate-type rlib -C opt-level=3 -C debuginfo=2 -C overflow-checks=no -C force-frame-pointers="$fp" -L dependency="$deps" --extern "num_traits=${num_traits[0]}" "$out/$source/src/lib.rs" -o "$out/$kind/librustpower_amd.rlib"
 rustc --crate-name profile_nv2_amd --edition=2024 -C opt-level=3 -C debuginfo=2 -C overflow-checks=no -C force-frame-pointers="$fp" "${args[@]}" -L dependency="$deps" --extern "rustpower=$deps/librustpower.rlib" --extern "rustpower_amd=$out/$kind/librustpower_amd.rlib" --extern "serde_json=${serde_json[0]}" examples/profile_nv2_amd.rs -o "$out/$kind/profile_nv2_amd"
done
sha256sum "$out"/{ordinary,frames,timed}/profile_nv2_amd crates/rustpower-amd/src/*.rs examples/profile_nv2_amd.rs > "$out/source-binary.sha256"
cp examples/profile_nv2_amd.rs "$out/";cp "${BASH_SOURCE[0]}" "$out/"
{ uname -a;lscpu;rustc -Vv;perf --version;git rev-parse HEAD;cat /sys/bus/event_source/devices/cpu_{core,atom}/cpus;powerprofilesctl get;cat /sys/devices/system/cpu/cpu0/cpufreq/{energy_performance_preference,scaling_governor};} > "$out/environment.txt"
echo "$out"
