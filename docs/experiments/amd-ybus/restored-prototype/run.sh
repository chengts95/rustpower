#!/usr/bin/env bash
set -euo pipefail
repo=${1:-/home/cts/workspace/rustpower}
files=$(cd -- "$(dirname -- "$0")" && pwd)
cfiles=$(cd -- "$files/../../../.." && pwd)/crates/rustpower-amd/experiments/amd-ybus/restored-prototype
root=${RUN_ROOT:-/tmp/rustpower-super-amd-repeat-1ad70a7}
repeats=${REPEATS:-100}
cpu=${BENCH_CPU:-4}
mkdir -p "$root/super-amd-shim"
if [ ! -f "$root/Cargo.toml" ]; then
 git -C "$repo" archive 1ad70a7a5834dd6d93c5c967115beb63e853f91b | tar -x -C "$root"
fi
cp "$files/Cargo.lock" "$root/Cargo.lock"
cp "$files/bench_super_amd.rs" "$files/check_super_memory.rs" "$root/examples/"
cp "$cfiles/super-amd-shim/"*.c "$cfiles/super-amd-shim/"*.h "$root/super-amd-shim/"
cp "$files/summarize.py" "$root/super-amd-summarize.py"
cd "$root"
cc -O3 -fPIC -shared -DDLONG -Isuper-amd-shim -I/usr/include/suitesparse super-amd-shim/trace.c super-amd-shim/fast.c super-amd-shim/ybus.c super-amd-shim/amd_super2.c -o super-amd-shim/librpsuperamd.so -lklu -lamd -lsuitesparseconfig -ldl -lm
cargo rustc --release --offline --locked --features klu_dyn --example bench_super_amd -- -L "native=$root/super-amd-shim" -C "link-arg=-Wl,-rpath,$root/super-amd-shim"
for round in 1 2; do
 taskset -c "$cpu" target/release/examples/bench_super_amd "$repeats" "super-amd-round${round}.jsonl" > "super-amd-round${round}-stdout.log" 2> "super-amd-round${round}.log"
done
python3 super-amd-summarize.py super-amd-round1.jsonl super-amd-round2.jsonl | tee super-amd-summary.txt
cargo rustc --release --offline --locked --features klu_dyn --example check_super_memory -- -L "native=$root/super-amd-shim" -C "link-arg=-Wl,-rpath,$root/super-amd-shim"
taskset -c "$cpu" target/release/examples/check_super_memory > super-memory.jsonl
