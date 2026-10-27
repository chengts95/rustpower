#!/usr/bin/env bash
set -euo pipefail
source_repo=${1:-/home/cts/workspace/rustpower}
files=$(cd -- "$(dirname -- "$0")" && pwd)
cfiles=$(cd -- "$files/../../../.." && pwd)/crates/rustpower-amd/experiments/amd-ybus/restored-early-amd
run_root=${RUN_ROOT:-/tmp/rustpower-early-amd-repeat-1ad70a7}
repeats=${REPEATS:-200}
bench_cpu=${BENCH_CPU:-4}
mkdir -p "$run_root/amd-shim" "$run_root/early-amd-shim"
if [ ! -f "$run_root/Cargo.toml" ]; then
 git -C "$source_repo" archive 1ad70a7a5834dd6d93c5c967115beb63e853f91b | tar -x -C "$run_root"
fi
cp "$files/Cargo.lock" "$run_root/Cargo.lock"
cp "$files/stage1/bench_amd_cost.rs" "$run_root/examples/"
cp "$cfiles/stage1/amd-shim/"*.c "$cfiles/stage1/amd-shim/"*.h "$run_root/amd-shim/"
cp "$files/stage2/bench_early_amd.rs" "$files/stage2/check_amd_memory.rs" "$run_root/examples/"
cp "$cfiles/stage2/early-amd-shim/"*.c "$cfiles/stage2/early-amd-shim/"*.h "$run_root/early-amd-shim/"
cp "$files/summarize.py" "$run_root/early-amd-summarize.py"
cd "$run_root"
cc -O3 -fPIC -shared -DDLONG -DNDEBUG -Iamd-shim -I/usr/include/suitesparse amd-shim/profile_analyze.c amd-shim/trace.c -o amd-shim/librpamd.so -lklu -lamd -lbtf -lcolamd -lsuitesparseconfig -ldl
cargo rustc --release --offline --locked --features klu_dyn --example bench_amd_cost -- -L "native=$run_root/amd-shim" -C "link-arg=-Wl,-rpath,$run_root/amd-shim"
for round in 1 2; do taskset -c "$bench_cpu" target/release/examples/bench_amd_cost "$repeats" "amd-cost-round${round}.jsonl" 2> "amd-cost-round${round}.log"; done
cc -O3 -fPIC -shared -Iearly-amd-shim -I/usr/include/suitesparse early-amd-shim/trace.c early-amd-shim/fast.c -o early-amd-shim/librpamd_early.so -lklu -lamd -lsuitesparseconfig -ldl
cargo rustc --release --offline --locked --features klu_dyn --example bench_early_amd -- -L "native=$run_root/early-amd-shim" -C "link-arg=-Wl,-rpath,$run_root/early-amd-shim"
for round in 1 2; do taskset -c "$bench_cpu" target/release/examples/bench_early_amd "$repeats" "early-amd-round${round}.jsonl" > "early-amd-round${round}-stdout.log" 2> "early-amd-round${round}.log"; done
python3 early-amd-summarize.py early-amd-round1.jsonl early-amd-round2.jsonl | tee early-amd-summary.txt
cargo rustc --release --offline --locked --features klu_dyn --example check_amd_memory -- -L "native=$run_root/early-amd-shim" -C "link-arg=-Wl,-rpath,$run_root/early-amd-shim"
taskset -c "$bench_cpu" target/release/examples/check_amd_memory > amd-memory.jsonl
