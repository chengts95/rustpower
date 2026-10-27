# AMD v3/AVX and LTO experiment, 2026-10-27

Four verified `cdylib` builds of the same source were compared. `opt-level=3`,
`codegen-units=16`, Python features `klu_ybus_amd_rust,python`, checked Rust AMD,
and the same system dynamic KLU were held fixed. No Cargo default profile,
global compiler configuration or installed virtual-environment package changed.

| Variant | rustpower-amd target | Cross-crate Rust LTO |
|---|---|---|
| baseline | default | false |
| v3 | x86-64-v3 | false |
| thin | x86-64-v3 | thin |
| fat | x86-64-v3 | fat |

`x86-64-v3` already includes AVX, AVX2 and FMA. The local compiler's `--print cfg`
and CPU capabilities were checked. A workspace compiler wrapper appended the
CPU flag only to `rustpower_amd`. LTO operated on the final Rust extension and
its Rust dependencies. System KLU and the C AMD reference retained their existing
compiler settings. LLVM vector instructions were found in the v3 AMD kernel.

## Effective LTO verification

The manifest currently emits both `rlib` and `cdylib`. Inspection of the actual
rustc command showed that Cargo omitted the final LTO flag for this combination,
even when a profile requested LTO. Those intermediate builds were excluded.
Every measured build instead used `cargo rustc --lib --crate-type cdylib`; the
actual commands for the thin/fat products contain `-C lto=thin` and `-C lto=fat`.
This behavior is tracked in [Cargo issue 14612](https://github.com/rust-lang/cargo/issues/14612).
The ordinary false setting retains rustc's possible local optimizations and is
not a cross-crate LTO build.

## Measurement

CPU0, 39/1354/9241 cases, one frozen input fixture per case. Two serial rounds,
configuration order baseline/v3/thin/fat then reversed. C and Rust backends were
alternated within each sample pair. Each case/backend/mode/round had 10 warmups
and 100 recorded samples: 14,400 recorded samples in total. Governor/EPP snapshots
were unchanged for every run; no power settings were modified.

The numerical clocks match the boundaries of `scripts/bench_py.py`:
- cold: `setup_context` including Ybus permutation, complete `solve` and `get_scalc`;
- cached core: `set_v_init`, `solve` and `get_scalc`;
- cached grid: `solve` and `post_process`.

Constructors, backend selection, fixture loading and validation were outside the
clocks. Arrays were fully loaded before timing. Tolerance was 1e-8; case iteration
counts were 4/5/6. Ordering preparation is the internal `prepare_ybus` interval:
structure validation, metadata and quotient graph, AMD, KLU Symbolic and exact
key copy. It excludes numerical factorization and J-cache construction.

### 9241 Rust backend, milliseconds

Each cell is the median of the two per-round medians (100 samples per round).
Preparation is a component of cold time, not an additional cost.

| Build | Ordering preparation | Complete cold | Cached core | Cached grid + postprocess |
|---|---:|---:|---:|---:|
| baseline | 3.1059 | 19.9634 | 11.9341 | 12.3746 |
| v3 | 3.1016 | 19.8783 | 11.9697 | 12.4042 |
| thin | 3.0848 | 20.0022 | 11.8735 | 12.5074 |
| fat | 3.0524 | 19.9731 | 11.7988 | 12.5469 |

V3 changed complete cold time by -0.43%, Thin by +0.19%, Fat by +0.05%.
Fat preparation was about 0.054 ms (-1.72%) lower. Fat cached core was -1.13%,
while its cached grid was +1.39%. Small-case results also varied: 1354 Fat cached
core/grid were about +5%; 39 Fat grid was about -5%. This dataset shows no
consistent end-to-end performance gain from adopting these flags as defaults.

Fat reduced extension size from 24,997,304 to 18,144,600 bytes (-27.4%). Thin was
24,397,720 bytes (-2.4%). Binary size is a separate result from runtime performance.

## Correctness and evidence

Each configuration passed the two Native/ECS backend-selection and numerical
agreement tests. All timed solves converged and reported the requested optimized backend and
one preparation on reused solvers. Native solves matched the expected iteration
counts in every sample. Maximum measured voltage difference was below 7.2e-13 p.u.

Raw per-sample JSONL, per-round summaries, flags, extension SHA256, original
inputs, compiler wrappers, driver scripts, assembly and test/build logs are in
`/home/cts/Documents/Codex/2026-10-26/task/amd-codegen`. Each configuration has
`build-verified.json` with the actual final rustc invocation and extension hash.
`run-all.py` reruns the serial paired measurements; `summarize.py` regenerates
aggregated results. Existing source/default settings and virtual environment
installation were retained. No commits or remote pushes were made.
