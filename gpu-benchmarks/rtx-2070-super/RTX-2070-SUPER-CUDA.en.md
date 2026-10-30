# RTX 2070 SUPER CUDA Performance Report

Test date: 2026-10-29  
Status: Complete testing; all computed results passed validation.

[中文版本](RTX-2070-SUPER-CUDA.md) · [RTX 5070 English report](../rtx-5070/RTX-5070-CUDA.en.md)

## Executive summary

This RTX 2070 SUPER achieved median results of **9.20 TFLOP/s in FP32 FMA**, **8.66 TFLOP/s in FP32 GEMM**, and **37.84 TFLOP/s in FP16 Tensor GEMM** in the pure-compute microbenchmarks. Device-to-device copy bandwidth reached a best result of **391.73 GB/s**. Median FP16 Tensor GEMM throughput was approximately **4.37×** median FP32 GEMM throughput.

For same-path RTX 5070 comparisons, including a **2.65× improvement in the PEGASE 9241 complete Newton benchmark**, see the [RTX 5070 report](../rtx-5070/RTX-5070-CUDA.en.md).

## Test environment

| Item | Record |
|---|---|
| GPU | NVIDIA GeForce RTX 2070 SUPER, MSI model, 8 GB |
| Compute Capability | 7.5 |
| SM count | 40 |
| Memory bus width | 256 bit |
| CPU | Intel Core i7-10700K, 8 cores / 16 threads |
| Motherboard | MSI Z490-A PRO (MS-7C75) |
| Memory | 31.92 GiB recognized capacity |
| Operating system | Windows 11 Home, 10.0.26300 |
| NVIDIA driver | 610.62 (WMI: 32.0.16.1062) |
| CUDA Driver API | 13.3 |
| CUDA Runtime | 12.9 |
| Test framework | CuPy 14.2.0, Python 3.12.15 |
| Windows power mode | Ultimate Performance |

## Results

Median values represent typical performance across repeated measurements; best values represent the fastest round. Both are retained so that a single peak does not hide run-to-run variation.

| Test | Median | Best | Repetitions |
|---|---:|---:|---:|
| Device-to-device copy bandwidth | 349.03 GB/s | 391.73 GB/s | 10 |
| FP32 FMA | 9.200 TFLOP/s | 9.860 TFLOP/s | 7 |
| FP64 FMA | 0.2440 TFLOP/s | 0.2476 TFLOP/s | 5 |
| FP32 GEMM, 4096³ | 8.663 TFLOP/s | 9.416 TFLOP/s | 10 |
| FP16 Tensor GEMM, 4096³ | 37.840 TFLOP/s | 38.893 TFLOP/s | 10 |
| FP64 GEMM, 4096³ | 0.2859 TFLOP/s | 0.2874 TFLOP/s | 5 |

## Methodology

- All official results use CUDA Events to measure GPU execution time. Array initialization, first-time compilation, and warm-up are excluded.
- Copy bandwidth uses two 1 GiB buffers for device-to-device copies. Traffic counts one read plus one write, with 3 warm-up rounds and 10 measured rounds.
- FMA uses a custom CUDA kernel with 40,960 threads and 8 independent dependency chains per thread. FP32 executes 262,144 iterations per round; FP64 executes 32,768.
- GEMM uses cuBLAS with 4096 × 4096 × 4096 matrices. FP32 explicitly uses the default non-TF32 math mode, and FP16 enables Tensor Op math mode.
- GEMM uses all-one input matrices and spot-checks the output value 4096. The custom FMA kernel checks that outputs are nonzero and finite.
- The raw JSON retains every round's time and throughput rather than only summary values.

## Interpretation

- These are **pure CUDA microbenchmarks**, intended for comparison with another GPU under the same scripts, matrix sizes, and dependency versions.
- GEMM results are not equivalent to real deep-learning training or inference performance. Real models also depend on memory capacity, memory access, operator fusion, batch size, and framework overhead.
- Device copies showed noticeable transient variation during this session, so the median is lower than the best result. Both values should be retained.
- FP16 results describe only this HGEMM shape and cuBLAS configuration; they must not be treated as a fixed ratio for every FP16 workload.

## Unmeasured items and anomalies

- `nvidia-smi` returned `Failed to initialize NVML: Unknown Error`. This report therefore contains **no reliable temperature, power, or instantaneous clock data**, and no values have been inferred to fill those gaps.
- Several CUDA Toolkit directories on the system contained only residual files. The tests used isolated CuPy and CUDA runtime components inside the project directory; the system CUDA configuration was not repaired or modified.
- Temperature, power, and real AI-model performance require separate testing.

## Appendix: PEGASE 9241 batched cuDSS FP64 prototype

The tests used the Jacobian from Newton iteration 1 of the real PEGASE 9241 case. Matrix order was 17,036, with 129,412 nonzeros per system. The C++ host launched batched kernels through the CUDA Driver API. cuDSS 1.0 used uniform batching and a fully active `CUDSS_DATA_UBATCH_MASK`. Every system had different matrix values and right-hand sides. Results were checked element by element against a preconstructed known solution; no LU was shared across scenarios.

### Batch sweep: refactorization + solve only

Each batch executed three rounds. The table reports median factor/refactor and solve times. The single-system sustained baseline was 2.0716 ms, or approximately 482.72 systems/s.

| Batch | Median factor/refactor | Median solve | Combined throughput | cuDSS peak estimate |
|---:|---:|---:|---:|---:|
| 8 | 8.924 ms | 1.928 ms | 737.15 systems/s | 0.057 GiB |
| 32 | 34.769 ms | 5.366 ms | 797.32 systems/s | 0.217 GiB |
| 128 | 115.780 ms | 19.639 ms | 945.22 systems/s | 0.860 GiB |
| 256 | 229.145 ms | 36.046 ms | **965.34 systems/s** | 1.717 GiB |
| 512 | 485.399 ms | 71.766 ms | 918.94 systems/s | 3.430 GiB |
| 768 | 687.492 ms | 109.906 ms | 963.13 systems/s | 5.143 GiB |
| 1024 | 32,444.033 ms | 164.344 ms | 31.40 systems/s | 6.857 GiB |

Within the normal capacity range, batching provided approximately **1.5–2.0×** throughput, with B=256 among the efficient points. B=1024 showed a nonlinear collapse. Its cuDSS estimate excludes the external batched Jacobian, RHS, and solution vectors, so this capacity boundary must not be interpreted as general batch-LU behavior. B=1024 is not a reasonable default for this matrix on this GPU.

### Real batched CSR assembly and AC evaluation

The next step replaced “scale an existing Jacobian” with a GPU port of `JacobianOperator::fill::<true, false>`. It parallelizes over Ybus CSC entries and writes the four CSR quadrants directly using `mirror`, `pq_ends`, `active_ends`, and `diag_ptrs`. A mirrored traversal computes `S_calc` and mismatch for the batch. Every scenario is reassembled from its actual Ybus and a different voltage state; precomputed Jacobian values are no longer read.

Stage results at B=256:

| Stage | Whole batch | Per system |
|---|---:|---:|
| Voltage preparation | 0.458 ms | 1.79 μs |
| Batched power and mismatch | 17.811 ms | 69.57 μs |
| **Batched CSR Jacobian assembly** | **2.286 ms** | **8.93 μs** |
| Median factor/refactor | 248.484 ms | 970.64 μs |
| Median solve | 37.158 ms | 145.15 μs |

Power/mismatch + CSR assembly + refactor + solve took 305.740 ms, or approximately **837.31 systems/s**. This excludes voltage updates, convergence checks, and the complete Newton loop, so it is not complete ACPF throughput. Assembly itself accounted for only about 0.75% of this stage combination; LU remained the dominant cost.

Reference-state validation: maximum absolute Jacobian error `5.456968211e-12`, maximum absolute mismatch error `4.485301019e-12`, and maximum relative known-solution error across 256 systems `3.332149371e-12`. No non-finite solution was produced.

Raw single-step results are stored in `results/cudss-pegase9241-batch*-fp64-results.json` and `results/acpf-assembly-batch*-fp64-results.json`.

### Complete batched Newton power flow

The NVRTC single-step path was retained, and a `newton` mode was added to the same program. Every update recomputes power/mismatch, CSR Jacobian values, cuDSS refactorization, solve, and voltage update. After each round, the GPU computes each scenario's mismatch infinity norm and updates `CUDSS_DATA_UBATCH_MASK` with a `1e-8` threshold. Converged scenarios leave subsequent kernels and linear solves.

The pandapower reference power flow converged after **6 updates** from the same flat start, with a reference mismatch infinity norm of `8.487752065e-12`. GPU tests at B=2 and B=256 both completed **6 updates per scenario** and converged.

Complete six-update results at B=256:

| Stage | Six-update total |
|---|---:|
| Power and mismatch (7 evaluations, including the final check) | 102.697 ms |
| CSR Jacobian assembly (6 assemblies) | 11.746 ms |
| Convergence checks (7 checks) | 1.061 ms |
| Voltage updates (6 updates) | 9.625 ms |
| Factor/refactor (6 calls) | 1,397.868 ms |
| Solve (6 calls) | 218.104 ms |
| **Complete Newton wall time** | **1,742.531 ms** |

This corresponds to **6.807 ms per complete power flow**, **146.91 complete power flows/s**, or 881.48 Newton system updates/s. Wall time includes per-round mask and norm downloads but excludes reusable symbolic analysis and the preceding single-step checks. Factor/refactor accounted for approximately 80.22% of Newton wall time; CSR assembly accounted for approximately 0.67%.

The maximum final mismatch across all scenarios was `1.740252387e-11`. The maximum real/imaginary voltage-component error for scenario 0 against the converged reference was `9.587886041e-13`, with no non-finite voltage. The selected perturbations caused every scenario to converge together after the sixth update, so mixed 5/6/7-iteration exits were not measured. Managed-memory double buffering and the N-1 executor had not yet been integrated.

Reproduction:

```bash
target/acpf-cuda/acpf_batch_demo.exe target/acpf-cuda/pegase9241-jacobian.bin 256 1 newton
```

### Column-per-thread assembly without binary search

A column-per-thread variant removed binary search from the Jacobian kernel by scanning each column sequentially. Under the same B=256 complete six-update Newton test, itemwise assembly with binary search took **11.746 ms**, while the column version took **28.188 ms**: approximately **2.40× slower**. Serial intra-column work, long/short-column divergence, and store patterns outweighed the benefit of removing binary search. The column version remained numerically correct, with a maximum reference-voltage component error of `4.977129819e-13`, but the default path was restored to the itemwise kernel. The column kernel was retained for later warp execution-group experiments.

Raw results: `results/acpf-newton-batch2-fp64-results.json`, `results/acpf-newton-batch256-fp64-results.json`, `results/acpf-newton-entry-batch256-fp64-results.json`, and `results/acpf-newton-column-batch256-fp64-results.json`.

### Offline NVCC and L2/L4/L8 cooperative-column assembly

Kernel compilation moved to build-time NVCC. The program loads `target/acpf-cuda/kernels.ptx` and `newton_kernels.ptx`; the runtime path no longer invokes NVRTC. The Windows local code page misread UTF-8 Chinese comments without a BOM, so the build script first writes temporary `.cu` files with a BOM to prevent encoding problems from corrupting kernel declarations.

Cooperative assembly with 2/4/8 lanes per column was added. Each lane receives the known column number and CSR row segment directly, without binary search. PQ/PV row-segment templates are separated, the same lane applies the diagonal correction, and no atomics are used. Results at B=256 over a complete six-update Newton run follow; the J column is the total for six assemblies.

| Jacobian kernel | Total J time | Complete Newton wall time | Maximum final FP64 mismatch | Maximum reference-voltage component error |
|---|---:|---:|---:|---:|
| Itemwise | **11.644 ms** | 1,801.974 ms | 1.656497162e-11 | 6.680767051e-13 |
| L2 | 22.032 ms | 1,818.972 ms | 1.428590579e-11 | 5.587752483e-13 |
| L4 | **19.016 ms (best grouped variant)** | 1,805.046 ms | 1.656896842e-11 | 6.132871988e-13 |
| L8 | 22.355 ms | 1,782.368 ms | 1.609867795e-11 | 3.802513859e-13 |

All four paths completed six updates and converged for every scenario. Grouping removed binary search but did not accelerate assembly for this matrix; the itemwise kernel remains the default. Wall time is dominated by LU variation, so small wall-time differences cannot rank the J kernels reliably.

### `column_of_nnz` itemwise assembly without binary search

During symbolic preparation, a shared `int column_of_nnz[active_ynnz]` array is generated from `y_col_ptr`. Each GPU entry thread obtains `k` through one contiguous read, then computes the four destination offsets directly from `t = p - y_col_ptr[k]` and `row_ptr`. This preserves itemwise scheduling, does not store four complete destination-coordinate arrays, and performs no binary search.

B=256, complete six-update Newton, with the two paths alternated three times:

| Metric | Itemwise + binary search | `column_of_nnz` mapping | Change |
|---|---:|---:|---:|
| Median total J time | 12.333 ms | **9.833 ms** | **−20.27%** |
| J time per update | 2.055 ms | **1.639 ms** | −20.27% |
| Median complete wall time | 1,880.604 ms | 1,870.146 ms | Dominated by LU variation |

All three mapped runs completed six updates and converged for every scenario. Maximum final mismatch was `1.739230981e-11`, and maximum reference-voltage component error was `6.855627177e-13`. This optimization improved assembly itself but did not change the conclusion that factor+solve dominates the complete path. It is enabled through the fifth argument `map` and has not replaced the default path:

```bash
target/acpf-cuda/acpf_batch_demo.exe target/acpf-cuda/pegase9241-jacobian.bin 256 1 newton map
```

Alternating raw results are stored in `target/acpf-cuda/map-bench-{itemwise,mapped}-r{1,2,3}.json`.

### FP32 linear solves with FP64 state and residuals

A `newton32` complete Newton path was added. The Jacobian, power/mismatch, voltage state, and convergence checks remain FP64. Only the Jacobian and RHS are converted to FP32 for cuDSS factor/refactor and solve. The resulting FP32 Newton direction is then used for the FP64 voltage update. Final acceptance still requires an FP64 mismatch infinity norm no greater than `1e-8`; mixed precision was not rejected based on a single-step dx error, and the tolerance was not relaxed.

B=256 comparison using the same itemwise J kernel, with six updates for every scenario:

| Metric | FP64 solve | FP32 solve + FP64 residual | Change |
|---|---:|---:|---:|
| Total FP32 J/RHS conversion | — | 7.593 ms | — |
| Total factor/refactor | 1,438.902 ms | 1,233.997 ms | −14.24% |
| Total solve | 230.202 ms | 187.097 ms | −18.73% |
| **Complete Newton wall time** | **1,801.974 ms** | **1,561.405 ms** | **−13.35%** |
| Complete power-flow throughput | 142.07/s | 163.95/s | 1.154× |
| Maximum final FP64 mismatch | 1.656497162e-11 | 1.341172073e-09 | Both pass 1e-8 |
| Maximum reference-voltage component error | 6.680767051e-13 | 3.687250505e-11 | Both pass validation |

This demonstrates that the FP32 direction did not increase the iteration count for this case and produced a real end-to-end gain. It remains an experimental result for one topology, one perturbation family, and one convergence threshold; it should not be extrapolated to every ill-conditioned network.

Reproduction:

```bash
cmd.exe /c native\acpf_cuda\build.cmd
target/acpf-cuda/acpf_batch_demo.exe target/acpf-cuda/pegase9241-jacobian.bin 256 1 newton 4
target/acpf-cuda/acpf_batch_demo.exe target/acpf-cuda/pegase9241-jacobian.bin 256 1 newton32
```

Raw JSON for this round is stored in `target/acpf-cuda/newton-fp64-{itemwise,l2,l4,l8}-b256.json` and `target/acpf-cuda/newton-mixed-itemwise-b256.json`.

### Complete Newton batch sweep

All points use the same itemwise Jacobian, the same topology-perturbation family, six updates for every scenario, and final FP64 residual and reference-voltage validation. Each point is one complete run, suitable for observing batch and capacity trends but not a replacement for a repeated-measure median benchmark.

| Batch | FP64 wall time | FP64 complete power flows/s | Mixed-precision wall time | Mixed-precision complete power flows/s |
|---:|---:|---:|---:|---:|
| 32 | 273.990 ms | 116.79 | 219.537 ms | 145.76 |
| 64 | 481.027 ms | 133.05 | 402.474 ms | 159.02 |
| 128 | 918.764 ms | 139.32 | 808.563 ms | 158.31 |
| 256 | 1,801.974 ms | 142.07 | 1,561.405 ms | 163.96 |
| 512 | 3,593.475 ms | 142.48 | 3,044.352 ms | 168.18 |
| 768 | 11,344.843 ms | 67.70 | **4,479.452 ms** | **171.45** |
| 1024 | No complete FP64 sweep | — | 193,890.592 ms | 5.28 |

Mixed precision performed best at B=768 in this sweep; FP64 already showed clear factor-dominated degradation at that size. Although mixed-precision B=1024 converged everywhere, with a maximum final mismatch of `1.499268e-09`, its wall time lost all practical value because factorization alone accumulated `192,206.180 ms`. Its cuDSS peak estimate was `6,267,692,836` bytes, excluding external FP64 Jacobian, state, RHS, and mixed-precision buffers. This point should therefore be recorded only as a capacity boundary, not as normal large-batch performance.

Raw sweep results are stored in `target/acpf-cuda/newton-fp64-itemwise-b*.json`, `target/acpf-cuda/newton-mixed-itemwise-b*.json`, and the corresponding `sweep-*.log` files.

### Interpreting the external RTX 4090 / B=64 figures

On this machine at B=64, one J update took FP64 `0.600 ms` and mixed precision `0.848 ms`; one factor+solve update took `74.301 ms` and `60.426 ms`, respectively. The external statement “RTX 4090, batch 64, J 2 ms” therefore does not imply that its J is faster than this machine's.

If its reported “14 ms linear solve” means one factor+solve update, it is indeed substantially faster than this machine. However, “60 ms total” cannot then describe the same complete six-update Newton path, because the linear part alone would already require 84 ms. It may instead describe one update, fewer iterations, or a different timing interval. Without iteration count, precision, matrix, and raw stage measurements, 60 ms cannot be converted directly into complete N-1 throughput.

### Rust FFI and multi-stream retesting

A directly linked `acpf_cuda.dll` and Rust `experimental_n1_cuda` wrapper were added. Each executor owns one fixed CUDA stream, one cuDSS handle, and an independent workspace. Every scenario starts strictly from Rust `SharedBaseline::v_init`, without the benchmark's artificial voltage perturbations. All PEGASE 9241 scenarios complete six updates. At single-stream B=64 and B=256, maximum voltage-component errors against Rust `newton_pf` were `1.06e-12` and `1.03e-12`, respectively.

Each point below executes three rounds concurrently. Wall time covers all three rounds on every stream and excludes executor creation and cuDSS analysis. Throughput is `total scenarios × 3 / concurrent wall time`.

| Configuration | Total scenarios/round | Three-round concurrent wall time | Complete power flows/s |
|---|---:|---:|---:|
| 1 stream × 256 | 256 | 5.197 s | 147.78 |
| 2 streams × 128 | 256 | 5.069 s | 151.50 |
| 4 streams × 64 | 256 | 5.261 s | 145.98 |
| 8 streams × 32 | 256 | 5.188 s | 148.03 |
| 2 streams × 256 | 512 | 10.504 s | 146.23 |

Every multi-stream scenario converged. Maximum final residual did not exceed `2.03e-11`, and maximum voltage-component error did not exceed `1.48e-12`. The first 2×128 run was 2.5% faster than 1×256, but alternating retests produced 148.85/s and 148.80/s, eliminating the advantage. The 2×256 configuration also showed no refactor-overlap benefit. Multi-stream correctness is therefore validated, but stable throughput improvement is not; the first-run variation must not be presented as a multi-stream speedup. The next diagnostic step would be to measure J, factor, solve, and host synchronization separately under concurrency to locate the actual serialization point.

## Appendix: completed graphics baseline

GravityMark 1.72, Direct3D 12, 2560 × 1440, Temporal AA, 200,000 asteroids, vertical sync disabled:

- Score: 20,611
- Average frame rate: 123.4 FPS
- Effective test duration: 167.0 seconds

The ray-tracing test was not continued, so no valid result is available.

## Reproduction

From this directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_cuda_benchmark.ps1
```

For a fair RTX 5070 comparison, keep the `cuda_benchmark.py` and `python-packages` versions unchanged and run them as-is after the other GPU becomes the current CUDA device. Save the new results separately rather than overwriting this RTX 2070 SUPER report.

## Files

- `results/cuda-results.json`: complete raw CUDA data and individual rounds.
- `results/cuda-benchmark.log`: official test console log.
- `results/environment.json`: hardware, driver, and anomaly status.
- `results/gravitymark-d3d12-1440p-fullscreen.log`: graphics baseline log.
- `results/gravitymark-d3d12-1440p-frametimes.csv`: per-frame graphics timings.
