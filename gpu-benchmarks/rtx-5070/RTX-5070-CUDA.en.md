# RTX 5070 CUDA Performance Report

Test date: 2026-10-29  
N-1 validation refreshed: 2026-10-30  
Status: CUDA microbenchmarks, the complete PEGASE 9241 Newton path, and the full PEGASE 9241 N-1 sweep have passed numerical validation.

[中文版本](RTX-5070-CUDA.md)

## Executive summary

The RTX 5070 achieved a median **30.59 TFLOP/s in FP32 FMA**, **24.29 TFLOP/s in FP32 GEMM**, **68.42 TFLOP/s in FP16 Tensor GEMM**, and **576.75 GB/s of device-to-device copy bandwidth**. On the complete Rust → CUDA DLL → cuDSS path with six Newton updates per scenario, sustained throughput reached approximately **420 complete power flows/s** at batch sizes 512–768.

Compared with the RTX 2070 SUPER under the same scripts and measurement rules, the main improvements are:

| Workload | RTX 2070 SUPER | RTX 5070 | Improvement |
|---|---:|---:|---:|
| Device-to-device copy bandwidth | 349.03 GB/s | 576.75 GB/s | **1.65×** |
| FP32 FMA | 9.200 TFLOP/s | 30.593 TFLOP/s | **3.33×** |
| FP64 FMA | 0.2440 TFLOP/s | 0.456 TFLOP/s | **1.87×** |
| FP32 GEMM, 4096³ | 8.663 TFLOP/s | 24.289 TFLOP/s | **2.80×** |
| FP16 Tensor GEMM, 4096³ | 37.840 TFLOP/s | 68.423 TFLOP/s | **1.81×** |
| FP64 GEMM, 4096³ | 0.2859 TFLOP/s | 0.471 TFLOP/s | **1.65×** |
| PEGASE 9241 complete Newton, Rust FFI, B256 | 147.78 power flows/s | 392.07 power flows/s | **2.65×** |

The microbenchmarks show a particularly large FP32 improvement, while the FP64 gain is more modest. The complete ACPF result sits between those two regimes: it reflects the real FP64 sparse-linear-algebra path rather than peak arithmetic throughput.

## Test environment

| Item | Record |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| Compute Capability | 12.0 (`sm_120`) |
| SM count | 48 |
| CUDA-visible memory | 11.914 GiB |
| Memory bus width | 192 bit |
| CPU | Intel Core i7-10700K, 8 cores / 16 threads |
| Operating system | Windows 11, 10.0.26300 |
| NVIDIA driver | WMI: 32.0.16.2101 |
| CUDA Driver API | 13.5 |
| CUDA Runtime | 12.9 |
| Microbenchmark framework | CuPy 14.2.0, Python 3.12.16 |
| ACPF path | FP64 CUDA kernels + FP64 cuDSS, directly linked through `acpf_cuda.dll` |

The microbenchmarks reuse the RTX 2070 SUPER scripts and Python/CUDA dependency versions. The original RTX 2070 SUPER results and report were preserved rather than overwritten.

## CUDA microbenchmarks

The RTX 5070 column uses the third complete run. Median values represent typical performance; best values retain the fastest individual round.

| Test | RTX 2070 SUPER median | RTX 5070 median | Median ratio | RTX 5070 best |
|---|---:|---:|---:|---:|
| Device-to-device copy bandwidth | 349.03 GB/s | 576.75 GB/s | 1.65× | 580.45 GB/s |
| FP32 FMA | 9.200 TFLOP/s | 30.593 TFLOP/s | 3.33× | 31.452 TFLOP/s |
| FP64 FMA | 0.2440 TFLOP/s | 0.456 TFLOP/s | 1.87× | 0.456 TFLOP/s |
| FP32 GEMM, 4096³ | 8.663 TFLOP/s | 24.289 TFLOP/s | 2.80× | 24.842 TFLOP/s |
| FP16 Tensor GEMM, 4096³ | 37.840 TFLOP/s | 68.423 TFLOP/s | 1.81× | 69.562 TFLOP/s |
| FP64 GEMM, 4096³ | 0.2859 TFLOP/s | 0.471 TFLOP/s | 1.65× | 0.471 TFLOP/s |

The methodology matches the RTX 2070 SUPER report. CUDA Events measure GPU operations only. Copy bandwidth uses two 1 GiB buffers and counts one read plus one write. GEMM uses 4096³ matrices; FP32 uses the default non-TF32 math mode, while FP16 enables Tensor Operations.

### First-run anomaly and retesting

During the first run, FP32 GEMM round time decreased progressively from approximately 18.5–20.9 ms to 5.51 ms. Its median throughput was therefore only 7.41 TFLOP/s, despite a best-round value of 24.93 TFLOP/s. The second and third complete runs produced median values of 24.685 and 24.289 TFLOP/s, respectively. The third run is used as the steady-state record, and the first run is retained separately as a cold-process anomaly.

The available data establishes that the first run is not representative of steady-state performance. Without temperature, power, and instantaneous clock measurements, the recovery cannot be attributed uniquely to clocks, the driver, or library initialization.

## PEGASE 9241 complete batched Newton

### Measurement rules

- 9,241 buses; every scenario starts from Rust `SharedBaseline::v_init`.
- Jacobian values, power/mismatch evaluation, voltage state, cuDSS refactorization, and solves all use FP64.
- Every scenario executes six complete Newton updates, with a final mismatch infinity norm no greater than `1e-8`.
- Each process executes three rounds. Wall time covers all three solves and excludes executor creation and cuDSS analysis.
- Three-round throughput is `scenario count × 3 / three-round wall time`. Sustained throughput uses the median of the three individual round times.

### Batch sweep

| Batch | Individual round times / s | Three-round wall time / s | Three-round power flows/s | Sustained power flows/s |
|---:|---|---:|---:|---:|
| 1 | 0.087 / 0.008 / 0.008 | 0.106 | 28.29 | 117.90 |
| 2 | 0.088 / 0.009 / 0.009 | 0.108 | 55.75 | 219.01 |
| 4 | 0.093 / 0.013 / 0.012 | 0.121 | 99.37 | 312.35 |
| 8 | 0.101 / 0.021 / 0.021 | 0.146 | 164.92 | 374.02 |
| 16 | 0.118 / 0.041 / 0.040 | 0.201 | 238.63 | 394.96 |
| 32 | 0.156 / 0.079 / 0.078 | 0.318 | 302.00 | 402.62 |
| 64 | 0.246 / 0.155 / 0.155 | 0.564 | 340.46 | 411.96 |
| 128 | 0.397 / 0.306 / 0.308 | 1.023 | 375.27 | 415.05 |
| 256 | 0.706 / 0.616 / 0.615 | 1.959 | 392.07 | 415.31 |
| 512 | 1.327 / 1.217 / 1.226 | 3.803 | 403.92 | 417.77 |
| 768 | 1.975 / 1.825 / 1.819 | 5.667 | **406.55** | **420.71** |
| 1024 | 4.348 / 2.807 / 2.797 | 10.208 | 300.95 | 364.83 |

B=1 initially failed twice with `NonFinite`. The initial mismatch was valid, but the first cuDSS solve returned a non-finite solution vector. The issue reproduced with both `UBATCH_SIZE=1` and the ordinary non-batched mode. The executor now pads an external B=1 request with one identical shadow scenario, solves through the validated B=2 path, and returns only the first result. B=1 and B=2/64/256 regressions all completed six updates and passed numerical validation.

The small-batch sweep shows that B=64 already reaches approximately 97.9% of peak sustained throughput, while B=128–256 reaches approximately 98.7%. B=64–128 therefore provides near-peak throughput with lower batch latency and a smaller workspace. Increasing the batch to 512–768 gains only the final 1%–2%.

B=1024 still converged and did not reproduce the extreme collapse observed for some large batches on the RTX 2070 SUPER. Nevertheless, both throughput metrics declined, and the gap between the first and later rounds increased again. Without per-stage memory measurements, this boundary cannot be attributed solely to memory capacity.

All batch cases completed six updates and converged. Across the sweep, the maximum final residual was `1.894e-11`, and the maximum absolute voltage-component error was `1.420e-12`.

### Same-path comparison with RTX 2070 SUPER

The RTX 2070 SUPER achieved **147.78 complete power flows/s** on the single-stream Rust FFI path at B=256. The RTX 5070 achieved **392.07 complete power flows/s** on the same executable path, with the same batch size and three-round measurement rule: a **2.65× improvement**.

The older C++ benchmark used artificial voltage perturbations, whereas the Rust executor uses the common Rust initial voltage. Only the same-path Rust FFI results are therefore used for the quantitative speedup. Other older measurements are retained for batch-trend analysis and are not mixed into this ratio.

## Full PEGASE 9241 N-1 sweep

The current executor also completed the real full N-1 sweep with shared island screening, base-AC initialization, branch postprocessing, and constraint checks.

| Metric | RTX 5070, B64, 1 stream | CPU, 8 workers |
|---|---:|---:|
| Branch outages | 16,049 | 16,049 |
| Islanded cases excluded before numerical solves | 1,665 | 1,665 |
| Numerical scenarios | 14,384 | 14,384 |
| Failed numerical scenarios | 10 | 10 |
| Cases with constraint violations | 485 | 485 |
| End-to-end time | **22.669 s** | **60.251 s** |
| Numerical throughput | **634.53 scenarios/s** | **238.73 scenarios/s** |

The RTX 5070 was **2.658× faster end to end** in this run. Both paths used the same `N1Study::partition` island classification and the same prepared watch plan. The validation subset showed exactly matching iteration counts and a maximum absolute voltage error of `1.408e-12`.

This is a large-system result, not a promise that CUDA should receive every N-1 workload. Small systems can remain faster on the CPU because local KLU solves, cache locality, and multi-core scheduling outweigh GPU batching benefits.

## CPU + CUDA hybrid N-1

`run_n1_hybrid` and `run_n1_study_hybrid` use two independent bounded queues. The dispatcher first attempts a nonblocking send to the dedicated CUDA queue; only when that queue is full does the task overflow to the CPU queue. `cuda_queue_batches` retains two batches per stream by default. This is not a static partition: the CPU consumes overflow while CUDA is busy, and CUDA again receives priority when capacity becomes available.

Queueing is independent of solve batch size. CPU workers still consume small groups, while each CUDA consumer assembles its own fixed large batch. Tail padding is never counted as additional scenarios. The producer compiles each `OutageTask` once, and both backends retain the same base-AC initialization, island screening, watch plan, events, and convergence semantics. `HybridN1Summary` preserves separate CPU, CUDA, and combined statistics.

Full real PEGASE 9241 N-1, B=64, one CUDA stream, and four CPU workers, measured in one three-way comparison:

| Path | Numerical scenarios | Wall time | Numerical throughput | Failed | Violated | Islanded |
|---|---:|---:|---:|---:|---:|---:|
| CPU only | 14,384 | 58.765 s | 244.77/s | 10 | 485 | 1,665 |
| CUDA only | 14,384 | 23.036 s | 624.42/s | 10 | 485 | 1,665 |
| Hybrid | 14,384 | **19.544 s** | **735.98/s** | 10 | 485 | 1,665 |

CUDA completed 9,967 scenarios and the CPU completed 4,417, with exactly 14,384 submissions and processed results. Hybrid throughput was **3.007×** the CPU-only result and **1.179×** the CUDA-only result, establishing a clear end-to-end gain. All three paths reported identical failure, violation, and island counts; the validation subset had a maximum absolute voltage error of `1.481e-12`.

Adding CPU workers indiscriminately is not beneficial: an otherwise identical eight-worker run took 26.500 seconds, only **0.883×** the CUDA-only throughput. For this machine at B=64, four-worker overflow scheduling outperformed that eight-worker configuration. Queue depth and CPU worker count should still be retested for other GPUs and network sizes.

An additional IEEE118 run with eight repetitions processed 1,416 scenarios: 1,089 on the CPU and 327 on CUDA. Hybrid execution took 0.034432 seconds, only **1.008×** the CPU-only throughput but **2.675×** the CUDA-only throughput. Light workloads therefore have no compelling reason to enable CUDA.

## Limitations and unmeasured items

- `nvidia-smi` still returns `Failed to initialize NVML: Unknown Error`; reliable temperature, power, and instantaneous clock records are therefore unavailable.
- CUDA microbenchmarks are not application-level ACPF benchmarks, and the complete Newton batch sweep is not an N-1 contingency scan. The N-1 results are reported separately above.
- The complete Newton batch sweep used a single stream and FP64. It does not establish a multi-stream speedup or validate a mixed-precision path.
- Batch results use one topology and a common initial state. They should not be extrapolated to every ill-conditioned network or mixed-topology batch.
- The N-1 comparison covers this machine and the current CPU/GPU implementations. Hardware differences must not be confused with algorithm-only comparisons.

## Reproduction

Run the microbenchmarks from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gpu-benchmarks\rtx-5070\run_cuda_benchmark.ps1
```

Build and run the Rust CUDA ACPF batch benchmark:

```powershell
$env:CUDSS_DIR = (Resolve-Path .\gpu-benchmarks\rtx-2070-super\python-packages\nvidia\cu12).Path
$env:RUSTPOWER_CUDA_ARCH = "sm_120"
cargo run --release --features experimental_n1_cuda --bin bench_cuda_base -- cases/pegase9241/data.zip 256 1 3
```

Run the full real N-1 three-way comparison (CPU, CUDA, and hybrid):

```powershell
cargo run --release --features experimental_n1_cuda --bin bench_cuda_n1 -- cases/pegase9241/data.zip 64 4 16049 1 1
```

## Raw files

- `results/cuda-results.json`: third complete microbenchmark run with every round.
- `results/cuda-results-run1.json`: first-run cold-process anomaly.
- `results/cuda-results-run2.json`: second complete retest.
- `results/cuda-benchmark.log`: third-run console log.
- `results/acpf-batch-{1,2,4,8,16,32,64,128,256,512,768,1024}.json`: complete Newton batch sweep.
- `results/acpf-regression-{1,2,64,256}.json`: regressions after the B=1 fix.
- `results/n1-pegase9241-full-b64-s1.json`: earlier full PEGASE 9241 CPU/CUDA comparison.
- `results/n1-pegase9241-hybrid-priority-b64-s1-c4.json`: full PEGASE 9241 comparison with four CPU workers and CUDA-queue overflow dispatch.
- `results/n1-pegase9241-hybrid-priority-b64-s1-c8.json`: eight-CPU-worker overflow comparison.
- `results/n1-ieee118-hybrid-priority-repeat8-b64-s1-c2.json`: IEEE118 overflow comparison with eight repetitions.
- `results/n1-ieee118-*.json`: other IEEE118 batch, multi-stream, and repeated-scenario experiments.
