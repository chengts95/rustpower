# RTX 5070 CUDA 性能档案

[English version](RTX-5070-CUDA.en.md)

测试日期：2026-10-29
状态：微基准与 PEGASE 9241 完整 Newton 均已完成，全部数值校验通过

## 一句话结论

RTX 5070 在相同脚本下取得 **30.59 TFLOP/s FP32 FMA**、**24.29 TFLOP/s FP32 GEMM**、**68.42 TFLOP/s FP16 Tensor GEMM** 和 **576.75 GB/s** 设备内复制带宽；在 Rust→CUDA DLL→cuDSS 的完整 6 次 Newton 路径中，512–768 批次的稳态吞吐约为 **420 次完整潮流/秒**。

## 测试环境

| 项目 | 记录 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| Compute Capability | 12.0（`sm_120`） |
| SM 数量 | 48 |
| CUDA 可见显存 | 11.914 GiB |
| 显存位宽 | 192 bit |
| CPU | Intel Core i7-10700K，8 核 16 线程 |
| 操作系统 | Windows 11，10.0.26300 |
| NVIDIA 驱动 | WMI：32.0.16.2101 |
| CUDA Driver API | 13.5 |
| CUDA Runtime | 12.9 |
| 微基准框架 | CuPy 14.2.0，Python 3.12.16 |
| ACPF 路径 | FP64 CUDA kernels + FP64 cuDSS，直接链接 `acpf_cuda.dll` |

微基准复用 2070 SUPER 的脚本和 Python/CUDA 依赖版本。2070 SUPER 的结果与原始文件未覆盖。

## CUDA 微基准对照

5070 列采用第三次完整运行的结果。中位值用于典型性能，最佳值保留单次峰值。

| 测试 | 2070 SUPER 中位值 | 5070 中位值 | 中位值倍率 | 5070 最佳值 |
|---|---:|---:|---:|---:|
| 设备内显存复制带宽 | 349.03 GB/s | 576.75 GB/s | 1.65× | 580.45 GB/s |
| FP32 FMA | 9.200 TFLOP/s | 30.593 TFLOP/s | 3.33× | 31.452 TFLOP/s |
| FP64 FMA | 0.2440 TFLOP/s | 0.456 TFLOP/s | 1.87× | 0.456 TFLOP/s |
| FP32 GEMM，4096³ | 8.663 TFLOP/s | 24.289 TFLOP/s | 2.80× | 24.842 TFLOP/s |
| FP16 Tensor GEMM，4096³ | 37.840 TFLOP/s | 68.423 TFLOP/s | 1.81× | 69.562 TFLOP/s |
| FP64 GEMM，4096³ | 0.2859 TFLOP/s | 0.471 TFLOP/s | 1.65× | 0.471 TFLOP/s |

测试方法与 2070 SUPER 档案一致：CUDA Event 只测 GPU 操作；复制使用两个 1 GiB 缓冲区，流量按读加写计算；GEMM 为 4096³，FP32 使用默认非 TF32 数学模式，FP16 启用 Tensor Op。

### 首轮异常与复测

第一次运行时，FP32 GEMM 逐轮耗时从约 18.5–20.9 ms 逐步下降到 5.51 ms，导致中位吞吐只有 7.41 TFLOP/s、最佳值却为 24.93 TFLOP/s。第二次和第三次完整运行的 FP32 GEMM 中位值分别为 24.685 和 24.289 TFLOP/s，因此最终采用第三轮，并把首轮单独保留为冷过程异常记录。

现有数据能确认首轮不可作为稳态成绩，但没有温度、功耗和实时频率证据，不能把恢复过程唯一归因于频率、驱动或库初始化。

## PEGASE 9241 完整批量 Newton

### 口径

- 9,241 母线，所有场景均从 Rust `SharedBaseline::v_init` 启动。
- J、功率/mismatch、电压状态、cuDSS refactor 和 solve 均为 FP64。
- 每场景完整执行 6 次 Newton 更新，最终阈值为 mismatch 无穷范数不超过 `1e-8`。
- 每个进程执行 3 轮；壁钟覆盖 3 轮求解，不包含 executor 创建和 cuDSS analysis。
- “三轮吞吐”按 `总场景数 × 3 / 三轮壁钟` 计算；“稳态吞吐”按三轮逐轮时间的中位数计算，便于观察首轮之后的性能。

### 批次扫描

| Batch | 三轮逐轮时间 / s | 三轮壁钟 / s | 三轮完整潮流/s | 稳态完整潮流/s |
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

B=1 最初连续两次以 `NonFinite` 失败。诊断确认初始 mismatch 正常，但第一次 cuDSS solve 返回的解向量全部非有限；设置 `UBATCH_SIZE=1` 与普通非批量模式都能复现。修复后，对外 B=1 在 executor 内部使用两个完全相同的场景，只返回第一个场景的结果，继续走已验证的 B=2 cuDSS 路径。B=1 以及 B=2／64／256 回归均完成 6 次更新并通过数值校验。

小批次补测表明，B=64 已达到最高稳态吞吐的约 97.9%，B=128–256 均约为 98.7%。在当前实现中，B=64–128 能以更低单批延迟和更少工作区获得接近峰值的吞吐，继续增加到 512–768 只换来最后约 1%–2% 的收益。B=16 及以下则处于明显的平台前爬坡区。

1024 仍完成全部求解，没有重演 2070 SUPER 上某些大 batch 配置的极端崩塌，但三轮吞吐和稳态吞吐均已下降；首轮与后续轮的差距也重新增大。没有采集逐阶段显存占用，因此不能把批次边界单纯归因于显存容量。

所有批次均为 6 次更新并收敛。扫描中的最大最终残差为 `1.894e-11`，最大电压分量绝对误差为 `1.420e-12`。

### 与 2070 SUPER 的同路径比较

2070 SUPER 的 Rust FFI 单流 `B=256` 三轮吞吐为 **147.78 次完整潮流/秒**；本次 5070 相同可执行路径、相同批大小和相同三轮口径为 **392.07 次完整潮流/秒**，约为 **2.65×**。

旧档案中的 C++ benchmark 使用人为电压扰动，而 Rust executor 使用统一的 Rust 初始电压，因此只有 Rust FFI 同路径结果用于这里的定量加速比。其他旧数据仅用于观察批次趋势，不混合计算加速比。

## 未测得与限制

- `nvidia-smi` 仍返回 `Failed to initialize NVML: Unknown Error`，因此没有可靠的温度、功耗和实时频率记录。
- 微基准不是实际 ACPF 性能；完整 Newton 吞吐也不是已经接入 N-1 调度后的故障扫描吞吐。
- 本轮只测单流 FP64 executor；没有重新宣称 multi-stream 加速，也没有测混合精度路径。
- 当前结果属于同一拓扑、相同初始状态的批量基态潮流实验，不能直接外推到所有病态电网或不同拓扑批次。

## 复现

在仓库根目录运行微基准：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\gpu-benchmarks\rtx-5070\run_cuda_benchmark.ps1
```

构建并运行 Rust CUDA ACPF：

```powershell
$env:CUDSS_DIR = (Resolve-Path .\gpu-benchmarks\rtx-2070-super\python-packages\nvidia\cu12).Path
$env:RUSTPOWER_CUDA_ARCH = "sm_120"
cargo run --release --features experimental_n1_cuda --bin bench_cuda_base -- cases/pegase9241/data.zip 256 1 3
```

## 原始文件

- `results/cuda-results.json`：第三轮微基准完整逐轮数据。
- `results/cuda-results-run1.json`：首轮冷过程异常数据。
- `results/cuda-results-run2.json`：第二轮复测数据。
- `results/cuda-benchmark.log`：第三轮控制台日志。
- `results/acpf-batch-{1,2,4,8,16,32,64,128,256,512,768,1024}.json`：完整 Newton 批次扫描原始结果。
- `results/acpf-regression-{1,2,64,256}.json`：B=1 修复后的回归结果。
