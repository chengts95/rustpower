# RTX 2070 SUPER CUDA 性能档案

[English version](RTX-2070-SUPER-CUDA.en.md) · [RTX 5070 英文对比报告](../rtx-5070/RTX-5070-CUDA.en.md)

测试日期：2026-10-29
状态：完整测试，所有计算结果均通过校验

## 一句话结论

这张 RTX 2070 SUPER 在本次纯计算微基准中取得了 **9.20 TFLOP/s FP32 FMA**、**8.66 TFLOP/s FP32 GEMM** 和 **37.84 TFLOP/s FP16 Tensor GEMM** 的中位成绩；设备内显存复制最佳达到 **391.73 GB/s**。FP16 Tensor GEMM 的中位吞吐约为 FP32 GEMM 的 **4.37 倍**。

## 测试环境

| 项目 | 记录 |
|---|---|
| GPU | NVIDIA GeForce RTX 2070 SUPER，微星版本，8 GB |
| Compute Capability | 7.5 |
| SM 数量 | 40 |
| 显存位宽 | 256 bit |
| CPU | Intel Core i7-10700K，8 核 16 线程 |
| 主板 | MSI Z490-A PRO (MS-7C75) |
| 内存 | 31.92 GiB 可识别容量 |
| 操作系统 | Windows 11 Home，10.0.26300 |
| NVIDIA 驱动 | 610.62（WMI：32.0.16.1062） |
| CUDA Driver API | 13.3 |
| CUDA Runtime | 12.9 |
| 测试框架 | CuPy 14.2.0，Python 3.12.15 |
| Windows 电源模式 | 卓越性能 / Ultimate Performance |

## 成绩

“中位值”代表本次重复测量的典型表现，“最佳值”代表最快一轮；两者都保留，不用单个峰值掩盖波动。

| 测试 | 中位值 | 最佳值 | 重复次数 |
|---|---:|---:|---:|
| 设备内显存复制带宽 | 349.03 GB/s | 391.73 GB/s | 10 |
| FP32 FMA | 9.200 TFLOP/s | 9.860 TFLOP/s | 7 |
| FP64 FMA | 0.2440 TFLOP/s | 0.2476 TFLOP/s | 5 |
| FP32 GEMM，4096³ | 8.663 TFLOP/s | 9.416 TFLOP/s | 10 |
| FP16 Tensor GEMM，4096³ | 37.840 TFLOP/s | 38.893 TFLOP/s | 10 |
| FP64 GEMM，4096³ | 0.2859 TFLOP/s | 0.2874 TFLOP/s | 5 |

## 测试方法

- 所有正式成绩均使用 CUDA Event 测量 GPU 执行时间，不包含数组初始化、首次编译和预热。
- 显存带宽使用两个 1 GiB 缓冲区做设备到设备复制，流量按一次读取加一次写入计算；预热 3 轮，正式测量 10 轮。
- FMA 使用自定义 CUDA 内核，40,960 个线程，每线程 8 条独立计算链。FP32 每轮 262,144 次迭代，FP64 每轮 32,768 次迭代。
- GEMM 使用 cuBLAS，矩阵尺寸为 4096 × 4096 × 4096。FP32 明确使用默认非 TF32 数学模式，FP16 启用 Tensor Op 数学模式。
- GEMM 使用全一输入矩阵并抽查输出为 4096；自定义 FMA 内核检查输出非零且均为有限值。
- 原始 JSON 保存每一轮耗时和吞吐，不只保存汇总数字。

## 如何解读

- 这是**纯 CUDA 微基准**，适合以后在相同脚本、相同矩阵尺寸和相同依赖版本下比较另一张显卡。
- GEMM 成绩与实际深度学习训练或推理并不等价；真实模型还会受到显存容量、内存访问、算子融合、批量大小和框架开销影响。
- 显存复制在本次会话中有明显瞬时波动，因此中位值低于最佳值；两组数字都应保留。
- FP16 结果只代表本次 HGEMM 形状和 cuBLAS 配置，不应直接当成所有 FP16 工作负载的固定倍率。

## 未测得与异常

- `nvidia-smi` 返回 `Failed to initialize NVML: Unknown Error`，因此本档案**没有可靠的温度、功耗或测试期间实时频率数据**，不做推测补值。
- 系统中的若干 CUDA Toolkit 目录只有残留文件；本次使用项目目录内隔离安装的 CuPy 与 CUDA 运行组件，没有修复或改动系统 CUDA 配置。
- 温度、功耗和真实 AI 模型性能需要另行测试。

## 附：PEGASE 9241 批量 cuDSS FP64 原型

测试使用真实 Newton 迭代 1 的 Jacobian，矩阵阶数 17,036，每系统 129,412 个非零元。C++ 宿主通过 CUDA Driver API 启动批量 kernels，cuDSS 1.0 使用 uniform batch 和全活跃 `CUDSS_DATA_UBATCH_MASK`。各系统具有不同矩阵数值和右端；以预先构造的已知解逐元素校验，不做跨场景共享 LU。

### 批量扫描：仅 refactor + solve

每批运行 3 轮，表中取 factor/refactor 与 solve 各自的中位值。单系统稳态基准为 2.0716 ms，即约 482.72 systems/s。

| Batch | factor/refactor 中位值 | solve 中位值 | 合计吞吐 | cuDSS peak estimate |
|---:|---:|---:|---:|---:|
| 8 | 8.924 ms | 1.928 ms | 737.15 systems/s | 0.057 GiB |
| 32 | 34.769 ms | 5.366 ms | 797.32 systems/s | 0.217 GiB |
| 128 | 115.780 ms | 19.639 ms | 945.22 systems/s | 0.860 GiB |
| 256 | 229.145 ms | 36.046 ms | **965.34 systems/s** | 1.717 GiB |
| 512 | 485.399 ms | 71.766 ms | 918.94 systems/s | 3.430 GiB |
| 768 | 687.492 ms | 109.906 ms | 963.13 systems/s | 5.143 GiB |
| 1024 | 32,444.033 ms | 164.344 ms | 31.40 systems/s | 6.857 GiB |

正常容量区间内批量约有 **1.5–2.0 倍**吞吐收益，256 是本次扫描的高效点之一。1024 出现非线性崩塌；其 cuDSS 估算尚未包含外部批量 J、RHS 和解向量，不能把这个容量边界点解释为 batch LU 的一般性能。1024 并非本卡这张矩阵的合理默认批量。

### 真正的批量 CSR 组装与 AC eval

随后将“缩放现成 J”替换为 `JacobianOperator::fill::<true, false>` 的 GPU 移植：按 Ybus CSC 项并行，使用 `mirror`、`pq_ends`、`active_ends` 和 `diag_ptrs` 直接写 CSR 四象限。同一镜像遍历批量计算 `S_calc` 与 mismatch。每个场景从实际 Ybus 和不同电压状态重新组装，不再读取预制 J 数值。

`B=256` 的分阶段结果：

| 阶段 | 整批耗时 | 折合每系统 |
|---|---:|---:|
| 电压准备 | 0.458 ms | 1.79 μs |
| 批量功率与 mismatch | 17.811 ms | 69.57 μs |
| **批量 CSR Jacobian 组装** | **2.286 ms** | **8.93 μs** |
| factor/refactor 中位值 | 248.484 ms | 970.64 μs |
| solve 中位值 | 37.158 ms | 145.15 μs |

功率/mismatch + CSR 组装 + refactor + solve 合计 305.740 ms，约 **837.31 systems/s**。这里不包含电压更新、收敛判断和完整 Newton 循环，因此不是完整 ACPF 吞吐。组装本身只占该阶段组合的约 0.75%，当前主要成本仍在 LU。

参考状态校验：J 最大绝对误差 `5.456968211e-12`，mismatch 最大绝对误差 `4.485301019e-12`；256 系统已知解校验最大相对误差 `3.332149371e-12`，无非有限解。

原始单步结果保存在 `results/cudss-pegase9241-batch*-fp64-results.json` 与 `results/acpf-assembly-batch*-fp64-results.json`。

### 完整批量 Newton 潮流

保留上述 NVRTC 单步路径，在同一程序新增 `newton` 模式。每次更新都重新执行功率/mismatch、CSR J 组装、cuDSS refactor、solve 和电压更新；每轮在 GPU 上计算逐场景 mismatch 无穷范数，以 `1e-8` 为阈值更新 `CUDSS_DATA_UBATCH_MASK`。已收敛场景退出后续核函数和线性求解。

pandapower 参考潮流从相同 flat 初态经过 **6 次**更新收敛，参考 mismatch 无穷范数为 `8.487752065e-12`。GPU 的 `B=2` 与 `B=256` 测试均为每场景 **6 次**更新并全部收敛。

`B=256` 的完整 6 轮结果：

| 阶段 | 6 轮累计 |
|---|---:|
| 功率与 mismatch（7 次 eval，含最终检查） | 102.697 ms |
| CSR Jacobian 组装（6 次） | 11.746 ms |
| 收敛检查（7 次） | 1.061 ms |
| 电压更新（6 次） | 9.625 ms |
| factor/refactor（6 次） | 1,397.868 ms |
| solve（6 次） | 218.104 ms |
| **完整 Newton 壁钟** | **1,742.531 ms** |

折合 **6.807 ms/完整潮流**、**146.91 次完整潮流/秒**，或 881.48 次 Newton 系统更新/秒。壁钟包含逐轮 mask 与范数回读，不包含可复用的 symbolic analysis 和前面的单步预检。factor/refactor 占 Newton 壁钟约 80.22%，CSR 组装约占 0.67%。

全部场景最终 mismatch 最大值为 `1.740252387e-11`，第 0 场景与参考收敛电压的最大实部/虚部分量误差为 `9.587886041e-13`，无非有限电压。当前扰动使所有场景都在第 6 次后一起收敛，因此尚未实测不同场景 5/6/7 次混合退出的性能。managed-memory 双缓冲和 N-1 executor 仍未接入。

复现命令：

```bash
target/acpf-cuda/acpf_batch_demo.exe target/acpf-cuda/pegase9241-jacobian.bin 256 1 newton
```

### 逐列无线程二分组装对照

按一列一个 thread、列内顺序扫描的方式移除了 J 核函数中的二分查找。相同 `B=256`、完整 6 轮 Newton 下，逐项 + 二分的组装累计为 **11.746 ms**，逐列版为 **28.188 ms**；逐列版约慢 **2.40 倍**。列内串行、长短列分歧和写出模式的影响大于二分查找收益。逐列版数值仍正确，参考电压最大分量误差为 `4.977129819e-13`，但默认路径已恢复逐项版；逐列内核保留供后续 warp 执行组实验。

原始结果：`results/acpf-newton-batch2-fp64-results.json`、`results/acpf-newton-batch256-fp64-results.json`、`results/acpf-newton-entry-batch256-fp64-results.json`、`results/acpf-newton-column-batch256-fp64-results.json`。

### NVCC 离线与 L2/L4/L8 协作列组装

核函数已改为构建期 NVCC 编译，程序加载 `target/acpf-cuda/kernels.ptx` 和 `newton_kernels.ptx`，运行路径不再调用 NVRTC 编译。Windows 本地代码页会误读无 BOM 的 UTF-8 中文注释，构建脚本因此先写出带 BOM 的临时 `.cu`，避免编码问题破坏核函数声明。

新增每列 2/4/8 lane 的协作组装：直接已知列号和 CSR 行段，不执行二分查找，PQ/PV 行段模板分离，对角由同一 lane 修正，无原子操作。`B=256`、完整 6 轮 Newton 的结果如下；J 列为 6 次组装的累计耗时。

| Jacobian 内核 | J 累计 | 完整 Newton 壁钟 | 最大最终 FP64 mismatch | 参考电压最大分量误差 |
|---|---:|---:|---:|---:|
| 逐项版 | **11.644 ms** | 1,801.974 ms | 1.656497162e-11 | 6.680767051e-13 |
| L2 | 22.032 ms | 1,818.972 ms | 1.428590579e-11 | 5.587752483e-13 |
| L4 | **19.016 ms（分组中最佳）** | 1,805.046 ms | 1.656896842e-11 | 6.132871988e-13 |
| L8 | 22.355 ms | 1,782.368 ms | 1.609867795e-11 | 3.802513859e-13 |

四种路径均为全部场景 6 次更新并收敛。分组版消除了二分查找，但没有提升本矩阵的组装速度；当前默认仍为逐项版。壁钟主要受 LU 波动支配，不能据壁钟微小差异给 J 内核排名。

### `column_of_nnz` 无二分逐项组装

在符号准备阶段从 `y_col_ptr` 生成一张共享 `int column_of_nnz[active_ynnz]`，GPU 每项线程通过一次连续读取获得 `k`，再由 `t = p - y_col_ptr[k]` 和 `row_ptr` 直接计算四象限目的偏移。它保留逐项版的调度方式，不保存四套完整目的坐标，也不执行二分查找。

`B=256`、完整 6 轮 Newton、两种路径交替运行 3 次：

| 指标 | 逐项 + 二分 | `column_of_nnz` 映射 | 变化 |
|---|---:|---:|---:|
| J 累计中位数 | 12.333 ms | **9.833 ms** | **-20.27%** |
| 折合每轮 J | 2.055 ms | **1.639 ms** | -20.27% |
| 完整壁钟中位数 | 1,880.604 ms | 1,870.146 ms | 受 LU 波动支配 |

三轮映射版均为全部场景 6 次更新并收敛；最大最终 mismatch 为 `1.739230981e-11`，参考电压最大分量误差为 `6.855627177e-13`。这项优化改善了组装本身，但不改变 factor+solve 占主导的完整流程结论。当前通过第五参数 `map` 启用，尚未替换默认路径：

```bash
target/acpf-cuda/acpf_batch_demo.exe target/acpf-cuda/pegase9241-jacobian.bin 256 1 newton map
```

交替对照原始结果保存在 `target/acpf-cuda/map-bench-{itemwise,mapped}-r{1,2,3}.json`。

### FP32 线性求解 + FP64 状态与残差

新增 `newton32` 完整 Newton 路径：J、功率/mismatch、电压状态和收敛判断保持 FP64，仅将 J 与 RHS 转为 FP32 供 cuDSS factor/refactor 和 solve，再把 FP32 牛顿方向用于 FP64 电压更新。最终仍统一要求 mismatch 无穷范数不超过 `1e-8`，没有用单步 dx 误差否定混合精度，也没有放宽收敛阈值。

`B=256` 对照（同一逐项 J 内核，全部场景均为 6 次更新）：

| 指标 | FP64 求解 | FP32 求解 + FP64 残差 | 变化 |
|---|---:|---:|---:|
| FP32 J/RHS 转换累计 | — | 7.593 ms | — |
| factor/refactor 累计 | 1,438.902 ms | 1,233.997 ms | -14.24% |
| solve 累计 | 230.202 ms | 187.097 ms | -18.73% |
| **完整 Newton 壁钟** | **1,801.974 ms** | **1,561.405 ms** | **-13.35%** |
| 完整潮流吞吐 | 142.07/s | 163.95/s | 1.154× |
| 最大最终 FP64 mismatch | 1.656497162e-11 | 1.341172073e-09 | 均通过 1e-8 |
| 参考电压最大分量误差 | 6.680767051e-13 | 3.687250505e-11 | 均通过 1e-8 |

这证明本算例中 FP32 方向没有增加迭代次数，并带来了实际的完整流程收益；但它仍是单拓扑、相同扰动族和固定收敛阈值的实验结果，不能直接外推到所有病态电网。

复现命令：

```bash
cmd.exe /c native\acpf_cuda\build.cmd
target/acpf-cuda/acpf_batch_demo.exe target/acpf-cuda/pegase9241-jacobian.bin 256 1 newton 4
target/acpf-cuda/acpf_batch_demo.exe target/acpf-cuda/pegase9241-jacobian.bin 256 1 newton32
```

本轮原始 JSON 位于 `target/acpf-cuda/newton-fp64-{itemwise,l2,l4,l8}-b256.json` 和 `target/acpf-cuda/newton-mixed-itemwise-b256.json`。

### 完整 Newton 批次扫描

使用相同逐项 J、相同拓扑扰动族，每个批次均为全部场景 6 次更新并通过最终 FP64 残差与参考电压校验。每个数据点为一次完整运行，适合观察批次和容量趋势，不替代多次重复的中位数基准。

| Batch | FP64 壁钟 | FP64 完整潮流/s | 混合精度壁钟 | 混合精度完整潮流/s |
|---:|---:|---:|---:|---:|
| 32 | 273.990 ms | 116.79 | 219.537 ms | 145.76 |
| 64 | 481.027 ms | 133.05 | 402.474 ms | 159.02 |
| 128 | 918.764 ms | 139.32 | 808.563 ms | 158.31 |
| 256 | 1,801.974 ms | 142.07 | 1,561.405 ms | 163.96 |
| 512 | 3,593.475 ms | 142.48 | 3,044.352 ms | 168.18 |
| 768 | 11,344.843 ms | 67.70 | **4,479.452 ms** | **171.45** |
| 1024 | 未做完整 FP64 扫描 | — | 193,890.592 ms | 5.28 |

混合精度在本次扫描中以 `B=768` 最佳；FP64 在 `B=768` 已出现 factor 主导的明显劣化。混合精度 `B=1024` 虽全部收敛，但最终 mismatch 最大值为 `1.499268e-09`，壁钟却因 factor 累计 `192,206.180 ms` 而完全失去实用价值。其 cuDSS peak estimate 为 `6,267,692,836` bytes，尚未包含外部 FP64 J、状态、RHS 和混合精度缓冲区，因此该点只应记录为容量边界，不能作为正常大 batch 性能。

### 关于另一条 4090 / B=64 数据的口径

本机 `B=64` 每轮 J 为 FP64 `0.600 ms`、混合精度 `0.848 ms`；每轮 factor+solve 分别为 `74.301 ms` 和 `60.426 ms`。因此，外部所述“4090、64 batch、J 2 ms”并不表示其 J 比本机更快。

若其“线性求解 14 ms”是一轮 factor+solve，则确实显著快于本机，但“总时间 60 ms”就不可能是同样 6 轮完整 Newton，因为仅线性部分已达到 84 ms。它也可能是单次更新、较少迭代或不同计时区间；在缺少迭代次数、精度、矩阵和分阶段原始数据时，不能把 60 ms 直接换算成完整 N-1 吞吐。

扫描原始结果位于 `target/acpf-cuda/newton-fp64-itemwise-b*.json`、`target/acpf-cuda/newton-mixed-itemwise-b*.json` 和对应 `sweep-*.log`。

### Rust FFI 与 multi-stream 复测

新增直接链接的 `acpf_cuda.dll` 与 Rust `experimental_n1_cuda` 封装。每个 executor 固定一条 CUDA stream、一个 cuDSS handle 和独立工作区，所有场景严格使用 Rust `SharedBaseline::v_init`，不再施加 benchmark 的人为电压扰动。PEGASE 9241 全部场景均为 6 次更新；单流 B=64 和 B=256 对 Rust `newton_pf` 的最大电压分量误差分别为 `1.06e-12` 和 `1.03e-12`。

下表每点并发执行 3 轮，壁钟覆盖所有 stream 的全部 3 轮，不包含 executor 创建和 cuDSS analysis。吞吐按 `总场景数 × 3 / 并发壁钟` 计算。

| 配置 | 总场景数/轮 | 3 轮并发壁钟 | 完整潮流/s |
|---|---:|---:|---:|
| 1 stream × 256 | 256 | 5.197 s | 147.78 |
| 2 streams × 128 | 256 | 5.069 s | 151.50 |
| 4 streams × 64 | 256 | 5.261 s | 145.98 |
| 8 streams × 32 | 256 | 5.188 s | 148.03 |
| 2 streams × 256 | 512 | 10.504 s | 146.23 |

所有多流场景均收敛，最大最终残差不超过 `2.03e-11`，最大电压分量误差不超过 `1.48e-12`。首次 2×128 比 1×256 高 2.5%，但交替复测得到 148.85/s 与 148.80/s，优势消失；2×256 也没有产生 refactor 重叠收益。因此当前实现的多流能力已验证正确，但尚不能提高稳定吞吐，不能用首次波动宣称多流加速。下一步应记录并发下的 J、factor、solve 和 host 同步分阶段时间，定位真正的串行点。

## 附：已完成的图形基线

GravityMark 1.72，Direct3D 12，2560 × 1440，Temporal AA，200,000 个星体，关闭垂直同步：

- Score：20,611
- 平均帧率：123.4 FPS
- 有效测试时长：167.0 秒

光线追踪项目未继续执行，没有有效成绩。

## 复现

在本目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_cuda_benchmark.ps1
```

为了公平比较 5070，建议保留 `cuda_benchmark.py` 和 `python-packages` 的版本不变，并在另一张卡成为当前 CUDA 设备后原样运行。新结果应另存一份，避免覆盖这份 2070 SUPER 档案。

## 文件

- `results/cuda-results.json`：完整 CUDA 原始数据与逐轮结果。
- `results/cuda-benchmark.log`：正式测试控制台日志。
- `results/environment.json`：硬件、驱动和异常状态。
- `results/gravitymark-d3d12-1440p-fullscreen.log`：图形基线日志。
- `results/gravitymark-d3d12-1440p-frametimes.csv`：图形测试逐帧时间。
