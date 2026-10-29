# PEGASE 9241 Jacobian：RTX 2070 SUPER cuDSS 求解档案

测试日期：2026-10-29
状态：求解完成，残差验证和独立 SuperLU 交叉核对均通过

## 求解对象

本测试不是随机生成的 9241×9241 矩阵，而是 `pandapower.networks.case9241pegase` 在平坦启动下，第一轮 Newton 交流潮流的真实 Jacobian 与同状态失配向量。

| 项目 | 数值 |
|---|---:|
| 母线数 | 9,241 |
| PV 节点数 | 1,444 |
| PQ 节点数 | 7,796 |
| Jacobian 阶数 | 17,036 × 17,036 |
| 存储非零元 | 129,412 |
| 密度 | 0.004459% |
| 显式零值 | 2,912（保留原结构） |
| 右端项无穷范数 | 532.5249 |
| LU 非零元 | 246,972 |

求解方程为 `J × dx = -F`，其中 `F = [Re(mis[PV]); Re(mis[PQ]); Im(mis[PQ])]`，`mis = V × conj(Ybus × V) - Sbus`。

## 环境与方法

- GPU：NVIDIA GeForce RTX 2070 SUPER，8 GB。
- cuDSS：1.0.0；CUDA Runtime：12.9；CUDA Driver API：13.3。
- CuPy：14.2.0；矩阵值和右端项均为 float64。
- CSR 零基索引，行偏移与列索引均为 int32。
- 通用非对称矩阵、默认重排序、默认 LU 分解和自动主元策略。
- 计时方式：`cudssExecute` 调用加 CUDA Stream 同步的墙钟时间。
- 主机到显存传输单独计时，不计入分析、分解或回代。
- 一次符号分析后执行 1 次数值分解和 6 次同结构重分解，最后重复回代 20 次。

## 性能结果

| 阶段 | 中位耗时 | 最佳耗时 | 说明 |
|---|---:|---:|---|
| 主机到显存传输 | — | 0.887 ms | 单次测量 |
| 符号分析 | — | 68.505 ms | 单次测量，可复用 |
| 首次数值分解 | — | 2.589 ms | 单次测量 |
| 同结构重分解 | 1.708 ms | 1.681 ms | 6 次 |
| 回代求解 | 0.364 ms | 0.361 ms | 20 次 |

汇总：

- 首次求解各阶段之和：**72.479 ms**，包含传输、分析、首次分解和第一轮回代。
- 稳态“重分解＋回代”中位耗时：**2.072 ms**，不包含 Jacobian 在主机端的组装及 H2D 更新。
- 仅复用已有 LU 因子进行一次回代：中位 **0.364 ms**。

这些数字只覆盖本次线性系统求解，不代表完整 Newton 潮流已经收敛，也不包含 Jacobian 数值更新、主机与 GPU 间每轮数据同步或 ECS 其他开销。

## 数值验证

| 检查 | 结果 |
|---|---:|
| cuDSS 设备状态 | 0，正常 |
| `‖Jx-b‖∞` | 2.2975×10⁻¹¹ |
| 相对残差 `‖Jx-b‖∞ / ‖b‖∞` | 4.3143×10⁻¹⁴ |
| 范数型后向误差 | 5.6866×10⁻¹⁷ |
| 与 SciPy SuperLU 解的最大绝对差 | 2.2080×10⁻¹² |
| 与 SuperLU 解的相对无穷范数差 | 3.2681×10⁻¹³ |

`pivot_count = 0` 在 cuDSS 1.0 中表示受扰 1×1 主元计数为零，不应解读为算法没有执行任何主元选择。

## 文件与复现

- `export_pegase9241_jacobian.py`：重新导出第一轮 Newton 的 Jacobian 和右端项。
- `results/pegase9241-newton1-jacobian.npz`：真实 CSR 矩阵、失配向量及节点状态。
- `cudss_jacobian_benchmark.py`：cuDSS C API 自检、分解、求解、计时与残差验证。
- `results/cudss-pegase9241-results.json`：完整逐轮结果。
- `results/cudss-crosscheck.json`：独立 SuperLU 交叉核对。
- `results/cudss-pegase9241-dx.npy`：cuDSS 求得的 Newton 修正量。

在本目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_cudss_jacobian_benchmark.ps1
```

换用 5070 时，应保留矩阵文件、cuDSS 版本和测试脚本不变，并另存结果，避免覆盖 2070 SUPER 的档案。
