# 小系统普通 ACPF 串行数值成本估计

使用现有 `newton_pf`＋NewtonCache＋KLU，选取真实单支路退出样本，从共享收敛基态 AC 电压启动。CPU 绑定 0–3，容差 1e-10、迭代上限 30。生产求解与调度代码保持原有实现。

这是样本推算的串行数值成本估计，完整串行 N−1 路径未在此测量；数值也不构成严格数学下界。

| 母线 | 实际 PF 数 | 实际总 NR 迭代 | 普通 ACPF 摊销 μs/迭代 | 估计数值成本 μs | 样本成本跨度推算 μs | 已测单 worker 流水线 μs | 流水线/估计 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 19 | 63 | 0.776 | 48.893 | 48.715–49.187 | 242.404 | 4.958× |
| 30 | 38 | 119 | 1.962 | 233.438 | 222.887–241.530 | 530.148 | 2.271× |

## 计算方法与计时范围

每个系统选择按支路目录排列的首个、中间、末个连通退出场景，共三个样本；每个样本先完整求解一次建立缓存，再重复普通 ACPF 调用 2001 次取中位数。每次都从相同的共享基态 AC 电压重新启动，确认迭代数非零且稳定。测量外还用 N1Worker 对同一退出场景求解并验证电压最大差小于 1e-7。

设样本调用中位耗时为 tᵢ，实际迭代数为 kᵢ；使用 c=Σtᵢ/Σkᵢ，估计整个故障集合的串行数值成本为 c×K，其中 K 使用已保存的完整单 worker 批次实际迭代总数。样本跨度使用 min(tᵢ/kᵢ)×K 到 max(tᵢ/kᵢ)×K，表示样本差异。

计时从普通 newton_pf 进入到返回，包括初值复制、初始失配、Jacobian 数值填充、KLU refactor/solve、电压与失配更新、返回电压的构造。J 模式、KLU Symbolic/Numeric 在重复间复用，首次构造/分解另行保存。矩阵复制、场景 patch、基态准备、线程、队列、worker App 和后处理在此计时外。

流水线对照为此前 N1Study::run 的 1 worker、batch=8、StatusOnly、空约束表，包含实体解析、解列筛选、worker/App/工作区创建、逐场景 patch、求解、队列、收集和 join；14/30 分别重复整批 201/101 次。

两者差额同时包含线程与调度固定成本、Numeric 冷启动、跨场景缓存条件、代表样本选择和内核实现差异。估算用于判断串行计算量的量级，差额并非线程开销的独立测量。

## 样本

| 母线 | 退出设备 | NR 迭代 | 缓存调用中位 μs | 首次调用含模式/符号/factor μs |
| ---: | --- | ---: | ---: | ---: |
| 14 | Line 0 | 5 | 3.873 | 17.267 |
| 14 | Line 9 | 4 | 3.123 | 16.563 |
| 14 | Transformer 4 | 4 | 3.093 | 13.199 |
| 30 | Line 0 | 3 | 5.619 | 31.929 |
| 30 | Line 21 | 3 | 6.089 | 23.446 |
| 30 | Line 40 | 3 | 5.947 | 21.705 |

## 复现

测量入口 `examples/bench_acpf_floor.rs`，输入及命令在任务目录 `n1-acpf-estimate/metadata.json`，全部 2001 次调用样本及结果在 case14.json / case30.json。示例仅用于样本计时，执行器保持现有结构。

```sh
CARGO_TARGET_DIR=/home/cts/workspace/rustpower-amd-supervariable/target cargo build --offline --release --features experimental_n1 --example bench_acpf_floor
taskset -c 0-3 /home/cts/workspace/rustpower-amd-supervariable/target/release/examples/bench_acpf_floor /home/cts/Documents/Codex/2026-10-26/task/n1-scalability/cases/case14.zip /home/cts/Documents/Codex/2026-10-26/task/n1-scalability/case14-workers1/result.json /tmp/acpf-estimate14.json
```
