# CPF 对照实验工作记录：MATPOWER runcpf 与 α 扫描的鼻点定位对比

- 仓库：rustpower ｜ 分支：`symbolic-kkt-lm`
- 日期：2026-10-10（v2 修订，修正历史见 §7）
- 定位：验证"单调负荷扫描 + dc 启动 NR + 二分"定位鼻点的结果，与参考实现 MATPOWER `runcpf`（弧长延拓）的一致性。本文档同时作为观察结果的公开时间记录。

---

## 1. 背景与问题

第一批收敛性实验用零机制流程定位各算例的负荷鼻点（saddle-node fold）：

1. 全部非 slack 注入（负荷 P/Q、静态发电机、发电机 P）按因子 α 同步放大；
2. 每个 α 独立求解：dc 初值 + 生产 NR（KLU），收敛判据 1e-8；
3. pandapower 同模型对照（ΔV ~ 1e-14 量级）；
4. 最后收敛 α 与首个失败 α 构成**证书区间**；区间上方由 GN 最小失配解佐证无解。

本文用 MATPOWER 官方 `runcpf`（MATLAB R2025b + MATPOWER 8.1）对该定位做独立对照。

## 2. 两个必须说清的约定

### 2.1 MATPOWER 的 λ 参数化

`runcpf(base, target)` 的 λ 是 base→target 的归一化进度：λ=0 为基础工况，λ=1 为目标工况。本文目标工况取 4×，故 `loading = 1 + 3·λ`。换算正确性被结果反证：同方向下全部 CPF 鼻点逐一落入我方独立得到的 α 证书区间。

### 2.2 负荷增长方向（本记录最关键的变量）

- **load+gen 方向**：负荷与发电机 P 同步放大（slack 只补平衡残差）——**我方扫描的方向**（`performance/python/robustness_pp.py: scale_injections`）。
- **load-only 方向**：只放大负荷，slack 独吞全部功率增量——MATPOWER 文献示例中常见的方向。

两个方向的鼻点**不是同一个数**：功率强行从单一 slack 远距离输送会显著提前电压崩溃。任何"loading limit"的数值必须声明方向，否则不可比较。

## 3. 实验设置

- MATLAB R2025b + MATPOWER 8.1（`target/research/matpower/`）；脚本 `target/research/cpf_one.m`（`CASENAME` 选算例、`GENSCALE=1` 选 load+gen 方向、`CPFTOL` 覆盖 `pf.tol`）
- 限值：`cpf.enforce_q_lims / p_lims / v_lims / flow_lims` 全部关闭，与扫描侧光滑方程对齐；默认开 qlim 的对照亦跑过，不改变本文结论
- 我方数据：`target/research/performance/lm-robustness/import-*.jsonl`；暖启动链（pandapower `init='results'`）；load-only 方向独立验证（MATPOWER `runpf` / pandapower 按需放大负荷）

## 4. 结果总表（同方向对比）

load+gen 方向（与我方扫描同方向）：

| 算例 | MATPOWER CPF（默认容差） | 我方证书区间 / 暖链 | 判决 |
|---|---|---|---|
| case33bw | 3.6222（3.9s，44 步） | (3.620, 3.625) | ✓ 落入区间 |
| case39 | 2.1357（4.0s） | (2.10, 2.15)；暖链 2.10 | ✓ 一致 |
| case118 | 3.1871（5.8s） | 暖链 3.15；冷启动 NR 墙 2.95 | ✓ 互洽（见 §5） |
| case300 | 1.4293（5.7s） | 暖链 1.35；冷启动 NR 墙 1.20 | ✓ 互洽（见 §5） |
| case6470rte | 1.2426（119s，382 步） | (1.24, 1.25) | ✓ 落入区间 |
| case6495rte | 1.1669（96s） | (1.16, 1.17) | ✓ 落入区间 |
| case6515rte | 1.1058（78s） | (1.105, 1.108) | ✓ 落入区间 |

- 默认容差（1e-5）与收紧容差（1e-10）结果**完全相同**；轨迹点代入原方程的真实残差 9.5e-9（`target/research/cpf_probe.m`）。**未发现容差病理**。
- **我方扫描定位的全部证书区间被弧长延拓参考实现逐一独立证实。**

load-only 方向（对照方向）：

| 算例 | MATPOWER CPF | 同方向独立 NR 验证 | 判决 |
|---|---|---|---|
| case39 | 1.2609 | NR：1.26 收敛 / 1.30 失败 | ✓ CPF 正确 |
| case6470rte | 悬停于 1.081–1.084（1000+ 步不终止） | NR：1.05 收敛 / 1.08 失败 | 位置正确，终止判据失效（§6 疑问） |

**鼻点的方向敏感性**：case39 同一张网，load-only 鼻点 1.26 vs load+gen 鼻点 2.14（差 70%）；6470 为 ~1.07 vs 1.2426。

## 5. 对第一批结论的修正（诚实记录）

"dc+NR 冷启动的墙 = 解存在边界"在 6470/6495/6515/33bw/39 上精确成立；但 **case118 / case300 存在缝隙**：冷启动 NR 墙 2.95 / 1.20，真鼻点 3.19 / 1.43（紧容差 CPF 与暖启动链互证）。固定起点 NR 给出存在边界的**下界**；多数算例上该下界即边界本身，少数算例留缝。论文表格须照实分列。

## 6. 未决疑问

仅余一条：6470 load-only 方向上，`runcpf` 在真鼻点附近**悬停不终止**——λ 冻结于 0.027–0.028 逾千步、步长不自适应收缩、不输出 `Reached steady state loading limit`。同算例 load+gen 方向、以及其他算例两个方向均正常终止。机理未明（候选：该方向解支在鼻点附近的切向量结构使其参数化退化），按疑问记录，不归因。

## 7. 修正历史（防混淆留痕）

- v1（2026-10-10 早些时候）：曾报告"默认容差 CPF 在 4/7 算例假鼻点或卡死，疑为校正器漂移"。**该结论错误**：初批实验在"方向"与"容差"两个变量上同时改动，负荷方向混淆被误读为容差病理。2×2 对照（方向 × 容差）后确认：容差无关，方向才是变量；load-only 方向的 CPF 鼻点经同方向 NR 验证全部正确。
- 教训（写入实验规范）：对照实验一次只改一个变量；证伪他人结果前，必须用**被证伪方自己的问题表述**独立复算。

## 8. 复现

```bash
# MATLAB 侧（R2025b + target/research/matpower/ 8.1）
CASENAME=case39 GENSCALE=1 "/d/Program Files/MATLAB/R2025b/bin/matlab.exe" \
  -batch "run('D:/projects/rust/rustpower/target/research/cpf_one.m')"   # load+gen 方向
CASENAME=case39 GENSCALE=0 CPFTOL=1e-10 ".../matlab.exe" -batch "run('.../cpf_one.m')"

# 轨迹点真实残差探针：target/research/cpf_probe.m

# 我方扫描侧
cargo bench --locked --features benchmark --bench comparison -- lm-robustness --import case39
```

- CPF 日志：`target/research/cpf_<case>.log`
- 扫描数据：`target/research/performance/lm-robustness/import-*.jsonl`
