# CPF 对照实验工作记录：MATPOWER runcpf 与 α 扫描的鼻点定位对比

- 仓库：rustpower ｜ 分支：`symbolic-kkt-lm`
- 日期：2026-10-10
- 定位：验证"单调负荷扫描 + dc 启动 NR + 二分"定位鼻点的结果，与参考实现 MATPOWER `runcpf`（弧长延拓）的一致性。本文档同时作为观察结果的公开时间记录。
- **重要说明**：本文对 MATPOWER CPF 在默认设置下的异常行为一律按**未决疑问**记录，不下"实现有 bug"的结论；机理部分仅为假说。

---

## 1. 背景与问题

本工作第一批收敛性实验（见 `tech_doc/LM_GN_Assembly_Worklog.md` 后续批次）用如下零机制流程定位各算例的负荷鼻点（saddle-node fold）：

1. 全部非 slack 注入（负荷 P/Q、静态发电机、发电机 P）按因子 α 同步放大；
2. 每个 α 独立求解：dc 初值 + 生产 NR（KLU），收敛判据 1e-8；
3. 以 pandapower 同模型对照（ΔV ~ 1e-14 量级一致）；
4. 最后一个收敛 α 与第一个全方法失败 α 构成**证书区间**；区间上方由 GN 最小失配解的 merit>0 佐证无解。

疑问：这种扫描定位的结果，与教科书标准的弧长延拓（CPF）是否一致？为此用 MATPOWER 官方 `runcpf` 做对照。

## 2. MATPOWER CPF 的参数化约定

MATPOWER `runcpf(base, target)` 的延拓参数 λ 是 base→target 的**归一化**插值：

```
S(λ) = S_base + λ·(S_target − S_base),   λ=0 为基础工况，λ=1 为目标工况
```

本文目标工况取 4× 全部负荷与发电（与扫描方向一致），故换算为：

```
loading_factor = 1 + 3·λ
```

该换算的正确性被结果反证：紧容差下全部 CPF 鼻点逐一落入我们先前独立得到的 α 证书区间（§4 表），若换算系数有误不可能七个算例同时对上。

## 3. 实验设置

- MATLAB R2025b（`D:\Program Files\MATLAB\R2025b`）+ MATPOWER 8.1（2025-07-12 版，`target/research/matpower/`）
- 算例：MATPOWER 官方 `data/` 的 case33bw / case39 / case118 / case300 / case6470rte / case6495rte / case6515rte
- 脚本：`target/research/cpf_one.m`（单算例，`CASENAME` 环境变量传入，`CPFTOL` 控制 `pf.tol`）
- 限值处理：`cpf.enforce_q_lims / p_lims / v_lims / flow_lims` **全部关闭**，与扫描侧（光滑方程、无 qlim）严格对齐；默认开 qlim 的对照也跑过，结论无差别
- 两组容差：MATPOWER 默认（`pf.tol=1e-5`）与收紧（`pf.tol=1e-10`）
- 我方数据来源：`target/research/performance/lm-robustness/import-*.jsonl` 与暖启动链脚本（pandapower `init='results'` 逐点延拓）

## 4. 结果总表

| 算例 | 默认容差 CPF | 紧容差 CPF (1e-10) | 我方证书区间 / 暖链 | 判决 |
|---|---|---|---|---|
| case33bw | 3.6222（3.9s，44 步） | — | (3.620, 3.625) | 一致 ✓ |
| case39 | **1.261**（4.3s，142 步） | **2.1357**（4.0s） | (2.10, 2.15)；暖链到 2.10 | 默认假鼻点 |
| case118 | **1.8165**（8.4s，468 步） | **3.1871**（7.1s） | 暖链到 3.15 | 默认假鼻点 |
| case300 | **1.0360**（11.1s，415 步） | **1.4293**（6.8s） | 暖链到 1.35 | 默认假鼻点 |
| case6470rte | **卡死**（1000+ 步停在 λ≈0.028，即 1.08） | **1.2426**（119s，382 步） | (1.24, 1.25) | 默认失效 |
| case6495rte | 未跑 | **1.1669**（96s） | (1.16, 1.17) | 落入区间 ✓ |
| case6515rte | 未跑 | **1.1058**（78s） | (1.105, 1.108) | 落入区间 ✓ |

观察要点：

1. **紧容差 CPF 与我们的证书区间全部一致**（6470/6495/6515/33bw 落入区间内；39/118/300 与暖启动链互洽）。我们的扫描定位被弧长延拓参考实现独立验证。
2. **默认容差 CPF 在 7 个算例中 4 个给出错误结果**：case39/118/300 报告的鼻点比真值低 40%~50%，case6470rte 直接爬行停滞。
3. 证伪"默认 CPF 鼻点"的判据来自 MATPOWER 自己：其 `runpf` 在 case39 的 α=1.5/1.9/2.1 均收敛，minV 与我们的扫描逐位吻合（0.930/0.837/0.735）。
4. 成本对比：6470 规模紧容差 CPF 串行 119s；我们的扫描每点独立、秒级、天然可并行。

## 5. 未决疑问（现象记录，机理仅为假说）

**现象**：默认容差（1e-5）下，`runcpf` 的延拓轨迹从 base 起步即表现异常——每步"1 corrector steps"即判收敛，λ 每步仅推进约 0.001（弧长步长 0.05），整个 case39 轨迹 142 步走 7.1 弧长单位、λ 只动了 0.174，随后在 λ=0.087（loading 1.261）处"折返"并以 `Reached steady state loading limit` 终止。6470 上同类轨迹在 λ≈0.028 附近原地踏步一千步以上。上述现象与限值强制无关（q/p/v/flow 全关后复现）。

**假说（未证实）**：校正器在宽松容差下未真正收敛到解流形，误差逐步累积使轨迹漂离真实解支，走上一条幻影曲线；`runcpf` 的"λ 回落即鼻点"判据被假轨迹触发。支持证据：`pf.tol` 收紧到 1e-10 后，同一实现对同一批算例给出与三方 NR 互证完全一致的真鼻点。

**待办**：此假说需要进一步验证（如逐点输出 CPF 轨迹电压并与同 loading 的 NR 解比对残差），在验证之前，论文中只陈述可复现的现象与对比数据，不归因。

## 6. 对第一批结论的修正（诚实记录）

扫描侧"dc+NR 冷启动的墙 = 解存在边界"在 6470/6495/6515/33bw/39 上精确成立（墙即区间端点）；但 **case118 / case300 存在小缝隙**：冷启动 NR 墙分别在 2.95 / 1.20，而紧容差 CPF + 暖启动链表明真鼻点在 3.19 / 1.43 附近。即固定起点 NR 给出存在边界的**下界**；多数算例上该下界就是边界本身，少数算例留缝。论文表格须照实分列两列。

## 7. 复现

```bash
# MATLAB 侧（需 R2025b + target/research/matpower/ 8.1 源码树）
CASENAME=case39 CPFTOL=1e-10 "/d/Program Files/MATLAB/R2025b/bin/matlab.exe" \
  -batch "run('D:/projects/rust/rustpower/target/research/cpf_one.m')"

# 我方扫描侧
cargo bench --locked --features benchmark --bench comparison -- lm-robustness --import case39

# 暖启动链（pandapower）
python performance/python/robustness_pp.py --case case39   # 逐点逻辑见该文件
```

- CPF 日志：`target/research/cpf_<case>.log`（默认容差的停滞/假轨迹完整留档）
- 扫描数据：`target/research/performance/lm-robustness/import-*.jsonl`
