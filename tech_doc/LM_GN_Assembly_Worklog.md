# 符号化 KKT/LM 组装工作记录：GN-LM 五条路径的原理、效果与结论

- 仓库：rustpower ｜ 分支：`symbolic-kkt-lm`
- 工作区间：2026-07-29 — 2026-09-02（commit 时间线见 §7）
- 定位：潮流/OPF 最小二乘（LM）框架下，**增广 KKT 系统的符号结构一次推导、数值阶段纯偏移直填**的完整实现与五条组装路径消融对照。本文档同时作为该想法的公开时间记录。

---

## 1. 数学原理

### 1.1 潮流方程的最小二乘形式

极坐标潮流失配向量（保留全部行列、按 `[PQ | PV | slack]` 预排序后切出活跃部分）：

```
r(x) = S_calc(v) − S_spec ,  x = [θ_active ; |V|_PQ] ,  n = n_PV + 2·n_PQ
```

求解 `min ½‖r(x)‖²`。Gauss–Newton 步用法方程：

```
(JᵀJ + μI)·δ = −Jᵀr
```

### 1.2 增广 KKT 形式（本工作的求解对象）

法方程不显式形成 JᵀJ，而是解等价的对称不定**增广系统**：

```
[ μI   Jᵀ ] [ δ ]   [  0 ]
[ J   −I  ] [ λ ] = [ −r ]
```

该矩阵是 quasi-definite（μI ≻ 0，−I ≺ 0），因此**无主元 LDLᵀ 分解稳定可用**（Cholesky 不行：矩阵不定）。对称性意味着 LDLᵀ 类后端只读上三角——这是 §2.4 半矩阵布局的依据。

### 1.3 μ 自适应（Nielsen 增益比）

每次外迭代计算增益比 `ρ = (f − f_new) / pred`，`pred = −½·gᵀδ`，`g = Jᵀr`：

- `ρ > 1e-4`：接受试步；若 `ρ > 0.75` 则 `μ ← max(μ/3, 1e-12)`；
- 否则拒绝，`μ ← 2μ`；解出现非有限值则 `μ ← 10μ`；`μ > 1e12` 判定停滞。

实现见 `src/lm/gn_flat.rs`（`GnDriver::solve_gn`）与 `src/lm/gn_triu.rs`（`GnTriuDriver::solve_gn`），两条路径的 μ 规则逐字节一致。

### 1.4 极坐标试步

`θ ← θ + δθ`（全部活跃节点），`|V| ← |V| + δv`（仅 PQ 节点），再转回复数电压。无功越界/病态场景下由 μ 机制兜底，失败时得到最小二乘意义下的最近解——这是 LM 相对 NR 的卖点。

---

## 2. 核心思想：符号一次，偏移直算

### 2.1 Ybus 预排序与一次切列

Ybus 以 CSC 存储且节点已按 `[PQ | PV | slack]` 预排序、列内行号升序。因此对每列只需一次 `partition_point` 即可切出 PQ 段 / 活跃段 / slack 段的边界（`pq_ends` / `active_ends`），对角元位置 `diag_ptrs`（及其列内 rank `diag_off`）一次二分得到。全部缓存于 `src/lm/cache.rs`（`YbusAnalysisCache`），与生产 PF 的 `JacobianPattern2`（`src/basic/new_dsdvbus2.rs`）同一原理。

### 2.2 KKT 符号结构 = Ybus 偏移算术

增广系统的四个块（μI、Jᵀ、J、−I）的 CSC 结构**全部**由 Ybus 列偏移直接推出：每一列在循环开头用 base 地址加段长现算自己的写入起点，无 slot map、无逐支路表、无运行时查找。符号结构见 `src/lm/pattern.rs`（`KktPattern`）、块抽象 `src/lm/block.rs`（`BlockDesc`）、全局三元组 `src/lm/flat.rs`（`FlatLayout`）。

数值填充复用生产级 v4 Jacobian kernel（`src/basic/new_dsdvbus4.rs::fill_jacobian_v4`）与转置 kernel（`src/lm/kernels.rs::fill_jt`）。

### 2.3 模式对称 vs 值不对称（相移器教训）

Ybus 的**值**在带相移变压器时不对称（复变比），因此"拿列 i 当行 i 用"是错的——AUG-FS 早期版本在此栽过（commit `0d82060` 修复）。但 Ybus 的**模式**恒对称，故行向遍历可以原样复用全部列向缓存：行 k 中对角元的 rank 等于 `diag_off[k]`，分段切点同一组 `pq_ends` / `active_ends`。这是 §3 中 TRIU 路径成立的根基，也是本文档记录的关键观察之一。

### 2.4 半矩阵布局与后端配对

LDLᵀ 后端只读上三角 ⇒ 只存上三角：`nnz_triu = nnz + 2n`，对比全对称 slim 的 `2·nnz + 2n`（IEEE39 实测 1024 → 579；大规模系统 nnz ≫ n 时接近减半）。

但**不是所有 LDLᵀ 都能吃半矩阵**，这是实验抓出来的坑：

- **QDLDL**（纯 Rust，clarabel 移植）：输入约定就是原始序的上三角，triu 输入原样消费 ✓
- **SuiteSparse LDL**：读的是**置换后** PAP′ 的上三角，AMD 置换事先未知，因此必须给全对称 CSC——喂 triu 会在置换后丢元素，静默算错（被 `gn_plugin_ieee39_standard_case` 当场抓获，见 commit `75af009`）
- **KLU 等 LU 后端**：需要完整矩阵 ✗

配对逻辑集中在 `src/lm/mod.rs` 的 `newton_pf_gn_default` feature 阶梯，插件侧无感知（`src/basic/ecs/gn_plugin.rs`）。

---

## 3. 五条实现路径

| 路径 | 组装原理 | 符号复用 | 实现文件 |
|---|---|---|---|
| **NE-COO** | COO→CSC 组 J，每迭代 spgemm 重算 JᵀJ 模式+数值 | 每外迭代重来 | `src/lm/normal_eq.rs`（`dumb_mode`） |
| **AUG-FS** | 全量 2n_bus J（含 slack/PV 浪费象限）→ 切片 → COO 堆叠 | 每次 μ 尝试全新 | `src/lm/baseline/full_slice.rs` |
| **AUG-COO** | COO push + 排序转换整个 `[μI Jᵀ; J −I]` | 每次 μ 尝试全新 | `src/lm/baseline/aug_coo.rs` |
| **AUG-SDF** | v4 kernel 直填 J、Jᵀ 块 + 列拷贝进 slim CSC | 一次，之后纯数值 | `src/lm/gn_flat.rs` |
| **AUG-SDF-TRIU** | 行向直填上三角：沿 Ybusᵀ 走，J 行 r 直接写入 s-列 r，写全顺序 | 一次，之后纯数值 | `src/lm/gn_triu.rs` |

命名约定：NE = 法方程，AUG = 增广，FS = full-slice（最蠢基线），COO = 坐标格式重组，SDF = symbolic direct fill（本文方法），TRIU = 半矩阵变体。基线路径刻意白嫖 v4 数值 kernel——即让对照组借用我们的公式实现，只保留组装策略差异；这使测得的差距是**组装策略的下界**，对基线是保守（偏袒）的。

TRIU 相对 SDF 消去的正是实测的两个大头（PEGASE9241 每次迭代）：`fill_jt` 转置散射写 ~0.42ms 与 slim 列拷贝 ~0.38ms。散射写的随机 RFO（读使拥有）流量是物理税，原地填充治标不治本，故改为布局层面消灭——`fill_jt_rows` 单趟完成 Jᵀ，无 J 块、无转置、无拷贝。

---

## 4. 实验结果

### 4.1 正确性（全部自动化测试，复现见 §5）

1. **逐位相等**：TRIU 行向直填的 Jᵀ 段与 `fill_jacobian_v4` + `fill_jt` 输出 `to_bits()` 逐位相等（IEEE39）。单公式求值、无求和顺序差异，必须严格相等——同时验证了"模式对称 ⇒ 行向对角 rank = `diag_off`"这一假设。
2. **收敛一致**：IEEE39 两家均 4 次迭代；PEGASE9241 均 11 次；TRIU(QDLDL) vs SDF(KLU) 最终解 max|ΔV| = 6.2e-15。
3. **三家同解**：严格容差 1e-12 下 NR / GN-LM / exact-LM 解逐点一致（`three_way_matches_nr_tight`）。
4. **交叉验证**：AUG-FS 独立书写的教科书式全量 J 切片后与生产 v4 kernel 的简约 J 一致（IEEE39 与 PEGASE9241，`aug_fs_j_matches_v4*`）。
5. **插件端到端**：QDLDL→triu 与 SuiteSparse LDL→flat 两种配置下 IEEE39 均收敛。

### 4.2 性能（release，QDLDL 后端，五路径同 μ 策略同后端）

| 系统 | 路径 | 迭代 | 墙钟 | 组装/迭代 | 求解/次 |
|---|---|---|---|---|---|
| IEEE39 | AUG-SDF | 4 | 292 µs | 4.3 µs | 60.5 µs |
| (nb=39) | **AUG-SDF-TRIU** | 4 | **174 µs** | **2.2 µs** | 36.5 µs |
| | AUG-FS | 4 | 577 µs | 9.7 µs + 切片 30.0 µs/try | 95.8 µs |
| | AUG-COO | 4 | 483 µs | 21.4 µs/try | 90.4 µs |
| | NE-COO | 4 | 561 µs | spgemm 50.6 µs | — |
| IEEE118 | AUG-SDF | 5 | 566 µs | 8.4 µs | 93.9 µs |
| (nb=118) | **AUG-SDF-TRIU** | 5 | **499 µs** | **5.9 µs** | 83.8 µs |
| | AUG-FS | 5 | 2.41 ms | 26.2 µs + 111.1 µs/try | 284.8 µs |
| | AUG-COO | 5 | 2.14 ms | 93.4 µs/try | 281.9 µs |
| | NE-COO | 5 | 2.74 ms | spgemm 193.5 µs | — |
| PEGASE9241 | AUG-SDF | 11 | 202 ms | 1468 µs | 15.6 ms |
| (nb=9241) | **AUG-SDF-TRIU** | 11 | **174–212 ms** | **805–974 µs（−40%）** | 13.9–17.0 ms |
| | AUG-FS | 11 | 969 ms | 5081 µs + 21.8 ms/try | 56.5 ms |
| | AUG-COO | 11 | 837 ms | 17.2 ms/try | 54.2 ms |
| | NE-COO | 11 | 1.08 s | spgemm 34.2 ms | — |

（PEGASE9241 行给了两次独立运行的范围；其余为单次运行。）

### 4.3 结论

1. **符号直填 vs 一切 COO/spgemm 路径：组装快 12–42 倍**（9241 上 0.8–1.5ms vs 17–34ms），且符号只做一次。差距随规模扩大。
2. **半矩阵 + 行向直填再省 40% 组装**，并把存储近乎减半；小系统上组装占比高，总墙钟 −30~40%。
3. **大系统的瓶颈在求解器**：9241 每次迭代 QDLDL 求解 ~14ms，占墙钟九成以上。组装侧已压到同量级下接近内存带宽极限的水平，继续提速须换求解器路径（GPU LDLᵀ 等）。
4. μ 重试只改对角 μ 槽（δ-列唯一元素），**不触发重新组装**——符号一次架构在 LM 内循环的收益。

---

## 5. 验证与复现

```bash
# 全量回归（33 项）
cargo test --release --features klu lm::

# 五路径消融对照表（§4.2 数据源）
cargo test --release --features klu lm_ablation -- --nocapture

# TRIU 逐位对照 + IEEE39/PEGASE9241 收敛
cargo test --release --features klu lm::gn_triu -- --nocapture

# 插件路径（默认 qdldl→triu；加 ldl feature 则 →flat）
cargo test --release --features klu gn_plugin -- --nocapture
cargo test --release --features "klu,ldl" gn_plugin -- --nocapture
```

---

## 6. 文件地图

| 内容 | 文件 |
|---|---|
| Ybus 分析缓存（切点/对角/转置映射） | `src/lm/cache.rs` |
| KKT 四块符号结构 | `src/lm/pattern.rs`、`src/lm/block.rs`、`src/lm/flat.rs` |
| Jᵀ/H/μ 数值 kernel | `src/lm/kernels.rs` |
| 残差与测试夹具 | `src/lm/residual.rs` |
| AUG-SDF（全对称 slim）驱动+插件入口 | `src/lm/gn_flat.rs` |
| AUG-SDF-TRIU（半矩阵行向）驱动+插件入口 | `src/lm/gn_triu.rs` |
| 基线：NE-COO / AUG-COO / AUG-FS | `src/lm/normal_eq.rs`、`src/lm/baseline/aug_coo.rs`、`src/lm/baseline/full_slice.rs` |
| 五路径消融 bench | `src/lm/baseline/bench.rs` |
| 后端选择阶梯（LDL/QDLDL/KLU） | `src/basic/solver.rs`、`src/lm/mod.rs` |
| ECS 插件接线 | `src/basic/ecs/gn_plugin.rs` |
| 完整 Hessian 的 exact-LM（保留，非主线） | `src/lm/exact/` |
| 生产 v4 Jacobian kernel（数值公式来源） | `src/basic/new_dsdvbus4.rs` |
| 架构原始设计文档 | `Symbolic_KKT_LM_Architecture.md` |

## 7. 时间线（本分支 commit）

| 时间 | commit | 内容 |
|---|---|---|
| 2026-07-29 | `84d9ab9` | KKT 数值 kernel 对齐生产 v3 Jacobian |
| 2026-07-30 | `f029ada` | exact-LM 驱动（完整 Hessian） |
| 2026-09-01 | `6a077d2` | `src/lm` 顶层模块落定 |
| 2026-09-01 | `9d959a5` | NR/GN/exact 多方对照 + 可解性探针 |
| 2026-09-01 | `0ec1f12` | LDLᵀ 后端接入求解路径（ActiveSolver） |
| 2026-09-01 | `eae826c` | 后端对比、法方程消融、app 级性能 |
| 2026-09-01 | `f9e7c3b` | NE-COO / AUG-COO 基线 |
| 2026-09-01 | `2ff5adf` | AUG-FS 基线 + v4 交叉验证 |
| 2026-09-02 | `0d82060` | AUG-FS 相移器修复（§2.3 教训） |
| 2026-09-02 | `13517e3` | AUG-SDF-TRIU 行向半矩阵直填 |
| 2026-09-02 | `75af009` | 插件按后端选布局；SuiteSparse LDL 三角约定坑（§2.4） |

## 8. 后续（未做，有意挂起）

- 病态潮流算例（Tripathy 1982，DOI 10.1109/TPAS.1982.317050，11/13/43 节点）——LM 在无解区给出最小二乘解的卖点实验。
- KLU wrapper 自检（指针复用约定的运行时断言）。
- exact/ 完整 Hessian 路径保留不动；当前主线为 GN-LM。
- OPF 侧：同一套符号直填思想向全保留 KKT（`src/new_opf/`）的迁移，另文记录。
