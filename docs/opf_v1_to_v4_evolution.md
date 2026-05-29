# RustPower 最优潮流 (OPF) 引擎演进教材：从第一性原理到极限性能

本教材详细记录了 RustPower 的 OPF 计算核心从 V1 到 V4 的数学重构与工程优化过程。我们将从最基础的数学公式出发，一步步揭示我们是如何将工业级大电网（如 IEEE 118、PEGASE 9241）的计算时间压榨到极致，实现对传统软件（如 Pandapower）的 50 倍以上超越的。

---

## 1. 基础问题：最优潮流 (OPF) 的数学模型

最优潮流（Optimal Power Flow, OPF）的核心是在满足电网物理定律和设备安全限制的前提下，寻找发电成本最低的运行状态。

它的标准数学形式是一个典型的非线性规划（NLP）问题：

$$
\begin{aligned}
\min_{x} \quad & f(x) \\
\text{s.t.} \quad & g(x) = 0 \quad (\text{节点功率平衡}) \\
& h(x) \le 0 \quad (\text{支路潮流限制及变量越限})
\end{aligned}
$$

状态变量 $x = [\boldsymbol{\theta}, \mathbf{V_m}, \mathbf{P_g}, \mathbf{Q_g}]^T$，即节点的电压相角、电压幅值，以及发电机的有功和无功出力。

### 1.1 内点法 (Interior Point Method) 与 KKT 系统
为了求解这个带不等式约束的 NLP 问题，我们引入松弛变量 $z \ge 0$ 将不等式转化为等式 $h(x) + z = 0$，并构造拉格朗日函数：

$$
\mathcal{L}(x, \lambda, \mu, z) = f(x) + \lambda^T g(x) + \mu^T (h(x) + z) - \gamma \sum \ln(z_i)
$$

在牛顿-拉夫逊迭代（Newton-Raphson）中，我们需要在每一步求解庞大的 KKT 线性方程组，求出搜索方向 $\Delta x, \Delta \lambda, \dots$：

$$
\begin{bmatrix}
\mathbf{L_{xx}} + dh^T \cdot \text{diag}\left(\frac{\mu}{z}\right) \cdot dh & dg \\
dg^T & 0
\end{bmatrix}
\begin{bmatrix} \Delta x \\ \Delta \lambda \end{bmatrix} = \text{RHS}
$$

这里的核心，也是整个 OPF 计算最耗时的三大“铁块”：
1.  **雅可比矩阵 (Jacobian)**：$dg = \nabla g(x)$ 和 $dh = \nabla h(x)$。
2.  **海塞矩阵 (Hessian)**：$\mathbf{L_{xx}} = \nabla^2 \mathcal{L}_{xx}$，即拉格朗日函数对状态变量 $x$ 的二阶导数矩阵。
3.  **线性求解 (Linear Solve)**：对上述巨型稀疏矩阵的求逆/分解。

---

## 2. 演进之路：如何干掉耗时的大铁块？

### V1 时代：极坐标与矩阵的奴隶 (Legacy Path)
早期的实现（也是绝大多数开源软件的做法）直接在极坐标下对功率方程进行二次求导，并使用稀疏矩阵运算来拼接结果。

*   **节点功率 Hessian**：依赖复杂的中间复数矩阵 $F, D, E$，通过多步稀疏矩阵相乘（SpGEMM）得出。
*   **支路潮流 Hessian**：基于 $\frac{\partial^2 |S|^2}{\partial x^2}$，同样需要构建 $d2Sbr\_dV2$ 矩阵。
*   **性能特征 (IEEE 118)**：Hessian 组装耗时 **~9 ms**，KKT 拼接 **~6 ms**，KLU 求解 **~4 ms**（因为频繁重建 Solver）。不仅慢，且极坐标二次求导在长长的推导链条中极易埋下难以察觉的对角线数学 Bug。

### V3 时代：标量展开的初次尝试
我们意识到稀疏矩阵乘法（SpGEMM）中包含了大量的内存分配和拓扑搜索（0 分配的死敌）。于是 V3 将极坐标下复杂的矩阵公式在 `for` 循环中完全展开为标量（Scalar）计算。
*   **改进**：消除了中间矩阵分配，利用 CSC 格式直接定位元素。
*   **局限**：循环内部充斥着 $e^{j\theta}$、复数乘法和繁琐的极坐标链式求导公式。本质上，V3 只是优化了内存，并没有在数学层面实现降维。

### V4 革命：直角坐标常数内核 + 极坐标投影 (The Breakthrough)
V4 彻底抛弃了在极坐标泥潭中的挣扎，回到了物理方程的最底层（First Principles）：直角坐标。

**1. 降维打击：常数 Hessian ($H_{rect}$)**
在直角坐标 $V = e + jf$ 下，节点的复功率 $S_i = V_i I_i^*$ 是一个严格的**二次型方程**。这意味着，它对电压的二阶导数是一个**常数**！
定义复数权重矩阵 $W_{ij} = \lambda_i Y_{ij}^*$，则拉格朗日函数在直角坐标下的二阶导数矩阵块极其简单：
$$ H_{R, ij} = \begin{bmatrix} \operatorname{Re}(W_{ij} + W_{ji}) & \operatorname{Im}(W_{ji} - W_{ij}) \\ \operatorname{Im}(W_{ij} - W_{ji}) & \operatorname{Re}(W_{ij} + W_{ji}) \end{bmatrix} $$
*惊人的发现*：在内点法的每次迭代中，$\lambda$ 是已知的参量。因此，**这个 $H_{rect}$ 根本不需要电压信息就能算出来！**

**2. 极坐标投影 (Pull-back)**
求解器依然在极坐标 $x = [\theta, |V|]$ 下工作，怎么办？我们只需要用坐标变换的雅可比 $\mathcal{M}$ 进行极速投影（旋转）：
$$ H_{polar, ij} = \mathcal{M}_i^T H_{R, ij} \mathcal{M}_j $$
$\mathcal{M}_i$ 是一个极简的 2x2 矩阵 $\begin{bmatrix} -f_i & \cos\theta_i \\ e_i & \sin\theta_i \end{bmatrix}$。在代码中，这退化为了 8 次标量乘加（FMA），彻底秒杀了 V3 那长篇大论的三角函数推导。

**3. 对角线曲率修正 (Projection Leftovers)**
坐标变换（非线性）会产生额外的曲率，这被称为“投影残差”。但这仅仅表现为对角线上的一个修正项：$\Delta_{diag} = \nabla_{V_{rect}} \mathcal{L} \cdot V_{polar}$。我们只需在每列组装结束时减去这个标量残差即可，**无需任何全局矩阵修正**。

**4. 终结 SpGEMM：合并支路惩罚项 (Merged Slacks)**
KKT 矩阵中最丑陋的一项是 $dh^T \cdot \text{diag}\left(\frac{\mu}{z}\right) \cdot dh$。
V4 的天才之处在于：既然支路极限约束 $h$ 只涉及该支路两端的 4 个电压变量，那么它对 KKT 系统的贡献就只是一个 $4 \times 4$ 的秩-1 更新（Rank-1 Update）。
我们在遍历支路计算 $H_{branch}$ 时，**直接在标量层面计算 $w (\nabla h)(\nabla h)^T$ 并加到原有的 $4 \times 4$ 块中**。
*结果*：PIPS 内部的巨型稀疏矩阵相乘（`spgemm`）被彻底从物理世界中抹除。

---

## 3. 工程视角：CSC 内存魔法与零分配

拥有了纯粹的数学公式，还需要极致的代码实现。

### CSC 连续写入 (Column-Major Assembly)
稀疏矩阵（CSC）按列存储。V4 的外层循环设计为 `for j in 0..nb`（遍历列）。
我们通过 `unsafe` 直接获取目标内存切片：
```rust
let out_aa = unsafe { std::slice::from_raw_parts_mut(lxx_ptr.add(cache.lxx_cp[j]), nnz_j) };
```
算出 $H_{polar}$ 后，通过 `out_aa[offset] = val` 顺序推入。
**没有二分查找，没有越界检查，内存连续写入。** 缓存命中率拉满。

### 持久化线性求解器 (Persistent KLU)
原本的 PIPS 每一轮都要重新执行 `KLUSolver::default()`。
现在的架构在整个 OPF 启动前执行一次 Symbolic Factorization（画地图）。在 14 趟牛顿迭代中，求解器只进行极速的数值回刷（Numeric Refactor）。

---

## 4. 性能决战：各阶段耗时盘点 (基于 IEEE 118)

| 阶段 / 耗时 (ms) | V1 (旧基准) | V4 (当前终极版) | 提速分析 |
| :--- | :--- | :--- | :--- |
| **Hessian 组装** | ~ 9.41 ms | **~ 0.61 ms** | **15 倍速！** 拜常数内核投影与标量化融合所赐，此阶段已达计算机物理极限。|
| **KKT 矩阵拼接** | ~ 6.44 ms | **~ 4.24 ms** | **提速 34%**：通过标量层面合并松弛变量惩罚项，物理上消灭了原本极其昂贵的 `spgemm`（稀疏矩阵乘法）。目前的 4.2ms 纯粹是把 $L_{xx}$ 和 Jacobian 拼装打包的内存分配/拷贝耗时。 |
| **线性求解 (KLU)**| ~ 4.25 ms | **~ 3.83 ms** | 稳固提升。得益于持久化 KLU 消除重复符号分解。 |
| **总计 14 次迭代** | **~ 25.9 ms** | **~ 14.2 ms** | **相比 Python 版 Pandapower (681ms) 提速近 48 倍！** |

*(注：在更大规模的 PEGASE 9241 上，V4 使得 Hessian 组装耗时从 1.83s 暴跌至 149ms，单项加速 12 倍。)*

---

## 5. V5 蓝图：Symbolic KKT 与 100% 连续推流

V4 已经是一份“足以写论文”的答卷，它证明了直角坐标内核的数学威力。但性能剖析暴露了下一个瓶颈：**牛顿循环内最大的单项耗时不再是 Hessian（0.61ms），而是 KKT 矩阵拼接（4.24ms）**。V5 的唯一目标，就是把这 4.24ms 打到亚毫秒。

### 5.1 现状诊断：scatter 查表为什么慢

当前管线是「先成矩阵、再散列」：
1. `v4_rect_numeric_fill` 生成 $L_{xx}$（一个独立 CSC）。
2. `opf_consfcn` 生成雅可比 $dg$（另一个独立 CSC）。
3. `KKTSymbolicCache::fill_kkt` 用三张预存的映射表 `lxx_to_kkt[]`、`dg_to_kkt[]`、`dgt_to_kkt[]`，把上面两个矩阵的每个 nnz **逐个散列**进大 KKT 的 `values` 里：
   ```rust
   for (idx, &v) in lxx.values().iter().enumerate() { kkt_vals[self.lxx_to_kkt[idx]] = v; }
   ```

`kkt_vals[随机下标] = v` 是**乱序写**。映射表本身是连续读，但写入目标在 KKT 一维数组里到处跳——每次写都可能踩一个新的 cache line。再叠加 $L_{xx}$、$dg$ 两个中间矩阵的分配与拷贝，4.24ms 主要就耗在这里。**任何指针不连续的缓存，本质上性能都很差。**

### 5.2 核心定理：KKT 的结构 = $Y_{bus}$ 结构的确定性函数

要做到「列内顺序写」，必须先证明：整张 KKT 的稀疏结构在符号阶段就能从 $Y_{bus}$ 完全确定，无需任何运行期搜索。

记状态 $x=[\boldsymbol\theta(nb),\,\mathbf V_m(nb),\,\mathbf P_g(ng),\,\mathbf Q_g(ng)]$，$nx=2nb+2ng$；等式约束 $g=[\mathbf P(nb),\,\mathbf Q(nb)]$ 外加线性等式（参考母线角度固定等），$neq=2nb+neq_{lin}$。KKT 维度 $D=nx+neq$，结构为

$$
K=\begin{bmatrix} M & dg \\ dg^T & 0 \end{bmatrix},\qquad M = L_{xx} + \text{(merged-slack 对角)}.
$$

逐类列分析其非零行模式（$N(j)$ 表示 $Y_{bus}$ 第 $j$ 列的结构邻居，对工业网 $Y_{bus}$ 结构对称，行邻居 = 列邻居）：

| KKT 列 | 来自 $M$ 的行 | 来自 $dg^T$ 的耦合行 | 每列 nnz |
| :-- | :-- | :-- | :-- |
| $\theta_j$ (列 $j$) | $\{k\}$ (Haa), $\{nb{+}k\}$ (Hva)，$k\in N(j)$ | $\{nx{+}i\}$ (P), $\{nx{+}nb{+}i\}$ (Q)，$i\in N(j)$ | $4\,\deg(j)$ |
| $V_{m,j}$ (列 $nb{+}j$) | $\{k\}$ (Hav), $\{nb{+}k\}$ (Hvv) | $\{nx{+}i\}$, $\{nx{+}nb{+}i\}$ | $4\,\deg(j)$ |
| $P_{g}$ (列 $2nb{+}g$) | 对角 $\{2nb{+}g\}$ (成本) | $\{nx + \text{bus}(g)\}$ | 2 |
| $Q_{g}$ (列 $2nb{+}ng{+}g$) | 对角（结构占位） | $\{nx+nb+\text{bus}(g)\}$ | 2 |
| P 约束 (列 $nx{+}i$) | — | $\{k,\,nb{+}k:k\in N(i)\}\cup\{2nb{+}g:\text{bus}(g)=i\}$ | $2\deg(i)+|G_i|$ |
| Q 约束 (列 $nx{+}nb{+}i$) | — | $\{k,\,nb{+}k\}\cup\{2nb{+}ng{+}g:\text{bus}(g)=i\}$ | $2\deg(i)+|G_i|$ |

**关键结论**：每一个电压列恰好由它的 $Y_{bus}$ 邻居表 $N(j)$ 派生出 $4\deg(j)$ 个元素（Haa/Hva/dgP/dgQ 四块，全部跑同一张邻居表）；发电机列只有 2 个元素；约束列由邻居表 + 该母线的发电机集合派生。**整张 KKT 的 `col_offsets` 与 `row_indices` 都是 $Y_{bus}$ 结构 + `gen→bus` 映射的纯函数**，可在符号阶段一次性直接构造（直接计数得 col_offsets、直接按升序发射得 row_indices），连 V4 当前的 COO→排序→去重都省掉。

> 这正是 §5 标题里「拓扑同构」的严格版本：电压变量在 Hessian 中的邻居行，与它在 Jacobian 耦合中的约束行，是**同一张 $N(j)$** 经过平移 $(+nx,\,+nx{+}nb)$ 得到的。

### 5.3 列内生产序 = CSC 行升序（可流式写入的证明）

CSC 要求每列内部按行号升序存储。要做到「算一个写一个、指针只增不减」，必须保证我们**生产数值的顺序**恰好等于行升序。逐列验证：

* **电压列 $\theta_j$**：电压块行号 $\{k\}\cup\{nb{+}k\}$ 全部 $<2nb$；约束块行号全部 $\ge nx$；二者之间（发电机区 $[2nb,nx)$）在电压列里**结构为空**。又因 $k_{\max}\le nb{-}1<nb\le nb{+}k_{\min}$，所有 Haa 行严格先于所有 Hva 行。于是该列的 CSC 行序天然为
  $$[\,k:k\in N(j)\,]\;\Vert\;[\,nb{+}k\,]\;\Vert\;[\,nx{+}i\,]\;\Vert\;[\,nx{+}nb{+}i\,]$$
  即**对同一张已升序的邻居表做 4 次顺扫**，分别吐出 Haa、Hva、dgP、dgQ 的数值——四段连续 run，指针一路递增。
* **发电机列**：`[对角项, 单个耦合项]`，两次写。
* **约束列**：`[Va run | Vm run | gen run]`，三次顺扫。

因此整张 KKT 可以用**一个单调推进的写指针 `ptr`** 完成填充，全程 `kkt_vals[ptr]=val; ptr+=1`，**零随机寻址、零查表**。

### 5.4 Symbolic Permutation：让发电机与约束连续排列

上面唯一的「不规则」是发电机：若发电机按插入顺序（ext_grid 在前、gen 在后）排列，则 `bus(g)` 是乱序的，约束列在收集「本母线的发电机」时需要一次 gather。

**V5 绝招（与 `new_pf` 的 `[PQ|PV|slack]` 重排同源）**：对发电机按其（重排后）母线索引做预排序，使 `gen→bus` 单调。于是：
* 流式扫描母线 $j=0..nb$ 时，所需发电机用一个**游标顺序取用**，无随机查找；连接矩阵 $C_g$ 退化为准对角结构。
* 发电机的 KKT 列（$P_g/Q_g$）也随 `bus(g)` 单调，其 2 元素列按规律连续产出。

母线本身是否重排（即是否 permute $Y_{bus}$）是**正交**的选择：
* 对**装配流式**而言不是必需——每个电压列都是自洽的邻居表顺扫，与全局母线序无关。
* 但重排 $Y_{bus}$ 对**线性求解（KLU 的 fill-in / AMD 排序）**有益，且能与 `new_pf` 共享同一套内部母线序（`PFOrder.map`），保持 PF↔OPF 数据流一致。
* 因此建议：采用 `new_pf` 已经产出的母线排列作为 OPF 的内部序，发电机排序在此基础上叠加。

### 5.5 缓存索引设计（零查表）

符号阶段产出（全部为 `Vec`，一次性构造，跨牛顿迭代复用）：
1. `kkt_col_ptrs: Vec<usize>`、`kkt_row_indices: Vec<usize>` —— 预分配的 KKT 骨架，由 §5.2 直接构造。
2. `gen_perm: Vec<usize>` / `gen_cursor_by_bus` —— §5.4 的发电机排列与按母线游标。
3. （可选）每列各 run 的起始偏移；但更彻底的做法是**让数值内核与符号构造共用同一套循环结构**，使写指针 `ptr` 的推进与符号发射顺序逐位一致——这样连「起始偏移表」都不需要存。

注意：这套设计**没有** `lxx_to_kkt` / `dg_to_kkt` 这类「源 nnz → 目标位置」的散列表。数值内核不再「填两个矩阵再搬运」，而是**直接在 KKT 的连续内存上就地生成**。

### 5.6 数值内核草图

```text
ptr = 0
for j in bus_order:               // §5.4 的内部母线序
    // —— θ_j 列 ——
    for k in N(j): kkt_vals[ptr]=Haa(j,k); ptr+=1     // Haa run
    for k in N(j): kkt_vals[ptr]=Hva(j,k); ptr+=1     // Hva run
    for i in N(j): kkt_vals[ptr]=dgP_dθ(i,j); ptr+=1  // dgP run
    for i in N(j): kkt_vals[ptr]=dgQ_dθ(i,j); ptr+=1  // dgQ run
    // —— V_{m,j} 列 同理（Hav/Hvv/dgP_dV/dgQ_dV）——
    ...
// —— 发电机列（已按 bus 单调）——
for g in gen_order:
    kkt_vals[ptr]=cost_hess(g); ptr+=1
    kkt_vals[ptr]=-1.0;         ptr+=1   // dP/dPg 耦合
// —— 约束列（P_i / Q_i），沿 N(i) + 本母线发电机游标顺扫 ——
...
```

数值由 V4 的直角常数内核 $H_{rect}$ 投影 + merged-slack 秩-1 更新提供；本节只改变「写到哪、以什么顺序写」，不改变「算什么」。因此 V5 = V4 的数学 × 100% Cache 友好的内存管线。

### 5.7 ECS 落点（与新数据流对齐）

* 符号阶段是一个 **OPF 插件的 setup system**：读 `NetworkOperators.ybus` 结构 + 发电机实体的 `TargetBus`，产出上面的 KKT 骨架与排列，存为 Resource（如 `KKTSymbolicV5`）。仅当拓扑 `Changed` 时重跑。
* 数值阶段是牛顿循环内的 system：消费骨架 Resource + 各实体的导纳/乘子，就地写 KKT `values`。
* 支路实体、发电机实体只是「无感」地提供底层参数；计算引擎看到的是一条 100% 连续的高速数据管线。

**预期收益**：KKT 拼接 4.24ms → 亚毫秒；PEGASE 9241 万节点级 KKT 组装坍塌至数毫秒，实现全流程的「计算即组装」。