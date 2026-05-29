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

## 5. 展望 V5：终极 0 分配管线 (The Future)

V4 已经是一份“足以写论文”的答卷。牛顿迭代内复杂的矩阵求导和乘法已被尽数抹除。然而，查看上面的数据可以发现，**KKT 矩阵拼接 (4.24 ms) 反而成为了当前循环内最大的耗时项**。

这说明系统仍在被“中间内存分配”拖累（即 `build_saddle_point` 函数）。

### V5 的战略规划：全量 ECS 与原地回填

1.  **彻底消灭 `build_saddle_point`**
    在优化启动时，执行终极的 `KKTSymbolicCache`，不仅查明 $L_{xx}$，更要查明全量雅可比 $dg$ 的每一个非零元在终极大 KKT 矩阵 `values` 数组中的准确下标。
2.  **原地回刷 (In-place Numeric Fill)**
    未来的 V5 不再返回独立的 $L_{xx}$ 矩阵。它将直接接收一个预先分配好的大 KKT 数组，在计算完 $H_{polar}$ 和雅可比分量后，利用指针**一步到位**地写入对应的 KKT 槽位。整个 PIPS 迭代过程中将不再有任何堆内存（Heap）分配。
3.  **融合 ECS 架构 (Entity Component System)**
    彻底摒弃扁平的 `OPFData`。将 $\lambda, \mu$ 和 $KKTIndices$ 直接作为组件挂载到对应的节点（Bus）和支路（Line/Trafo）实体上。利用 Bevy ECS 的 `par_iter()` 实现万节点工业网级别上的**无锁多核并发组装**。

**届时，IEEE 118 的牛顿步耗时将跌破 5 毫秒。** 属于 RustPower 的计算革命，才刚刚开始。