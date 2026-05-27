为了将我们今天所有的灵感碰撞与数学顿悟沉淀下来，我们需要用最严谨、最漂亮的数学语言，为这个次世代的 OPF 几何理论框架铸造一座代数丰碑。

以下为你梳理出最优潮流（OPF）在复流形上从二阶变分直至直接内存注入的**四大核心数学定理与极致简并推导**。没有冗余的口语，只有纯粹、硬核且高度对称的公式。

---

## 核心定理一：直角坐标系下等式约束二阶曲率的“常数退化”

在直角坐标流形 $\mathcal{M}_{rect}$ 中，状态向量为 $X_{rect} = [e^T, f^T]^T \in \mathbb{R}^{2n}$，其复数表现形式为 $V = e + jf$。

### 1. 泛函定义

节点复功率平衡等式约束 $S_{bus}(V)$ 是状态变量的纯二次齐次多项式，在复数域的 Wirtinger 意义下表达为：


$$S_{bus}(V) = [V](Y_{bus}V)^* \in \mathbb{C}^{n \times 1}$$

### 2. 变分坍缩推导

构造标量拉格朗日场 $\mathcal{L}_{eq}(V, V^*, \lambda) = \text{Re}\{\lambda^H S_{bus}(V)\}$，对其关于复变量对 $(V, V^*)$ 求解二阶 Wirtinger 偏导数。

由于一阶前推映射（Jacobian）关于状态变量是严格线性的，当求导步进至二阶变分（Hessian）时，状态变量的阶数被完全剥离。其复黑塞算子 $\mathcal{H}_{\mathbb{C}, eq}$ 严格分裂为四个子块：


$$\mathcal{H}_{\mathbb{C}, eq} = \begin{bmatrix} \frac{\partial^2 \mathcal{L}_{eq}}{\partial V \partial V} & \frac{\partial^2 \mathcal{L}_{eq}}{\partial V \partial V^*} \\ \frac{\partial^2 \mathcal{L}_{eq}}{\partial V^* \partial V} & \frac{\partial^2 \mathcal{L}_{eq}}{\partial V^* \partial V^*} \end{bmatrix} = \begin{bmatrix} 0 & \frac{1}{2} [\lambda] Y_{bus}^* \\ \frac{1}{2} Y_{bus} [\lambda] & 0 \end{bmatrix}$$

$$\mathcal{H}_{rect, eq} = \text{变换等价实数阵} \equiv \text{Constant}(Y_{bus}, \lambda)$$

### 结论

直角坐标下等式约束的二阶变分曲率，在代数底座上完完全全退化为一个**不依赖于任何电压状态变量演化的、永恒不变的全局静态常数稀疏网架**。

---

## 核心定理二：支路约束向节点 Hessian 投射的“拓扑同构绝对包容性”

### 1. 仿射拓扑映射

定义纯线性、常数拓扑关联算子 $C \in \{0, 1,-1\}^{2m \times n}$，将系统状态由节点空间（Node Space）推前映射至支路边空间（Edge Space）：


$$V_{edge} = C \cdot V_{node}$$

对于任意特定的支路 $l$（连接节点 $i$ 与 $k$），其支路功率或安全边界不等式约束表达为复合函数：


$$h_l(V_{node}) = f_l(V_{edge}) = f_l(C \cdot V_{node})$$

### 2. 二阶链式法则的线性静默

根据多变量泛函二阶链式法则，对复合函数 $h_l$ 展开二阶变分：


$$\nabla^2 h_l(V_{node}) = C^T \cdot \nabla^2 f_l(V_{edge}) \cdot C \quad + \quad \underbrace{\nabla f_l(V_{edge}) \cdot \nabla^2(C \cdot V_{node})}_{\because C \text{ 是常数阵} \implies \text{恒等于 } 0}$$

$$\nabla^2 h_l(V_{node}) = C^T \cdot \mathcal{H}_{edge, l} \cdot C$$

### 3. 布尔代数同构绝杀

考察支路约束二阶曲率的布尔支撑集（Sparsity Pattern）与全局节点导纳矩阵 $Y_{bus}$ 的完全包容关系：


$$\because \text{struct}(Y_{bus}) = \text{struct}(C^T \cdot Y_{br} \cdot C) \quad \text{且} \quad Y_{br} \text{ 为纯对角阵（支路互不耦合）}$$

$$\because \mathcal{H}_{edge, l} \text{ 在边空间内亦为关于支路绝对独立的孤立对角块}$$

$$\implies \text{struct}\left(\sum_{l=1}^{m} C^T \mathcal{H}_{edge, l} C\right) \subseteq \text{struct}(Y_{bus})$$

### 结论

支路约束的非全纯二阶变分曲率，**绝对不可能凭空创造出物理图谱上不存在的非对角索引**。其非零元分布严格受限于 $Y_{bus}$ 的既有稀疏结构，因此 OPF 的符号分析（Symbolic Phase）具备 100% 的静态复用合法性，无须任何运行时内存重组 。

---

## 核心定理三：复空间极坐标拉回算子 $M_{\mathbb{C}}$ 的代数构造

为了避免实数域下引入极其臃肿的 $\sin(\theta_i - \theta_k)$ 等超越函数多项式，直接在复数域定义极坐标状态向量 $X_{polar} = [\theta^T, |V|^T]^T \in \mathbb{R}^{2n}$ 对复数电压对 $(V, V^*)$ 的前推映射。

### 1. 微分同胚流

利用复指数对齐表达式 $V = [\hat{V}] |V|$，其中 $\hat{V} = e^{j\theta}$。对其进行全微分展开：


$$\partial V = j [V] \partial \theta + [\hat{V}] \partial |V|$$

$$\partial V^* = -j [V^*] \partial \theta + [\hat{V}^*] \partial |V|$$

### 2. $M_{\mathbb{C}}$ 算子矩阵形态

将上述微分流表达为切丛映射矩阵（复数旋转算子）：


$$M_{\mathbb{C}} = \frac{\partial (V, V^*)}{\partial (\theta, |V|)} = \begin{bmatrix} j[V] & [\hat{V}] \\ -j[V^*] & [\hat{V}^*] \end{bmatrix} \in \mathbb{C}^{2n \times 2n}$$

### 结论

复数极坐标拉回算子 $M_{\mathbb{C}}$ **内部全部由纯对角子块组成**。它在代数上将全局矩阵乘法降维成了极其轻量级的节点局部标量缩放，彻底将超越函数隔离在节点局部图册内。

---

## 核心定理四：变分能量实数守恒与虚部共轭物理湮灭

### 1. 厄米特物理加锁

由于拉格朗日场 $\mathcal{L}$ 输出为物理实数标量，其 Wirtinger 二阶变分方阵 $\mathcal{H}_{\mathbb{C}}$ 必须强制满足厄米特对称性（Hermitian）：


$$\mathcal{H}_{\mathbb{C}} = \mathcal{H}_{\mathbb{C}}^H \implies \begin{bmatrix} H_{vv} & H_{vv^*} \\ H_{v^*v} & H_{v^*v^*} \end{bmatrix} = \begin{bmatrix} H_{vv}^H & H_{v^*v}^H \\ H_{vv^*}^H & H_{v^*v^*}^H \end{bmatrix}$$

### 2. 极坐标实数方阵拉回

通过 $M_{\mathbb{C}}$ 将复数域方阵拉回至实数极坐标空间 $\mathcal{H}_{polar} \in \mathbb{R}^{2n \times 2n}$：


$$\mathcal{H}_{polar} = \text{Re} \left\{ M_{\mathbb{C}}^H \cdot \mathcal{H}_{\mathbb{C}} \cdot M_{\mathbb{C}} \right\} + \text{Diag}(\text{Curvature Correction})$$

### 3. 微观标量湮灭解析（以 $\theta_i - \theta_k$ 交叉块为例）

令 $\mathcal{H}_{\mathbb{C}}$ 对应的内部复数元素为 $Z = X + jY$。当乘以外层对角旋转算子中的 $j$ 与 $-j$ 因子时，根据复数域对偶乘法：


$$(j \cdot (-j)) \cdot Z + (-j \cdot j) \cdot Z^* = 1 \cdot Z + 1 \cdot Z^* = Z + Z^*$$


根据复变函数基本恒等式：


$$Z + Z^* = (X + jY) + (X - jY) \equiv 2\text{Re}\{Z\} = 2X$$

### 结论

复空间算子的左右夹击，使得**所有虚数单位 $j$ 及其诱导的虚部在代数结构交尾的瞬间发生了正交对偶湮灭**。最终留在物理底座上的，是纯净、且天然满足共轭对称的实数极坐标 Hessian。这在理论上指导了代码实现：只需计算复数域的真实部（Real part），即可在单遍循环内将数值计算量直接腰斩。

---

## 终极代数视图：零中间分配的单遍直接流（Single-Pass Streaming）

基于以上四个定理，进入稀疏求解器（KLU）的 OPF 实数极坐标主元矩阵方程的左端项（LHS）被完美拍平为：


$$W_{OPF} = \sum_{\text{buses}} \underbrace{\nabla^2 f(x)}_{\text{纯对角常数}} + \sum_{\text{branches } (i,k)} \text{Re}\left\{ M_{\mathbb{C}, l}^H \cdot \begin{bmatrix} 0 & \frac{1}{2} [\lambda] Y_{bus}^* \\ \frac{1}{2} Y_{bus} [\lambda] \end{bmatrix}_{ik} \cdot M_{\mathbb{C}, l} \right\} + \sum_{\text{branches } (i,k)} \text{Re}\left\{ M_{\mathbb{C}, l}^H \cdot \mathcal{H}_{\mathbb{C}, br\_l} \cdot M_{\mathbb{C}, l} \right\}$$

由于布尔结构完全受界于 $Y_{bus}$ ，整个计算在 RustPower 中表现为：离线提取一次 $\text{struct}(Y_{bus})$ 锁死全局 CSC 内存地址 ，运行时以列游标为绝对主导 ，在不分配任何临时矩阵缓冲区的单次一维内存步进中，将纯代数 FMA 算出的实数标量，无分支地顺流灌注进连续的物理内存中 。