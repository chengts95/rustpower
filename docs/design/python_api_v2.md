# Python API v2 设计：命令网关驱动的电网模型

状态：**已定稿**（2026-06-09）
English Guide: [docs/python_api.md](../python_api.md)
原则：设计模式 + 自顶而下。先冻结 Python API 契约，再实现，最后把命令网关回移（backport）到 Rust 原生路径。

## 0. 写入的双模式（定稿补充）

写路径只有两条，全部最终经过命令网关：

- **立即模式**：元素代理的属性赋值（`load.p_mw = 30`）= 一条立即执行的命令
  （单命令自动提交事务）。实现上经正常 `Mut<T>` 写入，change tick 顺带被维护，
  但 **tick 不承担正确性职责**——脏信息由命令的 DirtyClass 显式给出；我们自己的
  系统不依赖 `Changed<T>` 做正确性判断，tick 维护与否无关紧要。
- **延迟模式**：`grid.edit()` 的指令队列。事务提交时一次 apply，吃 Harvard 架构
  指令队列 **fused insert** 的优势：实体的全部组件一次性落位，避免逐组件插入
  造成的 archetype 迁移与碎片。

**Reset / 重灌路径是一等公民**：`load_network()` / `from_pandapower` 的语义是
"清空世界实体 → 重新 ingestion → 全量重建"，必须有契约测试保证重复加载等价于
全新构造（不残留实体、不叠加注入）。pandapower 数据模型在 ingestion 结束即丢弃。

---

## 1. 顶层视角：三种用户场景决定 API 形状

API 不按"我们有什么系统"组织，按"用户要干什么"组织。

### 场景 A：批量求解（pandapower 迁移用户）

```python
import rustpower as rp

grid = rp.PowerGrid("cases/IEEE118/data.zip")
grid.solve()
print(grid.res_bus)          # DataFrame，镜像 pandapower 的 net.res_bus 心智模型
```

### 场景 B：参数循环（时序 / Monte Carlo / 优化内环）

```python
load = grid.load(bus=5)
for p in profile:
    load.p_mw = p            # 属性赋值 = 一条命令，自动提交
    grid.solve()             # 自动走轻量路径（只刷新 s_bus，热启动）
    record(grid.v)
```

循环体内**零仪式感**：没有 init_pf，没有 update()，没有"记得调 XX 否则结果是旧的"。

### 场景 C：拓扑研究（N-1、扩建规划）

```python
with grid.edit() as e:       # 事务边界
    b, _ = e.add_bus(110.0)
    e.add_line(b5, b, 12.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
grid.solve()                 # 事务提交时已标记 Topology 脏 → 自动全量重建

grid.undo()                  # 整个事务一键回滚（命令带逆操作）
grid.solve()                 # 回到改动前的网络
```

**核心承诺：`solve()` 永远正确。** 用户不需要知道也不可能知道"哪种修改需要哪种重建"——这个知识全部内化到命令的脏分类里。

---

## 2. 顶层 API 契约（冻结对象）

### 2.1 PowerGrid（Facade）

```python
class PowerGrid:
    # ---- 构造 ----
    def __init__(self, case_path: str | None = None, *,
                 f_hz: float = 50.0, sn_mva: float = 100.0): ...
    @classmethod
    def from_pandapower(cls, net) -> "PowerGrid": ...
    # 加载即转换：pandapower 数据模型在 ingestion 后丢弃，唯一事实是 ECS 组件

    # ---- 元素访问（查询，无影子索引）----
    def bus(self, id: int) -> Bus: ...
    def load(self, bus: int | None = None, name: str | None = None) -> Load: ...
    def gen(self, bus: int | None = None, name: str | None = None) -> Gen: ...
    def line(self, from_bus: int, to_bus: int) -> Line: ...
    def loads(self, bus: int | None = None) -> list[Load]: ...
    # 找不到 → raise KeyError（不返回 None；fail fast）

    # ---- 写入（唯一入口：事务编辑器）----
    def edit(self) -> GridEditor: ...        # with 块 = 一个事务 = 一次 apply
    def undo(self) -> None: ...              # 回滚上一个事务
    def redo(self) -> None: ...

    # ---- 求解 ----
    def solve(self, *, tol: float = 1e-8, max_it: int = 10) -> SolveReport: ...

    # ---- 结果（只读投影）----
    @property
    def res_bus(self) -> pd.DataFrame: ...   # vm_pu, va_degree, p_mw, q_mvar
    @property
    def res_line(self) -> pd.DataFrame: ...
    @property
    def v(self) -> np.ndarray: ...           # 复电压，原始母线序
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...

    # ---- 网络概览（只读）----
    @property
    def n_bus(self) -> int: ...
    @property
    def n_line(self) -> int: ...
    def describe(self) -> pd.DataFrame: ...  # 元素计数总表
```

**从公共 API 中删除**：`init_pf()`（内化为事件响应）、`load_network()`/`from_pp_net()`（并入构造函数与 `from_pandapower`）、`reset_state()`（内化）、`builder()`（被 `edit()` 取代）。
迁移期保留 deprecated 别名一个版本（`init_pf` → no-op + warning，`set_p` → 属性赋值）。

### 2.2 GridEditor（Unit of Work / 事务）

```python
class GridEditor:
    # 拓扑级命令（只在这里，PowerGrid 上不放 add_*）
    def add_bus(self, vn_kv, *, name=None, vm_min=0.9, vm_max=1.1, zone=0) -> tuple[int, Bus]
    def add_line(self, from_bus, to_bus, length_km, *, std_type=None, ...) -> Line
    def add_trafo(self, hv_bus, lv_bus, *, sn_mva, ...) -> Trafo
    def add_load(self, bus, p_mw, q_mvar, *, name=None) -> Load
    def add_gen(self, bus, p_mw, vm_pu=1.0, *, ...) -> Gen
    def add_ext_grid(self, bus, vm_pu=1.0, va_degree=0.0) -> ExtGrid
    def add_shunt(self, bus, q_mvar, *, p_mw=0.0, ...) -> Shunt
    def remove(self, element) -> None          # 任意 handle
    # __exit__ = commit（apply 命令队列 → 合并发出脏事件）
    # 异常退出 = abort（丢弃队列，世界无变化 —— 真正的事务语义）
```

事务即原子性：`with` 块内的命令要么全部生效（一次 apply、一次脏决策），
要么全部丢弃（块内抛异常）。这是 deferred buffer 从"性能优化"升格为"语义保证"。

### 2.3 元素代理（Proxy：live view，不持有数据）

```python
class Load:
    bus: int                  # 只读
    p_mw: float               # 读 = 查 ECS；写 = SetParam 命令（Injection 脏类）
    q_mvar: float
    in_service: bool          # 写 = Topology 脏类！
    name: str

class Gen:
    bus: int
    p_mw: float               # Injection
    vm_pu: float              # VoltageSetpoint
    in_service: bool          # Topology

class Bus:
    id: int
    vn_kv: float              # Admittance 脏类（影响所有关联支路标幺）
    # 结果（只读，solve 后有效）
    vm_pu: float
    va_degree: float
    # 导航
    loads: list[Load]
    gens: list[Gen]

class Line:
    from_bus: int; to_bus: int
    length_km / r_ohm_per_km / ...   # Admittance 脏类
    in_service: bool                  # Topology
    # 结果
    p_from_mw / q_from_mvar / i_ka / loading_percent
```

代理只持有 `(Entity, Py<PowerGrid>)`。**代理拿不到 `world_mut`**——它的写路径只有命令
队列。这是"构造上的纪律"：Rust 拦不住裸写，但 API 形状让正确的路是唯一的路。

### 2.4 SolveReport

```python
class SolveReport:
    converged: bool
    iterations: int
    runtime_ms: float
    rebuild: str       # "none" | "injection" | "voltage" | "admittance" | "full" —— 可观测性
    def __bool__(self): return self.converged
```

`solve()` 失败不抛异常（发散是合法结果），但**求解前验证**失败抛异常并给出可读信息：
无 slack、孤岛、参数非法（vn_kv≤0 等）。这是对 pandapower 神秘报错的差异化体验。

---

## 3. 分层架构（Rust 侧，2026-06-09 第二次修订后定稿）

> 修订记录：GridCmd 具象化命令对象 **取消**。GIL 下单线程、undo 出局（提交前
> 的撤销由 cmd buffer 的 pop/abort 覆盖）、重放无需求——Command 模式买的是
> "动作成为值"，我们的动作要么即时要么已躺在哈佛队列里，对象层是给静态已知
> 信息买的动态包装，反 DOD。"网关"保留为**模块边界**而非类型。

```
Python 层    PowerGrid(Facade) ── 元素 Proxy ── GridEditor(UnitOfWork)
                  │（管道完全内化，Python 不可见）
网关函数     powerflow::mutation::set_load_p / set_load_q / set_gen_p / set_gen_vm
               · 同步写 case（Target* 组件）——保证 read-your-writes
               · 符号约定唯一所有；no-op 过滤
               · 投递 ParamDiff 指令到 message bus（脏信息的唯一载体）
消息总线     ParamDiff::Injection{bus,dp,dq} / VoltageMag{bus,vm}
               （原生调用方可直接投递 = 状态级修改，case 不动）
消费系统     consume_param_diffs —— 普通并行系统（MessageReader+Query，
               非 exclusive，走调度器）：施加真正的 SBusInjPu/VBusPu diff，
               并在此处统一触发 SBusChangeEvent / VoltageChangeEvent
同步         structure_update：事件 → 全量 copy 组件 → mat（tick-free）
拓扑路径     HarvardCommandBuffer（fused insert）+ topology_dirty
               （Phase 3′ 改为 commit 写 FullRebuildEvent → run_schedule(PFInit)）
投影         PowerFlowMat = ECS 事实的物化视图
求解         ecs_run_pf → 结果回写 VBusPu（热启动，仅收敛时）
```

> 第三次修订（同日）：废除 DirtyBuses 资源 + exclusive 的 apply 函数——
> 在 `&mut World` 上做全部工作是错误路径（绕开调度器、不可组合）。脏信息
> 回归 message bus 唯一承载；消费者是普通系统；变更事件只在消费者一处发出。
> 全量重建（init_pf）时清空未消费的 ParamDiff（目标值已被重建吸收，再施加
> 即双重计数）。diff 语义（±增量）按用户决策采用：no-op 过滤 + 周期性全量
> 重建归零使浮点漂移远低于求解器容差（实测 2000 次编辑 ~3e-10，即容差本身）。

### Change tick 的最终地位

**零职责。** 不用它检测（检测需要 O(n) 全量扫描，而 setter 本来就 O(1) 知道
哪个实体变了——自有 DirtyBuses 严格更优）；不用它过滤（粗粒度路径全量 copy，
便宜且语义不依赖观察者 tick）；**绝不让它驱动 case 数据**。tick 仍被 Bevy
默认维护，仅作为未来调试/校验的旁路信息存在。

### Case 数据 vs 运行状态（双路径语义）

- **Case 编辑**（`load.p_mw = 30`）：经 mutation 管道，更新 Target* 组件
  （数据集本身）+ 重聚合派生状态。case 是会被序列化/导出的事实。
- **状态级 diff**（时序、OPF 内环）：直接写 `SBusInjPu` + 发粗粒度事件，
  **case 的 Target* 原封不动**。timeseries 模块已是这个形态；Python 侧如有
  需求可加 `bus.inject_diff(dp, dq)` 暴露同一路径。

### 与现有代码的对应关系

| 现有 | 去向 |
|---|---|
| `handles.rs` 中的增量算术（nudge_bus_injection 等） | **删除**。符号约定回归注入系统唯一所有 |
| `init_pf()` 的手抄管线 | 收编为 `PFInit` schedule，Topology 事件触发；原生 Startup 只跑它一次 |
| `GridBuilder` | 演化为 `GridEditor`（语义从"批量构建"升格为"事务"） |
| Python 包装层 `bus_to_elements`/`id_map` 影子索引 | **删除**，改为 ECS 查询（必要时 ECS Resource 索引由系统维护） |
| `python/solver.rs` NewtonSolver | 保留为 low-level 逃生舱，文档降级，不参与本架构 |
| `timeseries/state.rs` 手动发事件 | Phase 4 改为命令生产者（时序 = 带时间戳的命令计划表） |
| `FullRebuildEvent` 无写入者问题 | 根除：事件只由网关 apply() 发出，调用方无法遗忘 |

---

## 4. 实施阶段（每阶段独立可验证）

**Phase 0 — 本文档定稿**。冻结 §2 的 API 签名。

**Phase 1 — Python 外观先行**（不动 Rust 内核）：
新 API（属性、edit()、res_bus、KeyError 语义）实现在现有机制之上；
`verify_python_api.py` 重写为新 API 的契约测试（15 项 → 约 25 项，含事务回滚）。
旧方法保留为 deprecated 别名。
验收：契约测试全过 + compare_pp.py 数值不变。

**Phase 2″ — 统一突变管道**（✅ 完成 2026-06-09）：
`powerflow::mutation` 标准化指令集 + `DirtyBuses` 资源；删除 handles 全部
增量算术（散落的 nudge_bus_injection 收编）；`Changed<T>` 全面退出正确性
路径（sbus/vbus_pu_update 改为 tick-free 全量 copy）；按母线重聚合替代
增量 ±（精确、无漂移、符号约定唯一所有）。
验收：31 项契约测试全过（含 2000 次重复编辑无漂移）；pandapower 交叉 3e-9。

**Phase 3′ — PFInit schedule 统一**（✅ 完成 2026-06-09）：
`powerflow::pf_init` 定义可重跑的 `PFInit` schedule（ingestion → 清理投影 →
归零 → 元素 setup → 清标签重分类 → 注入(从0消费diff=sum) → init_states →
置换 → reset solvers），全量重建语义与增量路径统一为"归零与否的同一种累加"。
`FullRebuildEvent` 由 editor commit / in_service setter 投递，structure_update
消费后 `run_schedule(PFInit)`；`full_dirty` 与 `structure_dirty`（NodeType/qlim
路径）分离——全量重建会重新分类节点，绝不能吹掉 qlim 的 PV→PQ 降级。
`topology_dirty` 布尔旁路退役；solve() = 投事件(仅首次) + app.update() +
读 `LastStructureAction` 汇报；ecs_run_pf 对退化问题（无 slack/空网）插入
非收敛结果而非 panic。Python init_pf 手抄管线删除，改为同步跑同一 schedule。
（Admittance 级重建——线路参数 setter 暴露时一并实现。）

**Phase 4′ — Backport**：timeseries 等原生写入方迁移到 mutation 管道 /
粗粒度事件的统一契约；删除 Python 包装层 bus_to_elements 影子索引
（改为 ECS 查询或 Resource 索引）。
验收：cargo test 全过，原生与 Python 路径共享同一管线定义。

**已删除项**：undo/redo（提交前撤销由 cmd buffer abort/pop 覆盖，提交后
撤销无真实工作流对应）、GridCmd 具象化、命令日志重放。

---

## 5. 决策点（已定稿，2026-06-09）

| # | 决策 | 结论 | 备注 |
|---|---|---|---|
| D1 | 参数修改方式 | **属性赋值**（`load.p_mw = 30`） | 立即模式命令；赋值时按 handle 类型 emit 配套指令 |
| D2 | `add_*` 位置 | **只在 GridEditor** | 单一写路径 + fused insert 红利 |
| D3 | 元素查找失败 | **返回 None** | 不能逼用户每次 try/except；None 让调用方自己决定严格度 |
| D4 | `in_service=False` 脏级别 | **Topology（full rebuild）** | 确认触发全量重建 |
| D5 | 结果接口 | **`res_bus` 属性（DataFrame）** | display ≈ print，是另一个概念，不混用 |
| D6 | 旧 API 清除 | **干净切换，重写教程** | 不留 deprecated 别名包袱 |

### 遗留命名问题（Phase 2 处理）

模块根部 pandapower IO DTO 已占用 `Bus`/`Line`/`Load`/`Gen` 类名。Phase 1 元素代理
沿用 `BusHandle`/`LoadHandle`/... 命名（用户极少直接写类名）；Phase 2 把 IO DTO 移入
`rustpower.io` 子模块，代理接管干净类名。

### Phase 1 已知妥协（Phase 2/3 偿还）

- 属性 setter 暂时复用现有"增量 + 事件"实现（网关就位后改为立即命令，删除增量算术）。
- `GridEditor.remove()` 立即 despawn，事务 abort 无法撤销 remove（真正的逆操作命令
  在 Phase 3 随 undo 一起到位）。
- `SolveReport.rebuild` 暂只区分 "full" / "incremental"（五级梯子在 Phase 2）。
