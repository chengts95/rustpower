> 当前两种初始化的完整性能对照见 [N1_WARM_START_SOA.md](N1_WARM_START_SOA.md)。

> 早期逐场景 DCPF + AoS 路径的性能结果见 [N1_PERFORMANCE.md](N1_PERFORMANCE.md)；当前初始化可选，分支目录为 SoA。

# 真实单支路 N−1：输入、输出与实体触发器

当前 Rust 实验入口由 `experimental_n1` 开启。目标为固定 PQ/PV/Slack 分区、
关闭 Q-limit 的线路/两绕组变压器整设备退出，沿用共享固定结构与符号分析。

## 输入

### Rust 高层入口：借用资源、按实体描述故障

准备阶段可直接指定本次故障集合：

```rust
let faults = N1FaultSet::out_of_service([line_entity, transformer_entity]);
let prepared = prepare_n1_for(app.world_mut(), faults)?;
let study = prepared.study()?;
let report = study.execute(options, collect)?;
```

`PreparedN1::load_faults` 可在保留共享基准/符号分析的同时替换集合，`faults()`
可读取已加载集合。加载时校验实体和在役状态，失败时保留先前集合。显式空集合
表示执行零个场景；未绑定集合的 execute 默认选择全部在役支路。
`study.run(&subset, options, collect)` 与 details 继续用于指定子集补测。

```rust
let prepared = prepare_n1(app.world_mut())?;
let study = prepared.study()?;
// 每个实体是一个独立退出场景，顺序保留，重复实体合并。
let faults = N1FaultSet::out_of_service([line_entity, transformer_entity]);
let options = N1RunOptions {
    config: N1Config { workers: 4, batch_size: 8, ..Default::default() },
    ..Default::default()
};
let mut failed = Vec::new();
let report = study.run(&faults, options, |batch| {
    failed.extend(batch.events.into_iter()
        .filter(|event| event.outcome == N1Outcome::NumericalFailure)
        .map(|event| event.outage_entity));
})?;
let retry = N1FaultSet::out_of_service(failed);
let retry_report = study.run(&retry, N1RunOptions {
    initialization: Some(N1Initialization::DcPerCase),
    emit_all: true,
    ..options
}, |batch| { /* 接收所有补测场景的状态 */ })?;
study.details(&retry, options, |batch| { /* 配置规则的明细与每个所选场景状态 */ })?;
```

`N1Study<'a>` 借用 PreparedN1，持有一次构建的实体查找表。World/App 继续由调用方
管理；共享 SoA、矩阵模式和 Symbolic 保存在 PreparedN1。故障集合描述源实体的
独立 OutOfService 情景，worker 通过数值 patch 执行。源 World 的实体组件保留
基态内容，研究对象复用准备时的快照；主数据改变后重新准备下一份基准。

构造 study 时完成一次基态数值/约束检查；Always/SolveFailed 属于诊断。每轮先
解析实体集合，明确返回未知设备或基准已停运设备，再将无 Slack 的桥边退出分类到
report.islands，其他场景交给原 scoped worker 队列。report 的 passed、violated、
numerical_failed、islanded 四类计数覆盖整个故障集合。原始数值汇总在 report.solve。
`all_branch_outages()` 可构造全部在役支路的故障集合。

默认简要输出维持稀疏事件；emit_all 适合小批补测，成功场景也回传状态。
details 为本轮设置 Detailed＋emit_all，空命中场景仍有状态，明细内容遵循配置规则。
每轮初始化选项复用同一份 AC/DC 共享分析。底层 run_n1 保留原有默认，另有
run_n1_with_initialization 显式指定本轮初始化。

QLim 属于后续 worker 求解策略扩展。传统 QLimPlugin 通过源管线的 PV/PQ 标签、
QLimited、NodeTypeChangeEvent 与外层迭代推进。独立场景后续应以共享发电机限值表
和 worker 私有触限/分区状态实现；PV→PQ 后的 J/符号分析随该场景分区维护。
当前实现使用已验证的固定分区路径。

1. 已初始化主 World：包含设备实体、基准 Ybus、注入及母线映射。准备时提取
   一份共享 BranchCatalog 与固定结构/符号；worker 不复制设备实体。
2. 候选支路：主 World 提取目录的密集索引流。`active_branches()` 发射全部
   基准在役设备；也可传选择后的索引迭代器。每项独立退出一个完整设备。
3. `N1Config`：workers、batch_size、max_iter、tolerance、max_batch_bytes。
   batch_size 范围 1..=32；默认 8。线程默认 1，场景迭代默认 30、容差 1e-10。
   任务/事件队列容量分别为 2×workers，场景逐组生成。
4. 可选实体组件 `N1Watch`：配置该支路在所有候选场景中的监测规则。

组件挂在**被监测的支路或母线实体**上。退出哪个设备由候选场景决定，两者各自独立。
例如，候选退出 A/B/C，同时在 D 上挂电流阈值规则，观察 D 在这三种退出下的结果。

## 固定基态的桥边与解列查询

`prepare_n1` 额外构建一份共享 `N1Topology`。以置换后 Ybus 的母线索引为顶点，
以 BranchCatalog 中在役且有互导纳的设备为独立边，执行一次迭代 Tarjan DFS，
耗时与临时空间均为 O(V+E)。并联设备保留各自编号，合并后的自环和对地支路
对连通性贡献为零。DFS 邻接与 low 数组在准备结束后释放。

```rust
let topology = prepared.topology();
if let Some(cut) = topology.bridge_cut(outage_index) {
    let detached_buses = cut.slackless_bus_count();
    let subtree_buses = &topology.dfs_buses()[cut.subtree.clone()];
    // subtree_slacks / other_slacks 是两侧已有的固定 Slack 数量。
    // 已知某个 Slack 的 solver_index 时，两个整数比较即可判断其所在侧：
    let slack_in_subtree = cut.contains_in_subtree(topology.dfs_number(slack_solver_index).unwrap());
}
```

桥边查询和两侧计数均为 O(1)，枚举断开侧母线时按 DFS 连续区间读取。
`subtree` 表示 DFS 子树，`component` 表示该桥所在基态连通分量；另一侧由
component 去除 subtree 的最多两个连续片段组成。两侧 Slack 数量直接支持
多 Slack；切开后两侧各有参考母线时，slackless_bus_count 为零。

这组接口供发射端决定解列场景的去向；`run_n1` 按调用方提供的候选索引流求解。
例如在发射时记录无 Slack 的场景，将其余索引继续交给求解队列：

```rust
let mut islands = Vec::new();
let candidates = prepared.active_branches().filter(|&index| {
    if let Some(cut) = prepared.topology().bridge_cut(index) {
        if cut.slackless_bus_count() > 0 {
            islands.push((index, cut));
            return false;
        }
    }
    true
});
let summary = run_n1(&prepared, config, candidates, |batch| { /* 处理结果 */ })?;
// islands 进入解列处理；summary 统计送入求解队列的场景。
```

## 输出模式：筛选与明细

`N1Config.reporting` 提供两档：

- `N1Reporting::StatusOnly`：任一配置的数值约束越界即结束检查，数值求解失败
  直接作为未通过。每个未通过场景只发一个带 N1Outcome 的状态事件，WatchHits 为空，
  通过场景静默并计入汇总。Always/SolveFailed 属于诊断触发器；此模式直接按结果发射。
- `N1Reporting::Detailed`：完整评估规则，按配置发射明细与诊断事件，保留原有语义。

库的 N1Config::default() 与 N1Reporting::default() 均为 StatusOnly；`n1_run` / `bench_n1` 示例也默认 status。Detailed 由调用方显式选择。
示例最后一个参数可选 `status` 或 `details`。状态为 Passed、ConstraintViolation、
NumericalFailure，分别表示本次求解及已配置约束通过、约束越界、数值求解失败。

```rust
let config = N1Config { reporting: N1Reporting::StatusOnly, ..Default::default() };
let mut rejected = Vec::new();
let summary = run_n1(&prepared, config, prepared.active_branches(), |batch| {
    rejected.extend(batch.events.into_iter().map(|event| event.outage_index));
})?;
// 用户选择某些场景后，按稳定 outage_index 补算详细结果。
run_n1(&prepared, N1Config { reporting: N1Reporting::Detailed, ..config },
    rejected.into_iter().take(10), |details| { /* 处理需要的明细 */ })?;
```

简要检查先扫描电压平方，再按需计算支路两端电流平方和加载率平方。
首个越界直接返回，省去全结果功率、角度、开方和 WatchHit/BranchFlowResult 分配。
详细模式保留原 AVX2 两端口结果。为保持内存上限，简要运行默认不缓存所有场景的
电压；后续明细请求会重新求解用户选择的场景。

N1Summary::rejected() = failed + violated_cases；passed() 为其余完成场景。
failed 继续专指数值失败。StatusOnly 的 emitted_hits、violation_hits 为 0，表示
没有明细载荷；完整命中总数只有 Detailed 扫描可得，benchmark JSON 相应字段用null。
StatusOnly 的分组内存上限只计求解电压和任务/状态事件，省去明细最坏容量预留。

## 输出

### 按失败记录补测逐场景 DCPF 初始化

`n1_retry_dc` 读取 benchmark 的事件 JSON，选出 NumericalFailure，按稳定
outage_index 核对设备种类与 ID，将这些索引作为第二批任务交给 run_n1。
第二批通过 N1Study 选择 DcPerCase，默认四 worker、batch=1、max_iter=30、tol=1e-10；
重试批次共享其固定基态和 AC/DC 符号分析。解列场景进入专门的分类处理路径。
输出保留每个场景的原记录、DCPF 重试阶段、迭代数、残差和失败原因。
补测使用简要输出与 emit_all，成功且满足约束的结果也被记录。

```sh
cargo run --offline --release --features experimental_n1 --example n1_retry_dc -- 9241 events.json retry-dc.json 4 30
```

2026-10-27 对桥边筛选后的四 worker 失败记录补测：10 个场景的 DCPF 均成功，
后续 Newton 均达到 30 次迭代上限，收敛恢复数为 0。补测队列墙钟 0.228666 s，
另行加载、共享准备和基准检查合计 0.103915 s。原始结果为任务目录
n1-dc-retry/results.json；此次通过单独的第二批完成补测。

`run_n1` 返回 `N1Summary`：submitted、processed、failed、emitted_cases、
emitted_hits、workers_started、violated_cases、violation_hits。这些统计独立于 MPSC 业务消息。

收集回调只收到 `N1EventBatch`，其中事件包含：

- case_id、退出设备密集索引/原实体/源设备种类与 ID。
- worker_id、求解阶段、迭代数、残差和失败原因。
- 命中的规则、规则所属实体、目标设备与指标值、目标的支路结果。

`N1Event::hits` 现在是 `WatchHits`：内部为 `Soa<WatchHit>` 加独立支路结果列表。
命中保留 rule_id、源实体、WatchTarget、value 和 result_index；WatchTarget::Branch
携带目录索引与 BranchIdentity，WatchTarget::Bus 携带求解器索引与原 BusID。
同一场景同一目标的所有命中引用一份支路结果。只有实际命中的目标才进入结果列表，
失败场景的结果索引为空。字段列可通过 `records()` 读取，完整结果按命中位置查询：

```rust
for (index, hit) in event.hits.iter().enumerate() {
    let rule_id = *hit.rule_id;
    let source = *hit.source_entity;
    let value = *hit.value;
    let flow = event.hits.branch_for_hit(index); // Option<&BranchFlowResult>
}
```

支路结果按目标单独存放，多规则共用一份；电压命中仅保存母线身份与标量电压，
result_index 为空，省去完整支路载荷。本轮未加入字段选择、摘要聚合或规则过滤策略。
`max_batch_bytes` 的计算已统一计入命中列与独立结果列表。

Detailed 模式下，零命中任务不写 MPSC，失败事件由显式 SolveFailed/Always 规则决定。
StatusOnly 模式下，数值失败或任一阈值越界发一个状态事件，成功场景只计入汇总。

这是数值求解和监测事件流。电压范围、热额定阈值等安全判据由用户规则定义。
当前规则覆盖支路加载率、电流、母线电压上下界与求解状态。Converged 表示功率
残差收敛；violated_cases / violation_hits 统计已收敛场景中的阈值越界，Always 和
SolveFailed 诊断规则单独作为事件计数。

## 实体组件与共享规则表

使用组件与主 World 中设备配置保持一致。支路条件挂支路实体，电压条件挂母线实体：

```rust
world.entity_mut(target_entity).insert(N1Watch {
    rules: vec![
        WatchRule { rule_id: 1, condition: WatchCondition::BranchLoadingAbove(100.0) },
        WatchRule { rule_id: 2, condition: WatchCondition::BranchCurrentAbove(1.2) },
        WatchRule { rule_id: 3, condition: WatchCondition::SolveFailed },
    ],
});
```

母线电压规则直接读取该实体已有的 VmLimit<PerUnit>：

```rust
world.entity_mut(bus_entity).insert(N1Watch {
    rules: vec![WatchRule { rule_id: 3, condition: WatchCondition::BusVoltageOutsideLimits }],
});
```

prepare 时将实体、BusID、BusMatrixIndex 经合并映射及 to_perm 转为 solver_index，
和平方上下限一起存入共享 `Soa<BusWatch>`。worker 对求解电压计算 |V|²，与静态
平方阈值比较；仅命中时取平方根生成标量载荷。数值失败时跳过电压规则，保留
显式失败告警。限值在本批次内冻结；更改主 World 后重新 prepare。

rule_id 在同一实体内唯一。规则在 `prepare_n1(world)` 中一次编译成共享 WatchPlan，
worker 无需复制组件或遍历原 World。相同实体的多个规则复用一次目标计算。

`Always` 是显式诊断规则，适合正确性检查或需要每场景输出的调用方。
失败场景只发送状态，branch 数值载荷为空。正常场景的退出设备按本场景掩码置零。

关联实体列表也能作为前端配置方式：准备时展开到同样的目标/规则表即可。首版
只提供组件入口，后续无需改变 worker 规则执行方式。规则和基准在本批次内固定；
下一次 prepare 后读取新配置。

## 运行方式

```rust
// 默认：主系统求一次基准 AC，worker 每个场景从这份电压启动 Newton。
let prepared = prepare_n1(main_world)?;
// 可选：省去基准 AC 初始化，每个 worker 按场景执行 DCPF + Newton。
// let prepared = prepare_n1_with_initialization(main_world, N1Initialization::DcPerCase)?;
let config = N1Config { workers: 3, batch_size: 8, ..Default::default() };
let summary = run_n1(&prepared, config, prepared.active_branches(), |events| {
    // 独立收集线程：接收、存储或归档 events.events。
})?;
```

选择某个源实体时可用 `prepared.branch_index(entity)` 转成目录索引，或从
`prepared.base().catalog().identities()` 按源设备 ID 选择。

可运行示例：

```sh
cargo run --offline --features experimental_n1 --example n1_run -- 3 8
```

参数依次为 workers、batch_size、初始化方式（base-ac 或 dc，默认base-ac）、
输出方式（status 或 details，默认status）。
示例使用 IEEE39，默认线路加载率、母线电压范围及显式失败规则；变压器继续作为退出候选。
原始案例若已有母线越界，基准检查先返回对应目标。
迭代失败场景计数并继续。

完整 9241 性能测试支持桥边筛选：

```sh
cargo run --offline --release --features experimental_n1 --example bench_n1 -- 9241 4 8 /tmp/n1-results base-ac status bridges
```

最后一个参数默认 bridges：发射端逐场景查询已准备的 Tarjan 结果，无 Slack 的
解列场景写入单独 islands.json，其余送入 worker；all 可用于对照原始全场景求解。
summary 中 candidates/classified 为全部候选，processed 为实际数值求解数量，
islanded_cases 为筛出的解列数量，rejected 包含解列及数值/约束未通过场景。
计时包含发射筛选、worker 初始化、数值求解、worker 内后处理、结果收集及线程退出；
事件和解列 JSON 排序/写盘在该计时结束后完成。共享准备和基准检查单独记录。

## 示例的基准检查与监测范围

`n1_run` 和 `bench_n1` 共用 `examples/common/n1.rs`：默认监测线路加载率 >100% 和在役母线的电压范围，另挂一次求解失败告警。变压器加载率监测已撤掉，变压器退出候选继续参与求解。
主库保留显式配置的支路规则与变压器数值后处理接口。

示例在发射任务前，先用未退出任何支路的基准 AC 电压检查线路和母线电压规则。基准有命中时
返回错误，报告命中数量及首个目标；本次运行在基准检查结束。通过时打印
`BASELINE line_loading+bus_voltage violations=0`，随后启动 N−1。

BaseAc 模式复用准备阶段的基准解；DcPerCase 模式为基准检查单独求解一次 AC，
worker 继续按原策略逐场景 DCPF。基准检查时间独立记录为 `baseline_check_seconds`，
首批总时间包含共享准备、基准检查和整批执行。

本轮完整 PEGASE9241 验证：基准线路约束命中 0 项，全部 16,049 项完成。
四 worker、batch_size=8 下产生 1,716 条命中，其中线路越限 290 条（222 个场景）、
求解失败 1,426 条；相较旧配置 7,548,906 条命中，重复变压器加载率输出已移除。
旧测量中全部 233 个零步场景（含此前新增的 231 项）本轮均静默，失败场景集合与旧基准 AC 测量一致。
整批 37.461 s，基准检查 0.597 ms；当前整批时间仍包含后处理、传输和收集。

验证覆盖基准通过/基准越限中止、两种初始化、人工高变压器加载率下的规则排除和
变压器候选保留。日志及原始输出在任务目录 `n1-line-only/` 和 `n1-line-only-tests.log`。

## 当前失败场景的离线诊断

对最新 9241 四 worker 结果与原始在役线路/变压器多重图作对应。基准只有一个
连通分量，唯一 Slack 是母线 4230。1,426 个失败场景中：

- 1,416 项为桥边退出，切出子网全部没有基准 Slack；子网为 1–11 个母线，
  其中 1,162 项仅切出一个母线。
- 10 项保持网络连通，其中 6 项达到 Newton 50 次迭代上限，4 项线性解出现非有限值。

拓扑影响与数值终止状态应分别记录。残差通过的切岛场景同样具有拓扑标记；
备用求解器重试可针对保持连通的失败候选。此处为离线诊断，生产任务发射和重试
策略尚未改动。每项退出设备、端点、迭代、残差与切出规模见任务目录
`n1-line-only/failure-diagnosis.json`。数值终值只说明本次方法的终止状态；
进一步选择重试策略需要保留初始/最佳/最终残差，以及 refactor 与回退 factor
各自的 KLU 状态、零主元位置。后续以一份紧凑失败记录按场景发送。

## 连通失败复现与线性恢复修正

10 个连通失败场景已分别用原始 case 中设备退出后重新初始化全部矩阵来对照。
增量 Ybus 最大差 2.34e-13，DC B 最大差 4.55e-13，DC 初始复电压最大差
1.36e-13 pu，移相 RHS 差为零。独立全量矩阵上每轮使用完整 factor 的 Newton
在这 10 项中也均未于 50 步内收敛；AC 解的存在性尚未判定。

原逻辑在 refactor 状态成功但 solve 非有限时直接返回。现已增加同 J、同原始 RHS
的单次完整 factor 重试，原 RHS 缓冲由 worker 独占并复用，分配纳入 vector_bytes。
CaseSolution、N1Event、N1Summary 的 linear 字段记录 factor_calls、refactor_calls、
refactor_solve_retries；完整 factor 失败时记录 KLU 数值零主元与原矩阵列位置。
同时在进入 KLU 前检查 J/RHS，非有限输入记录行列；AC 错误附最小电压、对应母线、
最大电压，避免把非有限 J 的问题归入有效矩阵上的主元复用问题。

本批非有限复现已追到 J 组装前的零电压：例如 Transformer 1447 场景，母线3436
电压归零，下一轮 J(2880,12120) 出现 NaN。V4 对角修正包含 P/|V| 和 Q/|V|。
对场景488、7935、15244的末次有限 J/RHS 分别运行 refactor/full factor，最后一步
两种结果均为零或极小预测电压；病态阶段两者修正量已有明显差异。后续需研究从
有效迭代点回退和步长控制。诊断源代码、逐场景结果与每步对照保存在任务目录
`n1-failure-probe/`，其中 summary.json、details.json、same-system-trace.json 可直接查看。

回归测试：91 项主库通过、2 项忽略，25 项集成通过；新单元测试在有限输入上复现
旧主元导致 refactor solve 溢出，完整 factor 恢复正确解，另验证重试有界、原 RHS
恢复、非有限输入拦截和私有求解器失败后恢复。细化错误位置后两项恢复测试再次通过。
旧 Newton/KLU API 源码保持原样，改动位于实验 N−1 模块。

## 场景迭代上限与初值效果

N1Config 默认 max_iter 已降为 30，调用方仍可显式覆盖。历史 50 步测量中，
BaseAc 模式所有通过场景最大为 17 步，DcPerCase 最大为 16 步。主线程基准 AC
初始化和基准检查仍保留独立的 50 步上限；它们各执行一次。

BaseAc 传递完整的基准 AC 电压幅值与相角，DCPF 初始化则用 DC 相角配原始
电压幅值。历史结果中常规 DCPF 场景多为 6 步，基准 AC 初值多为 2–4 步。
额外通过的 243 项全部为桥边退出：231 项零步、12 项13–16步；保持连通的10个
失败场景在上述初值和完整factor对照中仍失败。这些“额外通过”应与常规迭代加速分开统计。

30 步默认配置已完整验证 9241 的 16,049 项：失败仍为 1,426 项，失败场景集合、
输出场景集合和每场景命中数均与之前 50 步测量一致。AC 尝试次数从 54,435 降到
50,926，四 worker 整批 36.743 s。本轮同时包含更早的有限输入检查和数值恢复改动，
耗时差异按组合改动记录；对应原始结果为任务目录 `n1-cap30/`。

## 12 个非零步切岛“通过”的电压复核

这 12 个退出分别切出两个母线的小岛，岛内指定 Sbus 均为 0，全部为 PQ，无 PV
发电机或原 Slack。用独立 AC worker 从基准 AC 电压复现，13–16 步后功率残差
低于 1e-10，但岛内电压仅为 1.57e-5～1.20e-4 pu，已接近失电。代表场景
Line 3549：母线2101/8018从约0.975 pu降到2.98e-5 pu，残差5.79e-11。
功率残差检验会接受这类近零电压解，现有Converged标签是数值终止状态。
业务结果应保留切岛/失电类别，电压约束与供电状态作为独立安全判定。
原始复核结果：任务目录 `n1-island-passes.json`，可复现源码为
`n1-failure-probe/src/bin/islands.rs`。本轮只完成电压复核，生产拓扑判定尚未接入。

## 电压规则完整集合验证

新增 `BusVoltageOutsideLimits` 后，9241 基准态线路和电压规则均为 0 项越界，
全部 16,049 项执行完成。四 worker、batch=8、max_iter=30、tol=1e-10、CPU0–3：
整批 36.056 s（之前仅线路规则 36.743 s，单次对照，数值工作量相同为 50,926 次尝试）。
本轮后处理/规则累计墙钟 10.225 s，之前为10.384 s；这些是worker区间之和。

数值失败 1,426 项，失败场景集合保持一致；503 个数值收敛场景有约束越界，
共2,387条阈值命中：线路加载率290条，电压2,097条（涉及291个场景）。加上
1,426条失败告警，事件共1,929个场景、3,813条命中。已核对12个近零电压孤岛
场景，每项恰好对其两个低压母线产生告警。原零步空末端场景仍按现有规则静默；
拓扑状态属于后续独立处理。

原始IEEE39基准在BusID35上有1.0636 pu电压，超过案例自身1.06上限，新基准检查
按规则提前返回该目标，未发射N−1。测试使用专门的合格基准fixture覆盖继续运行。

最新源码回归：91项主库测试通过、2项忽略；29项集成通过。新增覆盖上下边界、
欠压/过压、非有限值、失败场景、合并母线与非平凡置换、冻结限值、错误规则实体、
数值收敛同时电压越界的跨worker传输，以及两种初始化的基准电压拒绝。
原始日志在任务目录 `n1-voltage/`、`n1-voltage-build-tests.log`、`n1-voltage-tests-final.log`。

调度仍使用共享有界任务队列和动态分组，未改变成静态分区。本轮没有与LS2G静态
分区做同口径调度对照；既有3.18倍四P核加速是较早规则配置的整体扩展测量。

## 简要／详细模式完整集合对照

同一二进制、同一9241全部16,049项、BaseAc、四worker、batch8、max_iter30、tol1e-10、
CPU0–3，依次各运行一次：

| 模式 | 整批墙钟 s | 后处理／规则累计 s | 场景状态事件 | 明细命中载荷 |
|---|---:|---:|---:|---:|
| StatusOnly | 33.923 | 2.122 | 1,929 | 0 |
| Detailed | 36.315 | 10.218 | 1,929 | 3,813 |

两档通过14,120项、未通过1,929项（数值失败1,426、约束越界503），未通过集合及
逐场景outcome完全一致。AC尝试次数均为50,926。简要模式在这组单次对照中整批
耗时低6.6%，规则累计时间低79.2%；省去的主要是完整支路结果计算和命中载荷生成。
阶段累计是worker区间墙钟之和，包含调度停顿。

新增测试覆盖两档分类一致、选中场景补算明细、简要模式忽略Always诊断触发、
无失败watch时仍上报数值失败、命中缓冲零容量，以及使用小max_batch_bytes时
简要模式继续运行而详细模式因最坏载荷容量超限返回错误。电压平方与支路电流／
加载率平方判据分别与详细模式核对。91项主库测试与32项集成测试通过，2项忽略。
原始记录位于任务目录 `n1-reporting/`，回归日志 `n1-status-build-tests.log`、
`n1-status-tests-final.log`。

## 初始化选择和数值路径

`N1Initialization::BaseAc`（默认）：`prepare_n1` 在主线程执行一次基准
DCPF + AC Newton（max_iter=50，tol=1e-10），将收敛电压冻结在共享基准中。
`PreparedN1::base_solution()` 返回这一次主线程求解结果。准备失败直接返回错误。
worker 采用 `new_ac`，每项从同一份基准电压启动 `solve_ac_current`；私有 DC
数值数组、RHS、Common、Numeric 均省去。共享准备仍保留一份 DC 模式及 Symbolic，
用于主线程基准初始化。每项的 DC 时间为零。

`N1Initialization::DcPerCase`：通过 `prepare_n1_with_initialization` 选择。
准备阶段只构造共享结构和规则表，`base_solution()` 为空；worker 采用 `new`，
对每个退出后的网络执行 `solve_current`（DCPF + Newton）。它保留私有 DC
数值数组和 Numeric/Common。选项固定在 PreparedN1 中，worker 和任务使用同一模式。

主线程从 SoA 导纳和已置换端点解析 COO 对应的 CSC 槽；默认 `for_ac_branch`
只生成 Ybus 四槽绝对覆盖值。`DcPerCase` 使用 `for_branch`，另生成 DC B 与
DC RHS 绝对覆盖值，覆盖聚合导纳的模、并联设备、Slack 边界与变压器移相。
重复槽先合并，模式保持共享。

worker 回填前一项触及的基准槽，再覆盖本项的小 stencil。每场景保持独立初值，
连续任务复用私有 AC 分解；数值失败后重建自己的分解。正常切换只触碰固定槽。
退出信息同时传给后处理和规则判定，该设备结果为零。

## SoA 与原 AVX2/FMA 内核

BranchCatalog 仅保存一个 `Soa<BranchData>`，通过可选依赖 soa-rs 1.0.1 的
`#[derive(Soars)]` 生成端点、四个复导纳、电流基值、服务状态及加载率换算列。
`BranchData` 是唯一的字段定义，容器统一维护长度、容量和各列存储，所有 worker
只读共享。原手工逐字段 Vec、逐列 push 和 allocation_bytes 统计已移除。
额定值和基值换算在准备时完成，
数值后处理使用乘法。生产线程生成 outage stencil 直接读取 SoA 导纳列。

AVX2/FMA 由 OnceLock 检测一次，复用 `basic::ecs::post_processing` 原有的两端口
内核；其他架构选择 portable 内核。这个内核的 SIMD lane 对应单支路两个端口的
实虚分量。任务队列仍以 1..=32 场景分组；跨场景 SIMD 和批量规则内核留作下一步。
`branch(index)` 临时重构兼容诊断视图，数值后处理和任务生成直接读取 SoA。

## DCPF rank-1 修正研究记录

现有 `DcpfModel::from_ybus` 对非对角聚合导纳取模，再组装列和为零的 B。
设退出支路两端为 f/t，`d_f = |Y'_{tf}|-|Y_{tf}|`、
`d_t = |Y'_{ft}|-|Y_{ft}|`，则完整 B 的变化为：

```text
u = e_f - e_t
v = d_f e_f - d_t e_t
B' = B + u vᵀ
```

删除 Slack 行列后仍为 rank-1 或零；并联和移相造成的非对称权重仍包含在 v 中。
退出造成的移相/Slack RHS 改变量也沿约化后的 u。令 p 为该支路原有的移相
注入（f 加 p、t 减 p），则两端活动时 `α=p`；f 活动/t Slack 时
`α=p+d_f θ_t`；f Slack/t 活动时 `α=p-d_t θ_f`。这按现有实现读取
活动列中的 Slack 邻接权重。若 `r'=r+αu`，已有
`θ=B⁻¹r` 且 `z=B⁻¹u`，则：

```text
θ' = θ + z (α - vᵀθ) / (1 + vᵀz)
```

公式适用于固定 Sbus、母线分区和 Slack 参考，且基准 B 可逆。
构造每条支路的 z 需要利用基准 B 分解进行一次三角求解，修正向量长度随母线数
增长；后续需评估准备端计算和传输成本。分母接近零按数值失败处理。这条路线
目前为研究记录，运行入口提供上面的基准 AC 和逐场景 DCPF 两种实现。

## 验证

Rust 高层入口及原数值路径共 47 项相关回归通过，包含同一 Symbolic 的逐轮 DCPF
补测、实体故障去重/解析、解列计数、无监测规则时所选场景完整回传，以及母线和
支路 Always 诊断排除。日志位于任务目录 n1-study-*.log。
14～9241 的 1／2／4 worker 趋势、首次调用总成本及 14 母线 batch=1 补充对照见
[N1_SCALABILITY.md](N1_SCALABILITY.md)。

### 2026-10-27：调度 batch=1 与 batch=8

沿用上一轮完全相同的 release 二进制，CPU 0–3、BaseAc、StatusOnly、桥边筛选、
max_iter=30、tol=1e-10；依次完整运行下列三个配置各一次：

| 配置 | 批次墙钟 | 进程平均忙核数 | 峰值 RSS |
| --- | ---: | ---: | ---: |
| batch=1，4 worker | 31.107 s | 3.809 | 120.17 MiB |
| batch=8，4 worker | 30.563 s | 3.837 | 123.47 MiB |
| batch=1，1 worker | 98.287 s | 0.977 | 97.25 MiB |

本轮四 worker 的 batch=1 相对 batch=8 增加墙钟 1.78%；batch=1 对自身单 worker
基线加速 3.160 倍、并行效率 78.99%。三组 AC 总尝试均为 43709，逐场景结果
完全一致。当前调度采用共享队列，batch=1 将动态领取细化到单场景；work stealing
移出本轮实施范围，跨场景 SIMD tile 保留为独立的后处理设计项。
原始记录与逐场景比较位于任务目录 n1-batch-performance/comparison.json。

### 2026-10-27：桥边筛选后的完整 9241 测试

同一 release 二进制，BaseAc、StatusOnly、batch=8、max_iter=30、tol=1e-10。
每个配置完整运行一次；CPU 0–3 为四个 P 核，0–5 额外开放两个 E 核。

| 配置 | 批次墙钟 | 进程平均忙核数 | 进程峰值 RSS |
| --- | ---: | ---: | ---: |
| 4 worker，全场景求解，CPU 0–3 | 34.188 s | 3.917 | 135.36 MiB |
| 4 worker，桥边筛选，CPU 0–3 | 29.322 s | 3.909 | 123.63 MiB |
| 4 worker，桥边筛选，CPU 0–5 | 29.204 s | 3.915 | 123.13 MiB |
| 1 worker，桥边筛选，CPU 0–3 | 93.293 s | 0.989 | 97.59 MiB |

1665 个无 Slack 解列场景在发射端单独分类；其余 14384 个场景数值求解，
13889 通过、485 约束越界、10 数值失败，总 AC 尝试 43709 次。连通场景的
状态与全场景对照一致，单 worker、四 worker、六核可用三组的状态也一致。
四 worker 的桥边筛选缩短墙钟 14.24%，factor 调用由 2144 降为 6；
相对于单 worker 加速 3.182 倍、并行效率 79.54%。

四 worker 配置共有六条线程：主线程发射，四个 worker 内执行 PF、后处理和
规则检查，collector 收集结果。CPU 0–3 测试中，采样累计主线程约 0.149 CPU s、
collector 约 0.012 CPU s，每个 worker 约 28.5–28.9 CPU s。开放六个核的
单次墙钟差为 0.40%，主要计算持续集中在四个 worker。线程 CPU 数据按
/proc schedstat 每 0.1 s 采样，进程 CPU 总量与峰值 RSS 由 wait4 读取。

批次墙钟从进入 run_n1 前开始，到全部线程退出及收集完成结束，包含逐场景
桥边筛选；共享准备约 0.033 s、基准约束检查约 0.0006 s 单独记录。JSON
整理和写盘在批次墙钟之外。原始事件、解列分类、CPU/内存记录和逐场景比较
位于任务目录 n1-topology-performance/，comparison.json 汇总全部结果。

- 固定槽 patch 对照独立盖章/DC 重建：并联设备、7° 移相、Slack 角度、非平凡
  置换、合并重合槽、连续任务恢复，以及 AC/DC 模式配对。
- 两种初始化对同一退出场景的结果对照；基准电压复用、AC-only worker 的零 DC
  分配和计时；失败后继续下一场景。
- SoA portable/AVX2 数值逐字段对照、线路/变压器额定值、退出掩码；规则冻结、
  稀疏/密集命中和静默运行；多规则共用目标结果、跨目标索引及跨场景隔离。
- 完整重建参考 fixture 显式安装 StructureUpdatePlugin，并验证 full_rebuild、
  Ybus、电压和实体后处理结果。旧生产求解 API 保持原样。

最新维护重构日志：任务目录 `n1-soars-tests.log`（89 项主库和 24 项集成测试通过，2 项忽略，相关示例编译通过）。早期初始化改造日志为 `n1-warm-soa-tests.log`。基准和旧数据的初始化模式应分别标注。
