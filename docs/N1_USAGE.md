# Rust N−1：故障集合、筛选与详细复测

`experimental_n1` 开启固定 PQ/PV/Slack 分区的线路、两绕组变压器单设备退出。
每个 Entity 表示一个独立退出场景；主 World 保持基态。当前路径支持共享基态 AC
初值和逐场景 DCPF 初值，QLim 留待后续扩展。

## Rust 高层入口

主 World 初始化完成后，挂载需要的 N1Watch，再准备共享数据：

```rust,ignore
use rustpower::n1::*;

let faults = N1FaultSet::out_of_service([line_entity, transformer_entity]);
let prepared = prepare_n1_for(grid.world_mut(), faults)?;
let study = prepared.study()?;
let options = N1RunOptions {
    config: N1Config { workers: 4, batch_size: 8, ..Default::default() },
    ..Default::default()
};
let mut failed = Vec::new();
let report = study.execute(options, |batch| {
    failed.extend(batch.events.into_iter()
        .filter(|event| event.outcome == N1Outcome::NumericalFailure)
        .map(|event| event.outage_entity));
})?;
```

- `prepare_n1_for(world, faults)` 绑定指定故障集合，重复 Entity 保留首次出现。
- `prepare_n1(world)` 准备基态；未绑定集合的 `study.execute` 默认覆盖全部在役支路。
- `prepared.load_faults(faults)` 验证后整体替换；显式空集合产生零场景。
- `study.run(&faults, options, collect)` 运行所选集合；`all_branch_outages()` 提取全部候选。
- `study.details(&faults, options, collect)` 为每个所选数值场景回传状态与配置的明细。

PreparedN1 拥有共享快照，N1Study 借用它。修改源 World 后重新准备，以取得新快照。
`study()` 检查基准态的监测约束；基态已经越界时返回具体错误。

## 规则与输出

规则挂在主 World 的设备或母线 Entity 上，准备阶段编译成共享只读 WatchPlan：

```rust,ignore
world.entity_mut(line_entity).insert(N1Watch {
    rules: vec![WatchRule {
        rule_id: 1,
        condition: WatchCondition::BranchLoadingAbove(100.0),
    }],
});
world.entity_mut(bus_entity).insert(N1Watch {
    rules: vec![WatchRule {
        rule_id: 2,
        condition: WatchCondition::BusVoltageOutsideLimits,
    }],
});
```

母线电压上下限来自该 Entity 的 `VmLimit<PerUnit>`。支路还支持电流阈值；
`SolveFailed` 监测求解失败，`Always` 用于诊断。Always 命中独立于安全越界计数。

| 运行方式 | 通过场景 | 异常场景 | 明细 |
| --- | --- | --- | --- |
| 默认 StatusOnly | 仅更新本地统计 | 发送状态 | 首个越界即可结束检查 |
| 配置 Detailed | 有规则命中时发送 | 有规则命中时发送 | 所有配置规则的命中结果 |
| `study.details` | 发送所选场景状态 | 发送所选场景状态 | 配置规则的命中结果 |

`N1Event` 含 case_id、退出设备身份/Entity、求解阶段、结果类别、迭代数、残差、
错误和 WatchHits。每条命中含规则 ID、source_entity、目标及标量值；同支路的
多个规则共用一份 BranchFlowResult。详细模式按配置监测目标计算结果。

数值场景分为 Passed、ConstraintViolation、NumericalFailure；解列记录保存在
`N1StudyReport.islands`。报告的 processed/passed/violated/numerical_failed/islanded
提供整体计数。默认通过场景只更新 worker 本地 N1Summary，结束时归并一次。

worker 直接借用当前求解电压完成监测，然后继续下一个场景。共享基态由所有
worker 只读访问；每次初值复制由 worker 写入自己的迭代电压区。场景结果省去
完整电压 Vec 的分配与复制。公开 N1Worker 求解 API 仍返回独立电压快照。

## 失败批次与详细补测

同一个 study 可以继续执行：

```rust,ignore
let retry = N1FaultSet::out_of_service(failed);
let retry_report = study.run(&retry, N1RunOptions {
    initialization: Some(N1Initialization::DcPerCase),
    emit_all: true,
    ..options
}, |batch| { /* 每个补测场景的状态 */ })?;
study.details(&retry, options, |batch| { /* 保存明细 */ })?;
```

BaseAc 在主系统准备时求一次收敛基态 AC，各场景从相同电压开始。
DcPerCase 在 worker 内对退出后的 DC 数值系统求解，再启动 Newton。
两种路径复用准备好的 AC/DC Symbolic；每个 worker 独立拥有 Numeric 和可写缓冲。
refactor/solve 失败或产生非有限输出时，保留原 RHS，对当前矩阵重新 factor 后重试。

## 调度、拓扑与内存

调用线程发射任务，P 个 scoped worker 各持一个 App，另有一个 collector。
任务与结果使用 Crossbeam 有界 MPMC，容量各为 2P 批。worker 按组接收任务，
逐场景求解与监测，非空结果批交给 collector 的 `FnMut(N1EventBatch) + Send` 回调。
结果到达顺序取决于完成顺序，case_id 和设备身份用于关联。回调可统计或批量写文件。
当前 collector 为单线程；独立下游攒批及多个 collector 属于后续设计。

队列槽位已经预分配。任务 Vec 与非空事件批仍存在逐批分配；Auto/Full/Budget
缓冲策略尚待实现。后续以池化任务块复用容量，worker 数值工作区继续独立复用。

默认 workers=1、batch_size=8、max_iter=30、tolerance=1e-10。
batch_size 范围 1..=32。max_batch_bytes 默认 64 MiB，约束单组任务/事件和配置
明细的容量估计；worker 固定数值向量、KLU Numeric、线程栈及调用方留存另行核算。

准备阶段对支路多重图做一次 Tarjan DFS，保留桥边、DFS 子树区间和 Slack 计数。
每个桥边退出的解列查询为 O(1)，并联边按设备身份区分。高层 study 将无 Slack
的分区场景归入 islands，其他场景进入数值求解。固定结构的 Ybus patch 使用绝对
CSC 槽位，每次先恢复上个场景的修改，再应用当前退出。

设备名称与原始组件可通过结果 Entity 回查。跨 World 重建或持久化时，同时使用
稳定设备类型和 ID；示例复测会核对记录索引对应的设备身份。

## 两个 Rust examples

```sh
# CASE 为 39、9241 或 pandapower CSV ZIP 路径。
# 以下使用已经满足所配置基态监测约束的网络。
CASE=/path/network.zip
# 精简：输出异常数值场景 JSON 数组，整体计数打印到终端。
cargo run --release --features experimental_n1 --example n1_run -- "$CASE" 4 8 status events.json base-ac

# 详细：全部所选数值场景状态与配置规则命中明细。
cargo run --release --features experimental_n1 --example n1_run -- "$CASE" 4 8 details details.json base-ac

# 从精简文件提取 NumericalFailure，验证设备身份后用 DCPF 初值补测。
cargo run --release --features experimental_n1 --example n1_retry_dc -- "$CASE" events.json retry.json 4 30
```

示例监测线路加载率 >100% 和母线电压上下限；变压器参与退出候选。
n1_run 参数顺序为 CASE、workers、batch、status/details、输出文件、base-ac/dc、
可选 max_iter（默认 30）。数据加载及规则配置共用 `examples/common/n1.rs`。
基态通过约束检查后才执行，
请为自己的网络配置合适的监测规则。空失败集合会提示无需补测。
筛选文件与复测文件来自同一网络快照；拓扑解列计数单独打印，保留在 Rust report 中。

## 源码与验证

| 文件 | 职责 |
| --- | --- |
| src/n1/study.rs | 故障集合、Entity 解析、高层运行和复测 |
| src/n1/execution.rs | worker App、队列、状态统计、借用电压监测及批输出 |
| src/n1/solver.rs、numeric.rs | 共享分析、私有数值求解与 J 填充 |
| src/n1/outage.rs、topology.rs | 数值 patch、桥边与分区信息 |
| src/n1/postprocess.rs、watch.rs | 共享 SoA、portable/AVX2 后处理与规则 |
| src/n1/runner.rs | 共享基态准备及早期底层 base-batch 接口 |

六个 N1 集成入口：study、execution、outage_patches、postprocess、watch、topology。
分别覆盖故障集合、独立重建与跨线程 Symbolic 生命周期、独立盖章、AVX 一致性、
精简/详细规则一致性以及拓扑映射。传统缓存复用保障在 `tests/klu_cache_reuse.rs`。

```sh
cargo test --features experimental_n1 --lib n1::
cargo test --features experimental_n1 --test n1_study --test n1_execution --test n1_outage_patches --test n1_postprocess --test n1_watch --test n1_topology --test klu_cache_reuse
cargo test --features experimental_n1 --example n1_run
```

## 性能工具与历史结果

研究入口位于 tools/n1，以 feature-gated binary 运行；参数保持原有顺序：

```sh
cargo run --release --features experimental_n1 --bin bench_n1 -- 9241 4 8 /tmp/n1-results base-ac status bridges
cargo run --release --features experimental_n1 --bin bench_n1_scale -- case14 /path/case14.zip 4 201 /tmp/n1-scale.json
cargo run --release --features experimental_n1 --bin n1_memory_layout
```

另外保留 bench_acpf_floor 和 n1_bevy_memory 诊断工具。
历史基准以共享收敛基态 AC、batch=8、CPU 0–3 的 P 核运行，执行计时包含发射、
worker 初始化、求解/监测、收集及线程退出；一次性准备另计。原始 6515 结果：
1/2/4 worker 为 25.513/13.231/7.722 s，6474 个数值场景；9241 为
92.753/48.054/28.259 s，14384 个数值场景。这些记录早于移除结果电压 clone。

清理前报告、研究测试与源码备份位于本任务工作区旁的 n1-cleanup-archive，
manifest.json 记录性能工具的新命令。既有原始测量仍在 n1-scalability、n1-6515、
n1-acpf-estimate 和 n1-spsc-performance；这些是本地实验资料。
