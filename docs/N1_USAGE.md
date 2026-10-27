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

## Python 绑定

编译同时开启 `python,experimental_n1`，例如在自己的 venv 中执行：

```sh
maturin develop --release --features python,experimental_n1
```

```python
import rustpower as rp
import pandas as pd

grid = rp.PowerGrid("network.zip", qlim=False)
# faults 可传 line/transformer handle 列表；省略时选择全部在役支路。
study = grid.prepare_n1()
report = study.run(workers=16, batch_size=8)
print(report.summary)

retry = study.run(report.numerical_failures, initialization="dc", workers=16)
details = study.details(report.rejected_faults, workers=16)
events = pd.DataFrame(details.events)
hits = pd.DataFrame(details.hits)
flows = pd.DataFrame(details.branch_results)
```

默认只检查数值收敛。准备时设置 `voltage_limits=True` 和/或
`line_loading_limit=100.0` 启用对应监测；基准态需满足选定约束。
配置规则在准备后恢复到源 World 原状，编译后的规则归共享快照持有。

N1Faults 关联源 grid，可直接传回同源 study；重复设备去重，显式空集合保持为空。
`report.numerical_failures` 选择数值失败，`rejected_faults` 选择数值失败或安全越界，
`island_faults` 单独选择拓扑解列。跨 grid handle 与故障集合会报错。

N1Study 持有 PreparedN1 快照与源 grid 引用。每轮通过 Rust 高层入口运行，重用
共享结构和 Symbolic；本轮的高层验证、Entity 查询表构造及执行均计入
summary.execution_seconds。summary.preparation_seconds 记录最初准备时间。
run/details 整批释放 GIL，worker 与 collector 内处理 Rust 数据；返回后按需访问
结果列时才生成 Python/NumPy 数据。源数据后续修改影响下一次准备，已有 study
继续使用自己的固定快照。

精简通过场景仅计入汇总；详细结果在本轮返回前保存在 Rust 结果容器。
events 的数值列为 NumPy 数组，类别/错误列为列表。hits.event_row 关联事件行，
hits.result_index 关联 flows 行（-1 表示无支路结果）；flows.event_row 也关联事件。
缺少标量的命中值使用 NaN。Entity 列保留原 World 的 u64 标识，设备类型和 ID
用于持久化关联。结果数组拥有自己的内存，可独立于 report 生命周期保留。

Python 回归集中在 `scripts/test_n1_api.py`，需要 numpy 和 pandapower。

## 性能工具与历史结果

研究入口位于 tools/n1，以 feature-gated binary 运行；参数保持原有顺序：

```sh
cargo run --release --features experimental_n1 --bin bench_n1 -- 9241 4 8 /tmp/n1-results base-ac status bridges
cargo run --release --features experimental_n1 --bin bench_n1_scale -- case14 /path/case14.zip 4 201 /tmp/n1-scale.json
cargo run --release --features experimental_n1 --bin n1_memory_layout
```

### 一键扫描：1 与 4 worker 的跨规模效率

```sh
cargo build --release --features experimental_n1 --bin bench_n1_scale
# 使用安装了 pandapower 的 Python；缺失的案例 ZIP 会从本机 pandapower 导出。
python tools/n1/scan_scale.py --cpus 0-7 --out /tmp/n1-scale-4
```

依次覆盖 14/30/57/89/300/1354/2869/9241 母线，比较 1 与 4 个 PF worker。
两档共用 CPU 范围，默认 0–7 给生产者与 collector 留调度空间；严格限定四个
物理核时传 `--cpus 0-3`，编号以本机拓扑为准。
14/30/57/89/300 的重复数为 201/101/101/51/21；大案例默认一次，可用
`--large-repeats 3` 增加重复。`--cases case14 case30 case300` 可缩小集合，
`--case-dir PATH` 可复用输入 ZIP，`--binary PATH` 可指定已构建的二进制。

终端逐案例输出中位耗时、加速比和效率；summary.csv/summary.json 保存汇总。
每档原始 JSON 含准备时间、首次端到端成本、计时样本和实际亲和性。metadata.json
记录二进制/输入哈希、CPU 拓扑、电源设置与命令；逐项校验两档场景判定和迭代数。
电源快照中的 scaling_cur_freq 是瞬时读数。已有测量目录会被保留，请为新扫描
指定新的输出目录。性能工具每次调用重建 worker，复用该进程内准备好的基态。

另外保留 bench_acpf_floor 和 n1_bevy_memory 诊断工具。

### N−1 性能记录总览（2026-10-27）

完整机器可读记录见 [performance-history.json](../tools/n1/results/performance-history.json)，
包括 AMD 各轮结果、Intel 跨规模数据、功耗来源、旧实验和时间边界。
以下现代9241记录均为16049候选、14384数值求解、1665解列、10数值失败。
执行时间包含分类、任务发射、worker初始化、PF、监测、收集和线程退出；共享准备
另计。端到端从模型加载前算到首批返回。能耗估算统一使用执行时间。

#### AMD AI Max 395：默认频率与SMT

初始1/2/4/8/16 worker各重复三次，CPU亲和性和二进制哈希未随终端记录提供。
32 worker为随后报告的约9.5 s。效率按本组单worker84.257541 s计算。

| Worker | 执行 s | 加速比 | 效率 |
| ---: | ---: | ---: | ---: |
| 1 | 84.257541 | 1.000× | 100.0% |
| 2 | 42.832640 | 1.967× | 98.4% |
| 4 | 22.997484 | 3.664× | 91.6% |
| 8 | 15.193549 | 5.546× | 69.3% |
| 16 | 8.559683 | 9.844× | 61.5% |
| 32 | ≈9.500000 | 8.869× | 27.7% |

32 worker使用16个物理核的SMT资源。对应满载讨论中用户报告约120 W和约4.6 GHz，
这些是观察值；各轮持续有效频率尚未提供。

#### AMD：CPU 0–7亲和性补测

| PF worker | CPU范围 | 执行 s | 重复 |
| ---: | --- | ---: | ---: |
| 4 | 0-7 | 23.181788 | 3 |
| 5 | 0-7 | 20.045501 | 1 |
| 6 | 0-7 | 17.999506 | 1 |
| 8 | 0-7 | 15.380010 | 1 |

另有四worker约23.3 s、约5.0 GHz的观察。CPU范围约束整个进程，生产者和collector
也共享此范围，每个worker仍可在范围内迁移。

#### AMD：3 GHz固定频率

基线单worker145.29 s；四worker38.483238 s；五worker约32 s，以上重复数未提供。
8–16 worker最新一组各重复三次。效率使用本组3 GHz单worker基线。

| Worker | 执行 s | 加速比 | 效率 | 重复 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | ≈145.290000 | 1.000× | 100.0% | 未提供 |
| 4 | 38.483238 | 3.775× | 94.4% | 未提供 |
| 5 | ≈32.000000 | 4.540× | 90.8% | 未提供 |
| 8 | 19.980268 | 7.272× | 90.9% | 3 |
| 10 | 16.842420 | 8.626× | 86.3% | 3 |
| 12 | 13.859751 | 10.483× | 87.4% | 3 |
| 14 | 12.140084 | 11.968× | 85.5% | 3 |
| 16 | 11.087117 | 13.104× | 81.9% | 3 |

功耗：四worker约26 W（修正早先约30 W估计）；10–12 worker约48–50 W，14 worker
约60 W；先前16 worker约11.3 s的一轮也报告约60 W。旧的四worker约38 s、16 worker
约11.3 s在历史记录中保留，最新精确时间用于主表。
原始8–16 worker终端行见 [ai-max395-3ghz.json](../tools/n1/results/ai-max395-3ghz.json)。

#### Intel 288V：历史四P核规模扫描

共享基态AC、batch=8、StatusOnly空约束规则、max_iter30、tol1e-10；CPU0–3。
这些数据早于结果电压clone移除；各案例逐场景判定在1/2/4 worker间一致。

| 母线 | 候选/求解 | 重复 | 1 worker ms | 2 worker ms | 4 worker ms | 四worker效率 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 20/19 | 201 | 0.242 | 0.194 | 0.281 | 21.5% |
| 30 | 41/38 | 101 | 0.530 | 0.355 | 0.371 | 35.7% |
| 57 | 80/79 | 101 | 2.776 | 1.693 | 1.365 | 50.8% |
| 89 | 210/194 | 51 | 8.413 | 4.651 | 2.779 | 75.7% |
| 300 | 411/322 | 21 | 45.371 | 24.001 | 13.899 | 81.6% |
| 1354 | 1991/1430 | 3 | 800.974 | 409.649 | 230.846 | 86.7% |
| 2869 | 4582/3804 | 3 | 5866.724 | 3000.912 | 1684.674 | 87.1% |
| 9241 | 16049/14384 | 1 | 92753.212 | 48054.057 | 28259.399 | 82.1% |

6515 RTE另测9037候选、6474数值求解、2563解列、2数值失败，21385次AC尝试；
1/2/4 worker各完整一次，执行25.512806/13.231136/7.722467 s，四worker效率82.6%。
14母线batch=1的201次补测：1/2/4 worker为0.295512/0.215572/0.312129 ms。

#### Intel：当前代码的四P核与全八核端到端

提交a3a4375，release默认参数。系统识别Core Ultra9 288V；CPU0–3为P核，4–7为E核。
governor=powersave、EPP=performance、Turbo开启。每档完整一次，依次执行。

| 配置 | 执行 s | 加载到返回 s | 平均使用核数 | 峰值RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| 4 worker，CPU 0-3 | 30.386957 | 30.486794 | 3.84 | 106.85 |
| 8 worker，CPU 0-7 | 21.641621 | 21.740990 | 7.48 | 129.54 |

两档逐项判定与43709次尝试迭代一致。平均使用核数为子进程user+system CPU时间
除以进程墙钟，含生产者和collector。全核端到端吞吐提高40.2%。
本轮包能量计数读取不可用；用户确认此Lunar机器约30 W，以下据此估算能耗。

#### 每批能耗粗估

功耗按用户报告近似为整轮平均值，以下为P×T估算；10–12 worker的48–50 W为
两档合并观察范围，16 worker约60 W来自先前11.3 s的一轮。

| 配置 | 执行 s | 功耗 W | 每批能耗 J |
| --- | ---: | ---: | ---: |
| Lunar全八核，当前 | 21.642 | ≈30 | ≈649 |
| AMD3GHz四worker | 38.483 | ≈26 | ≈1001 |
| AMD3GHz十worker | 16.842 | ≈48–50 | ≈808–842 |
| AMD3GHz十二worker | 13.860 | ≈48–50 | ≈665–693 |
| AMD3GHz十四worker | 12.140 | ≈60 | ≈728 |
| AMD3GHz十六worker，借用此前功耗 | 11.087 | ≈60 | ≈665 |
| AMD高频四worker，CPU0–7 | 23.182 | ≈70–72 | ≈1623–1669 |

#### 归档与后续方向

历史记录JSON另外保留：早期四P核16 worker超订阅、四worker开放两个E核、已撤回
SPSC对照。它们的初始化、监测和代码阶段单独标注。清理前报告与源码在本任务
工作区旁n1-cleanup-archive；本地完整测量在n1-scalability、n1-6515、n1-all-cores、
n1-acpf-estimate、n1-spsc-performance。当前总览同时保存准备与首次端到端字段。

GPU方向暂缓：用户报告J填充约100 μs，分解成本远高于J。GPU实验优先验证批量
稀疏factor/refactor/solve后端，再决定整体迁移。
