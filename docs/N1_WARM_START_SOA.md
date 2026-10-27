> 当前示例已撤掉变压器加载率监测，并在发射 N−1 前检查基准态线路约束。下文的百万级命中为旧监测配置的历史数据。

> 以下性能和分配数据对应 SoA 派生库接入前的版本。当前 `Soa<BranchData>` 与 `WatchHits` 的维护调整及输出接口见 [N1_USAGE.md](N1_USAGE.md)；此处保留原测量口径。

# SoA 后处理与可选初始化：完整 N−1 对照

2026-10-27。两种初始化共用当前 SoA/AVX2 后处理、系统动态 KLU、release opt-level=3。
CPU 为 Intel Core Ultra 7 288V，进程亲和性固定 CPU 0–3，batch_size=8、max_iter=50、tol=1e-10。
PEGASE9241 全部 16,049 项独立支路退出，加载率 >100% 监测与单个显式失败监测。

| 初始化 | worker | 整批 s | AC 尝试次数 | 失败 | DC 累计 s | 后处理/规则累计 s |
|---|---:|---:|---:|---:|---:|---:|
| 基准 AC | 1 | 121.724 | 54,474 | 1,426 | 0.000 | 9.954 |
| 基准 AC | 4 | 38.231 | 54,450 | 1,426 | 0.000 | 11.920 |
| 逐场景 DCPF | 4 | 65.130 | 92,979 | 1,669 | 12.942 | 11.654 |

四 worker 的整批耗时下降 41.3%，AC 尝试次数下降 41.4%。基准 AC 的单→四 worker
加速为 3.184×，并行效率 79.6%。共享准备包含主线程一次基准 DCPF+AC，耗时
28.881 ms；逐场景 DCPF 模式的共享准备为 11.145 ms。两者均从已经初始化的
主 World 开始计时。整批从 run_n1 调用前至返回后，包含线程/App 创建、生产任务、
patch、求解、后处理/规则、队列、收集者和 join，事件 JSON 落盘在整批计时外。
累计阶段时钟为 worker 阶段墙钟之和，包含调度停顿。

两种模式在 14,380 项共同收敛场景中，最严重加载率的最大绝对差为
1.40e-9 个百分点。基准 AC 额外收敛 243 项，额外失败 0 项；剩余 1,426 项均为
Newton 失败。基准 AC 的 1/4 worker 失败场景、阶段与逐场景命中总数一致。
逐场景 DCPF 模式仍为 DC 失败 1,504 项、Newton 失败 165 项。

额外通过的 243 项细分：239 项原先在 DC 分解失败，4 项原先在 Newton 分解失败；
全部旧错误均为 KLU factor/refactor status 1。其中 231 项退出后使用基准电压
即满足残差阈值，0 次 Newton 迭代返回（省去了数值分解）；另 12 项经过 13–16 次
Newton 迭代通过。因而这些统计应分别理解为初始残差通过和经过迭代收敛。
旧 Newton 失败含 121 项迭代上限、43 项分解失败和 1 项非有限线性解。

这次收益主要对应初值改善和省去逐场景 DCPF。当前四 worker DCPF 模式的
65.130 s 与旧 AoS 版 64.260 s 接近；后处理/规则累计时间也没有显示改善。
SoA/AVX2 已接通并通过数值对照，独立性能收益尚未测得。跨场景 SIMD 与批量
规则内核继续作为后续工作。

AC-only worker 的代表成功场景分配账目：数值向量 2,513,344 bytes、AC KLU
5,353,160 bytes，共约 7.50 MiB；DC 数值存储和 KLU 分配均为零。此账目排除
主 World、App、线程栈、队列、批输出和分配器保留页面，区别于进程 RSS。

正确性检查：主库 89 项通过、2 项忽略；23 项集成测试通过（包括最新 rank-1
矩阵及 RHS 恒等式对独立重建的验证）。旧 ECS 后处理、Newton 和 KLU 源码保持原样。

## 未退出设备的基准态安全规则结果

单独对原始基准 AC 电压调用 BranchCatalog.compute(outage=None)，按当前额定值与
电流加载率口径，16,049 个支路中有 516 个变压器加载率超过 100%，线路越限为 0。
最大为 Transformer 820 的 1732.489875572674%。零步场景的 516 个规则命中因此
继承基准态已有越限。当前测试直接使用原始工况；后续安全报告应分别列出基准已有、
故障新增和故障加重越限。原始结果：任务目录 n1-warm-soa/baseline-security.json。

## 后续拓扑复核：桥边退出混入残差通过统计

读取原始 PEGASE9241 在役线路/变压器构造无向多重图，按设备边编号处理并联边，
一次 DFS 桥边分析得到 1,665 条桥边（全部设备边为 16,049）。此为离线诊断，
运行时任务发射规则尚未修改。

- 旧 DC 分解失败 1,504 项，全部为桥边退出。
- 旧 Newton 数值分解失败 43 项，全部为桥边退出。
- 基准 AC 额外残差通过的 243 项，也全部为桥边退出。
- 其中 231 个零步场景全部具有度为 1 的空末端母线，该末端没有在役 load、
  sgen、gen、shunt、ext_grid；退出后残差检查跳过了分解。
- 其余 12 项迭代通过仍是切岛场景。旧 DCPF 模式也有 5 项桥边退出记为通过。

因此前文的“额外收敛”是当前残差终止条件的通过数。按用户此前 N−1 采用保持
主网连通的候选定义，应将切岛候选单独标记后再统计有效候选的收敛与性能。
桥边标记可以在主线程准备阶段一次生成，供候选发射使用；本轮只完成诊断。
原始统计见任务目录 `n1-warm-soa/topology-analysis.json`。

## 入口与复现

```rust
let prepared = prepare_n1(world)?; // N1Initialization::BaseAc
// 或：
let prepared = prepare_n1_with_initialization(world, N1Initialization::DcPerCase)?;
```

```sh
cargo build --offline --release --features experimental_n1 --example bench_n1
taskset -c 0-3 target/release/examples/bench_n1 9241 1,4 8 /tmp/n1-base-ac base-ac
taskset -c 0-3 target/release/examples/bench_n1 9241 4 8 /tmp/n1-dc dc
```

原始结果位于任务目录 `n1-warm-soa/`：两模式 case39/case9241 汇总、每场景事件摘要、
进程资源统计、comparison.json、allocation-ledger.json 和二进制/源码摘要 metadata.json。
集成日志为 `n1-warm-soa-tests.log` 与 `n1-warm-soa-patch-final.log`。
rank-1 DCPF 数值求解作为后续研究项；本轮只验证其矩阵和 RHS 恒等式。
