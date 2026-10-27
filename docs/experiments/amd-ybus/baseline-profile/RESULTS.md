> 范围纠正：本文测的是实验用FullTableControl，从Ybus准备输入。它不是用户要求的原版J→AMD火焰图，不作为该问题的结论。正确测量另见../../amd-j-profile/。

# 完整表 AMD 性能诊断（2026-10-26）

## 先看结论

1. 9241 完整排序入口的主要热点是 **Ybus 校验/列信息准备约17.2%**、**AMD Scan2 中的变量邻居筛选/度数与哈希累计约15.5–15.7%**、**完整初始邻接表准备约9.0%**。这是互不重叠的源码采样分组，占完整入口的 self 样本比例；不是对嵌套调用百分比求和。
2. 完整表准备另测约 **0.295 ms**，完整入口约 **3.20 ms**。原先希望省去的这一步约占9%；将它省掉也还要支付替代路径的成本。
3. 按需版每次约多执行 **391万条指令（约19.6%）**。分支错误反而少约13%，硬件计数并不支持“只是分支预测变差”的解释。按需版 L1/L2 未命中的绝对数也有所增加。没有测出内存带宽饱和或每条加载的等待时间，不能把剩余差异全部归因于缓存。
4. 同口径正式耗时中，按需版9241首次分配慢约2.5–2.6%，复用慢约3.1–3.3%。它省了初始表的写入，却仍需生成全部112376个初始邻居，并承担生成/列表状态与动态写入管理的工作。
5. 下一步只建议测清 **Scan2存活邻居筛选这一段的重复访问和指令构成**，再决定怎样减少工作；该段目前约占总入口16%，这只是优化目标范围，不是预计已经得到的加速。本阶段不实施新算法。

## 对照与环境

主基线是 `YbusPattern::analyze → FullTableControl::order → amd_2::<usize,true,false> → postorder → 输出排列/逆排列`。不是已认可的PQ预合并路径，也不是系统C AMD。新路径为 `YbusPattern::analyze → on_demand::Workspace::order → lazy_core::amd_lazy`。两者singleton初始Nv=1，相同默认Control，数值J组装及PQ/PV/Slack顺序未改。

- 持久工作树 `/home/cts/workspace/rustpower-amd-supervariable`，base `1ad70a7a5834dd6d93c5c967115beb63e853f91b`；基于未提交的已有实验代码。生产原仓库未改，未提交推送。
- Intel Ultra 9 288V，CPU0。`/sys/bus/event_source/devices/cpu_core/cpus=0-3`、`cpu_atom/cpus=4-7`，由此确认CPU0为P核；进程固定CPU0，开始/结束检查CPU编号。
- CPU0 L1d48KiB、L2 2.5MiB、L3 12MiB（后者CPU0-3共享）。容量是拓扑记录，不能据此断言数组全部命中缓存。
- Linux7.3.0-2-cachyos-rt-bore-lto；perf7.2.11-1；rustc1.99.1/LLVM23.1.3。
- opt-level3，release overflow-checks=false，debug=1（行号/有限调试信息），LTO未启用（Cargo默认false），默认x86_64目标，无额外RUSTFLAGS、无frame-pointer改动，trace/详细计数关闭。
- 电源profile=performance、EPP=performance、scaling_governor标签=powersave；前后快照相同。governor标签和电源profile分别记录。
- 原始结果 `results-20261026-204212/`。正式二进制SHA256 `7273626423d26dec37286cf9e78949597e93050f14455f9e68467b4032311943`；阶段计时二进制 `95bfade95e577bb7244de02bcb83dacc1dd812dab9662fd7666583b11d47c872`。各AMD源码哈希见该目录 `source-binary.sha256`。

## 计时边界

案例先加载并完成原有Ybus预置换。每次测量包含YbusPattern校验/列信息、工作区准备、初始表（基线）、排序、后序、生成两个排列和局部清理；输出排列保留至计时结束。案例解析、网络初始化、参考排列、100次预热、JSON序列化/写文件均在测量区外。每次重新计算排列，不复用旧排列。首次分配不是CPU缓存冷启动；复用指工作区数组容量复用。

正式时间每个case/模式/实现为两批各300次，第二批交换实现顺序；分批顺序运行，未让两组基准争抢CPU。结果被black_box消费，批末检查排列/逆排列/lnz与完整表对照一致。

perf以FIFO一次开启/关闭整批区域，排除解析和预热。CPU-clock平面采样每批2500次、1999Hz。硬件计数另跑，每事件组每批400次，2批，相同核心/输入/配置。细计数与正式时间分离。阶段计时又用独立feature构建、每格300次，不把插桩时间当正式时间。

## 正式时间

以下单位µs；每格给出两批中位数。每格600次的完整逐次数据及每批P10/P90见 `summary.json`。

|案例|模式|完整表：批1 / 批2|按需：批1 / 批2|
|---|---|---:|---:|
|39|首次分配|2.765 / 2.715|3.705 / 3.446|
|39|复用|3.028 / 2.707|3.434 / 3.417|
|118|首次分配|8.748 / 10.093|10.975 / 10.873|
|118|复用|8.716 / 9.091|10.910 / 11.983|
|9241|首次分配|3200.634 / 3204.339|3284.599 / 3283.817|
|9241|复用|3172.250 / 3169.962|3271.618 / 3273.438|

9241首次完整表P10–P90：3087–3434、3116–3416µs；按需3180–3507、3169–3501µs。复用完整表3076–3325、3066–3326µs；按需3174–3457、3177–3441µs。分布有重叠，报告小幅变慢，不声称大幅差距。小案例波动更明显。

### 独立粗阶段计时（完整表，中位数µs）

|案例/模式|Ybus分析|输入缓冲resize|初始邻接建表|AMD内核含后序|
|---|---:|---:|---:|---:|
|39首次|0.593|0.051|0.353|1.789|
|39复用|0.686|0.014|0.419|2.226|
|118首次|2.175|0.115|0.891|6.748|
|118复用|2.300|0.014|0.929|6.046|
|9241首次|587.537|27.660|294.871|2323.833|
|9241复用|596.551|0.015|305.611|2368.183|

这些是单独诊断构建的阶段中位数，各项中位数之和不等于总耗时中位数。resize只覆盖输入数组；amd_2内部辅助数组分配/初始化仍包含在内核列，不能把27.660µs当作全部分配成本。阶段边界在 `on_demand.rs::order_profiled`。

## 到具体源码的热点

平面采样只能给self，不提供真实调用链inclusive。下表是源代码位置归类，不用虚构inclusive。两批复用数据的self百分比，分母是完整入口的CPU-clock样本。

|实际工作|批1 / 批2 self|源码|
|---|---:|---|
|校验Ybus并提取列边界|17.23 / 17.21%|`powerflow.rs:31`、`:265`|
|扫描变量邻接，检查Nv，累加度数/哈希，写回存活邻居|15.73 / 15.53%|`amd_2.rs:653–690`|
|完整表入口中输入准备/建表|9.00 / 9.08%|`on_demand.rs:324–350`|
|Scan1：更新已有消元元素的外部度|7.95 / 7.80%|`amd_2.rs:525–574`|
|后序遍历|7.57 / 7.71%|`postorder.rs`、`post_tree.rs`|
|度数上界/哈希桶插入|7.70 / 7.01%|`amd_2.rs:692–813`|
|构造当前消元元素，包括arena处理|7.40 / 7.17%|`amd_2.rs:281–524`|
|超变量比较/合并|4.00 / 4.20%|`amd_2.rs:814–932`|
|Scan2：消元元素筛选/吸收|3.20 / 4.04%|`amd_2.rs:577–652`|
|恢复度数桶|3.00 / 3.01%|`amd_2.rs:933–998`|
|选择pivot/摘除度数链表|0.78 / 0.79%|`amd_2.rs:211–280`|

另外约6.1–6.3%样本在amd_2内但调试行号为0，约4.1%属未归入上述块的库函数/指令。没有把它们按比例分摊。详细源码行、函数、反汇编保存在 `flat-*-annotate.txt`、`flat-*-operations.json` 和 `operations-summary.json`。

源码行680附近 `Nv[j]>0` 的判断获得较多样本，但采样存在偏移，不能把落在这条指令的样本全解释为该比较本身的延迟。邻居循环包含多次间接读取、筛选、累计和写回，需作为整体理解。

## 硬件计数（9241，复用，每次排序，两批均值）

|计数|完整表|按需|
|---|---:|---:|
|cycles|14,498,460|14,881,692|
|instructions|19,937,146|23,844,228|
|IPC|1.375|1.602|
|branches|4,282,099|4,573,115|
|branch-misses|322,792|281,755|
|branch-misses / branches|7.54%|6.16%|
|mem_load_retired.l1_hit|4,805,964.9|7,192,477.7|
|mem_load_retired.l1_miss|234,003.5|257,405.1|
|mem_load_retired.l2_hit|227,097.8|243,189.9|
|mem_load_retired.l2_miss|6,279.8|13,697.6|
|mem_load_retired.l3_hit|6,559.4|12,683.5|
|mem_load_retired.l3_miss|43.0|44.3|
|dtlb_load_misses.stlb_hit|465,724.8|511,800.1|
|dtlb_load_misses.walk_completed|69.6|98.9|

事件均实际使用 `cpu_core/<原名>/u`，定义/encoding保存于 `perf-list-details.txt`。每组计数actual running比例100%，未发生复用缩放。L1/L2/L3项是retired-load来源/未命中事件；TLB项不是同一retired-load口径，未把它们组合成命中率。事件名称明确保留，不把通用cache-misses当L1 miss。首次分配的cycles均值：完整表14,624,931、按需14,867,567，IPC分别1.363、1.604。

本机这些数据主要表现为L1命中，以及L1未命中后由L2提供；LLC之外的未命中绝对数较少。它们支持“此重复采集不表现为大量主存miss”，不证明没有缓存命中时的依赖等待或加载吞吐瓶颈，也不能从IPC直接判断主因。没有测量实际带宽、加载use-latency和具体数组地址来源。

## 采样限制与异常记录

- 带DWARF栈的cycles采样及CPU-clock采样均被内核权限拒绝，原文见 `hardware-sampling-denied.stderr`、`software-dwarf-sampling-denied.stderr`。没有更改系统权限、没有提权。已知不带栈的CPU-clock采样可用，后续改用该测量。
- 8份平面采样报告均记载Lost Samples=0。没有采集调用栈，所以调用栈解析成功率/inclusive/fire graph不可得；`*-hotspots.svg`是明确标注的叶函数条形图，**不是火焰图**。没有伪造缺失的SVG火焰图或加载延迟结果。
- 无额外编译参数改变的平面采样，批1首次完整表中位数3.220ms，对应正式3.201ms（约+0.6%）；批1复用3.190ms对应3.172ms（约+0.6%）。批2首次采样时间与用户另一终端的编译重叠，3.645ms，不将它作为剖析开销或正式时间；复用批2已结束该编译，主要源码占比仍与批1稳定一致。
- perf软件干扰事件实际回退成`:u`范围并报告0；这不够证明没有内核调度干扰。driver检查CPU起止相同，样本记录CPU；没有把这些0当成完整系统调度统计。
- runner中途编辑了较早执行中的平面采样脚本，八份测量和报告写完后，shell读取偏移导致尾部`ne: command not found`。现盘脚本语法检查通过，8份数据各有完整driver输出/参考排列断言，分析脚本已重新处理；未重跑无关算法。
- 用户手动运行生成 `results-20261026-204644/`，编译完成，环境记录停在Git状态。原因是脚本`git status | head`在pipefail下触发SIGPIPE，测量未开始。已改成读完整输入的sed，保留该目录原状。不能称该目录已有用户手动跑出的性能结果。

## 可复跑与本次代码边界

```bash
cd /home/cts/workspace/rustpower-amd-supervariable
BENCH_CPU=0 bash docs/experiments/amd-ybus/baseline-profile/run.sh
```

输出目录打印并保存于`latest.txt`。如果用户终端支持DWARF，则保存完整调用栈perf.data及self/inclusive文本；不支持则保存拒绝原因并停止后续栈采样，继续计数/平面采样/阶段计时。平面数据可独立重读：

```bash
python3 docs/experiments/amd-ybus/baseline-profile/summarize.py docs/experiments/amd-ybus/baseline-profile/results-20261026-204212
python3 docs/experiments/amd-ybus/baseline-profile/analyze-samples.py docs/experiments/amd-ybus/baseline-profile/results-20261026-204212
```

只新增诊断example、脚本、报告和`profile_stages` feature下的粗阶段计时方法；普通`FullTableControl::order`、amd_2、lazy_core和生产默认配置未改。阶段诊断通过39/118/9241的排列/逆排列/lnz断言，正式/计数/采样每批均有相同断言。脚本语法和git diff whitespace检查通过。本阶段结束于诊断，未实现共享原型或新排序算法。
