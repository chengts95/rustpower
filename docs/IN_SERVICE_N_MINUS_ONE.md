> 最新实施边界见 [N_MINUS_ONE_IMPLEMENTATION_PLAN.md](N_MINUS_ONE_IMPLEMENTATION_PLAN.md)：
> 场景投递时解析偏移，地址缓存可选；以下固定地址表描述属于前期方案。

# in_service 与 N−1 数值更新：源码核对及复用验证

当前检查目录基于 main `3eedbb1` 加既有 0.5.5 读取修复。当前任务只研究
`in_service` 的增量路径；用户明确要求省去孤岛可达性预检查，求解失败后再诊断。

## 当前调用链

1. `src/python/handles.rs::set_entity_in_service` 修改 `OutOfService`，无条件发送
   `FullRebuildEvent`。重复设置相同布尔值也会发送事件。
2. `structure_update::event_update` 设置 `full_dirty`。
3. `structure_update` 运行完整 `PFInit`。
4. `pf_init::cleanup_solver_state` 移除矩阵、求解结果、NewtonCache、DC 模型。
   初始化还重新派生支路状态、标记 PQ/PV/Slack、汇总注入并排列 Ybus。
5. `systems::create_y_bus` 的支路 Query 使用 `Without<OutOfService>`，以 COO
   重新盖章后生成 CSC，因此退出支路的独占非零位置会从结构中消失。
6. `reset_solvers` 调用旧 KLU reset，释放 Symbolic 和 Numeric。

实际 case14 线路 0（母线 0—1）退出及恢复，Ybus 条目数 54→52→54，
消失的位置为 (0,1)、(1,0)，两次报告均为 full rebuild。恢复后数值与基准最大
差约 2e-15。记录：`../main-in-service-trace.json`（任务目录中的文件）。

把事件改为现有 `admit_dirty` 仍会进入 `reset_solvers`，同样释放符号分析。
需要独立表达“结构保持的导纳数值变更”。

## 现有 KLU 已具备所需机制

`src/basic/solver/klu.rs::solve` 仅在 Symbolic 为空时调用分析；已有 Numeric 时
先尝试 refactor，失败时用现有 Symbolic 重新 factor。NR 接口与 KLU API 可沿用。
当前性能障碍在上游无条件重建/重置，以及稀疏位置变化。

## 固定结构的可行性验证

新增 `tests/n_minus_one_symbolic.rs`，用 IEEE39 线路 0 验证：

- 基准完整求解后保存 KLU Symbolic 指针和 Jacobian 列指针缓存地址。
- 用现有完整重建路径生成退出线路后的参考 Ybus。
- 将参考数值映射回基准 CSC，保留原 indptr/indices，把消失位置写成显式零。
- 同一个 KLU/NR 缓存求解退出场景，电压与独立普通求解的最大复数误差 <1e-9。
- 恢复基准数值后重新求解，电压恢复到基准，误差 <1e-9。
- 退出及恢复后 Symbolic 指针、Jacobian 列指针缓存地址均保持相同。

执行且通过：

```sh
cargo test --offline --features klu_dyn --test n_minus_one_symbolic
```

这是可行性测试，采用参考矩阵回填数值。生产 `in_service` 目前仍走完整重建；
本测试未实现增量盖章或测量性能。

## 数据流的具体改动边界

首批对象是已准备的线路、两绕组变压器的整设备退出/恢复，保持母线映射及
PQ/PV/Slack 分区。新增删除设备、母线合并/拆分及分区变化继续使用原结构路径。

- Gateway 保留 `OutOfService` 的即时可见性，过滤同值设置；将支路标记为
  待处理的数值变更。消费时对比“已盖章状态”和最终服务状态，合并同帧反复开关。
- 基准准备时记录每条支路的实际有效 2×2 导纳贡献，以及 OriginalYBus 和排列后
  Ybus 中四个值槽的位置。已有端点开关时采用实际处理后的 patch 与正确基值。
- 退出时减去该设备贡献，恢复时恢复对应贡献；并联支路共享值槽，需保留其他
  设备的累加贡献。稀疏位置与结构对角位置保留。
- 两份 Ybus、求解结果有效性和设备服务状态同步；已有后处理会把退出设备结果
  置零。Standalone DC/DCPF 的数值模型及变压器移相注入也要同步更新。
- 保留 AC 的 Symbolic、J 结构缓存和可用 Numeric，让现有 KLU 路径执行 refactor。
  refactor/factor/NR 失败以失败结果返回；下一场景恢复数据后继续。
- 每个 batch worker 独立持有可变数值状态和求解器，在自己的连续场景中复用符号。

默认 `qlim=false` 适合首批固定分区验证；开启 Q-limit 后实际 PV→PQ 会改变 J
分区，需要沿现有结构失效路径处理。已有 `BranchAnalysisRes` 仅保存占位数据，
尚无可直接使用的设备→Ybus 值槽映射。

## DC/AC Symbolic 的跨线程共享验证

本机 `/usr/include/suitesparse/klu.h` 的 factor/refactor 输入注释将 CSC、数值
输入和 Symbolic 列为不修改；Numeric、Common 及 RHS 的可变状态分别由 worker
持有。旧包装的析构会同时释放 Symbolic/Numeric，因此借用共享 Symbolic 时
需要单独管理释放职责。

`tests/n_minus_one_shared_symbolic.rs` 使用 IEEE39 派生的真实 DC 与 AC 稀疏模式，
各分析一次。四个 scoped 线程借用相同的两个 Symbolic 和两套 CSC 索引数组，
每个线程为每套结构执行 24 轮 factor/refactor/solve，总计 192 轮。数值是根据
模式构造的对角占优线性系统，已知解为全 1；用于验证共享边界，区别于完整
DCPF 初始化加 Newton 潮流的并发测试。全部解误差 <1e-10，指针保持同一份。
主线程等待所有 worker 及 Numeric 析构后统一释放两个 Symbolic。

```sh
cargo test --offline --features klu_dyn --test n_minus_one_shared_symbolic
```

测试通过。实现采用借用和主线程限定生命周期，未使用 Arc、锁或每 worker 的
符号拷贝。索引在发布前统一转换为 KLU ABI 的 i64，一份模式供所有 worker 读取。
测试中的小型析构适配器仅释放自己的 Numeric，并将借入的 Symbolic 指针置空后
交回旧析构逻辑；旧 KLU API/源码保持原样。

完整 N−1 插件应把共享部分显式拆出：

- 主系统拥有：DC pattern/Symbolic，AC J pattern/Symbolic，J 的分段边界和
  对角索引，固定 PQ/PV/Slack 映射及基准支路贡献。
- worker 资源拥有：Ybus/J/B 数值，电压/注入/RHS，DC 与 AC 各自的
  Numeric/Common，以及本次任务/结果。
- DC 和 AC 各有自己的 Symbolic 对象；同一种对象在 worker 间共享。
- 使用只读的固定模式求解入口。旧 `Solve` 的 CSC 参数是 `&mut [usize]`，
  旧 NR 也把 J 索引转换成可变切片；共享 Rust 结构数组时需在专用入口消除
  这类可变引用。FFI 的历史非 const 指针只在边界进行转换。
- worker App 无须装设备实体；message 输入场景，DC 初始化系统与 NR 系统
  消费共享结构和本地数值资源。禁用 Q-limit 保证首批场景分区固定。

这是专用入口/插件的设计及两项底层验证，生产 N−1 插件尚未接入。
