> 真实退出任务与实体触发器已接入，最新接口见 [N1_USAGE.md](N1_USAGE.md)。
> 本文保留基础框架的阶段记录；参考退出 fixture 已补装结构更新插件并重跑。

# N−1 基础框架：共享符号、worker App 与后处理

本轮由布洛妮娅实现共享数值求解核心，重装小兔实现只读支路目录及后处理，主助手
接入 App 插件、队列和集成测试。实现位于 main 隔离目录，默认 feature 保持原样。

## 可运行入口

```sh
cargo run --offline --features experimental_n1 --example n1_base
cargo test --offline --features experimental_n1 --test n1_foundation --test n1_postprocess
```

示例使用 IEEE39，3 个常驻 worker App、4 组各 3 个真实 base 求解。全部 12 次
DCPF+Newton 收敛，AC 迭代 4 次，残差约 2.185e-13；每次后处理 46 条线路/变压器。
这是正确性示例，使用 dev 构建，未做吞吐或加速比较。

## 代码与所有权

- `src/n1/solver.rs`：SharedBaseline 准备 Y/J/DC 模式、J 分段信息及 DC/AC 两个
  Symbolic；N1Worker 独占数值缓冲和两套 Numeric/Common。每个 worker 读取同一
  baseline，基准模式与符号对象在 worker 间零拷贝共享。
- `src/n1/numeric.rs`：V4 数值组装的只读 CSC 切片入口，公式与原 V4 对齐。
- `src/n1/postprocess.rs`：BranchCatalog 提取一次有效 2×2 模型、solver 顺序母线、
  电流基值和额定值，所有 worker 只读共享。场景 outage 索引单独传入，命中设备
  的全部支路结果置零，其他并联设备继续正常计算，原目录保持基准服务状态。
- `src/n1/runner.rs`：prepare_base 生成绑定的 PreparedBase（求解基准+支路目录），
  运行器只接收这一配对对象。scoped 线程内部创建 App，配置资源、消息与两阶段
  base 求解/批量后处理插件。普通设备实体留在主 World。

数值计算沿专用只读接口进行；旧 Newton、DCPF、KLU 包装和原 API 源码保持原样。
基准快照在准备时复制一次主模型，worker 初始化复制私有数值数组；此处的零拷贝
指 worker 间的结构、符号和后处理模型共享，不包含输出数组及必要的私有数值状态。

## 当前任务和结果流

BaseBatch 包含 1..32 个 BaseTask；任务与结果队列容量分别为 2×worker 数。
生产者可以逐组生成任务，无需一次保存全部场景。每个线程只领取一组并连续求解，
然后完成组内后处理。队列承担 SPMC 投递与 MPSC 回传职责，收集者独立消费。

BaseTask.emit 是本阶段显式的“总是/从不输出”诊断规则：emit=false 时不做支路
后处理、完全不写结果队列，processed/failed 仍计数。后续实体规则可替换该诊断层。

producer_done 使用 Release/Acquire；worker 先读取结束标记，再取任务。空闲先
短暂 yield，持续空闲后 sleep；收集者在全部结果发送端关闭后排空并结束。join
保证 App 与 Numeric/Common 已销毁，主系统随后释放共享结构和 Symbolic。

## 测试结果

新框架测试 6 项通过：

1. IEEE39 加入 7° 变压器移相，3 个 worker App 处理 36 次真实 DCPF+Newton。
   电压与原 ECS 求解复数误差 <1e-8；线路/变压器功率、电流、加载率误差 <1e-6。
   所有结果报告相同 DC/AC Symbolic 与模式地址，共享基准保持原样。
2. 空输入、32 上限、静默输出、正常—迭代失败—正常的连续场景和独立完成计数。
3. IEEE39 线路 0 实际退出：将旧完整重建的参考数值映射回固定结构，DC+NR 求解
   电压匹配原路径，后处理退出掩码输出零；DC 奇异失败及错误缓冲后可恢复 base。
4. 并联设备身份、单个退出掩码、原本停运设备和再次 base 后处理。
5. 合并母线与 solver 顺序映射，以及输入尺寸/索引/基值错误。
6. 变压器电流基值、并联额定和与主线一致的 current-based 加载率。

已有固定模式退出/恢复测试、4 线程共享符号线性验证各 1 项也通过，共 8 项集成
验证。主库回归 87 项通过、2 项忽略。

本轮命令日志：
`/home/cts/Documents/Codex/2026-10-26/task/n1-foundation-tests.log`、
`n1-foundation-lib-tests.log`、`n1-base-example.log`。

## 下一层实施边界

- BaseTask 当前运行 base；实际退出任务在测试中通过参考数值回填验证。COO→CSC
  直接偏移描述符、生产级 O(1) Y/B/RHS patch 尚待接入队列任务。
- 后处理当前支持普通两端口线路/变压器；启用的端点开关特殊模型返回明确错误。
- 数据驱动实体 WatchPlan、条件触发、按字段提取及按输出字节限制分段尚待实现。
- 此实验模块固定分区、无 Q-limit、无孤岛可达性预检查。数值失败返回状态，worker
  继续后续任务。正常结果与失败输出是否回传由显式诊断 emit 配置决定。

完整设计仍以 [N_MINUS_ONE_IMPLEMENTATION_PLAN.md](N_MINUS_ONE_IMPLEMENTATION_PLAN.md)
为准。本轮完成共享 base→worker 计算→后处理的基础闭环。

## 后续设计记录

已有 AVX 后处理内核的复用、小型实体身份关联 SoA、SIMD tile 与 4/8/16 动态
场景 batch、批量触发器命中掩码，已记录到实施计划。当前优先推进真实 N−1
退出任务流，base 示例保持其基础验证用途。
