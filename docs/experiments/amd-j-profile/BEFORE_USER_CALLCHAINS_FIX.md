# 原版 J→AMD：范围纠正与交付状态

此前FullTableControl报告测错对象，不能回答用户要求。它从Ybus准备输入；本次改为原版接受J的标量函数，未改算法。

实际调用：

```
预先构造好的 J CSC (j_col_ptrs, j_row_indices)
  -> rustpower_amd::order_unchecked (lib.rs)
  -> validate_csc / order_impl
  -> valid / aat
  -> amd_1 (构造 A+Aᵀ 工作图)
  -> amd_2::<usize,true,false> (seed=None, sharing=None)
  -> postorder
  -> 排列 / 逆排列
```

这次采样区不包含Ybus分析、YbusPattern、FullTableControl、预合并或按需实验路径，也不包含J组装。接受的是J结构，数值不参与AMD。使用当前Rust移植中的原版标量入口、unchecked工作区索引，未启用任何分组/启发式。

## 已完成

- `examples/profile_j_amd.rs`：先准备案例和J，100次预热；只对上述原版函数重复调用，结果black_box消费，输出在测量区外。
- CPU0，Ultra9 288V的P核；性能profile/EPP，governor标签powersave；环境清单已保存。
- AMD和测试入口分别编译frame-pointers=no/yes两份，opt3、debug2、overflow-checks=no、LTO关闭、无target-cpu=native。其余依赖复用已有release产物，避免重编整个项目。原库源文件未修改。
- 39/118/9241，两个独立批次，每批每版本400次，交替版本顺序。每批最后验证排列、逆排列、lnz与同一原版函数的参考结果一致。首次函数调用100次预热，不称缓存冷启动；此API每次重新分配工作区，不虚构复用API。

原版函数耗时，单位µs，两个批次中位数：

|案例|帧指针|批1|批2|
|---|---|---:|---:|
|9241|no|3033.694|3038.708|
|9241|yes|3064.064|3037.386|
|39|no|3.133|2.690|
|39|yes|3.058|3.209|
|118|no|8.681|9.005|
|118|yes|8.454|8.576|

全部逐次数据、二进制、源码快照和哈希保存在本目录的 `results-20261026-205936/`。

## 具体阻碍：调用栈采样被拒绝

执行目标为CPU0上、接受J的原版AMD帧指针版本，命令核心为：

```
perf record -e cpu-clock:u -F 1999 --call-graph fp --delay=-1 --control=fifo:... --sample-cpu -- profile_j_amd 9241 2500 ...
```

内核返回 `Failure to open event 'cpu-clock:u' on PMU 'software'`、权限限制，`perf_event_paranoid=2`。原文在结果目录 `j-amd-1.stderr`。

状态：**denied**，未获得调用栈样本，未生成火焰图；没有把旧Ybus实验采样或平面热点冒充本次结果。没有更改系统权限或绕过限制。帧指针方案没有解决此执行环境的栈采样权限问题。先前不带调用栈的采样成功，不证明带栈的采样被允许。

用户之前已表示可手动运行profiling。可在本机终端直接执行：

```bash
cd /home/cts/workspace/rustpower-amd-supervariable
bash docs/experiments/amd-j-profile/run.sh
```

会使用已编译的上述二进制，输出两批真实调用栈perf.data、自占/包含占比文本、源码反汇编注释、折叠栈及 `j-amd-1-flame.svg` / `j-amd-2-flame.svg`。如果本机终端也被拒绝，脚本会保存错误并停止，不生成伪图。

SVG绘制脚本从perf的stackcollapse输出生成标准按样本数宽度的调用栈火焰图；没有样本会明确拒绝绘制。目前它尚无实际栈数据可渲染。`build.sh`记录了可复跑的精确rustc命令，`run.sh`负责计时与采样。

这次只修正测量对象和诊断工具，没有新增排序算法，没有修改默认配置，没有提交或推送。下一步是读取用户本机实际采集到的上述调用栈，而不是继续沿用此前错误入口的百分比。
