# 原版接受J的AMD：火焰图结果

状态：已完成两批真实用户态调用栈采样，并生成SVG火焰图。此前权限失败已定位并修正：perf命令遗漏`--user-callchains`，仍请求内核调用栈；加上该参数后成功，未更改系统权限。前一份测Ybus实验入口的报告不用于本结论。

## 查看火焰图

- [第一批](results-20261026-205936/j-amd-1-flame.svg)
- [第二批](results-20261026-205936/j-amd-2-flame.svg)

测量对象是当前Rust移植中原版标量入口`rustpower_amd::order_unchecked`，以已构造好的J CSC为输入，未启用预分组、共享、按需或Ybus专用路径。调用链：

```
J CSC -> validate_csc / order_impl -> valid -> aat
      -> amd_1 -> amd_2::<usize,true,false> (seed=None,sharing=None)
               -> postorder -> 排列、逆排列
```

案例9241：J维度17036、存储非零129412。J组装、网络加载、参考结果及100次预热均在测量区外；区域内只重复原版排序函数，输出缓冲/序列化在区域外。每批2500次，固定CPU0 P核；CPU-clock用户态1999Hz。

## 函数级热点（完整AMD调用为分母）

|函数/工作|批1 self|批2 self|批1 inclusive|批2 inclusive|
|---|---:|---:|---:|---:|
|amd_2：消元、近似度更新和合并主循环|66.34%|65.86%|74.65%|74.14%|
|amd_1：构造A+Aᵀ工作邻接|8.88%|9.07%|84.33%|见原始inclusive表|
|postorder：后序输出|7.87%|7.76%|7.87%|7.76%|
|aat：对称结构计数|6.74%|6.53%|6.74%|6.53%|
|valid：检查输入J的CSC结构|5.33%|5.44%|5.33%|5.44%|

self是函数自身样本；inclusive包括子调用，因此amd_1、amd_2、postorder的inclusive相互嵌套，不能相加。这里的valid检查的是输入J，不是之前误测的YbusPattern校验。

## amd_2里面具体慢在哪里

源码/反汇编在两份`j-amd-*-annotate.txt`。下面通过样本指令地址的内联调用位置映射回源码段；分母仍是完整原版AMD调用，不是只在amd_2内归一化。

|操作|批1 self|批2 self|源码|
|---|---:|---:|---|
|逐个邻居取Nv、筛掉失效/当前元素内变量、累计度数与哈希、写回存活邻居|12.74%|12.96%|amd_2.rs:653–690|
|Scan1：更新相邻消元元素的外部度|7.99%|8.06%|amd_2.rs:525–574|
|度数上界处理、加入哈希桶|7.74%|7.42%|amd_2.rs:692–813|
|构造当前消元元素、处理工作邻接存储|6.41%|6.17%|amd_2.rs:281–524|
|超变量比较、合并|3.78%|3.85%|amd_2.rs:814–932|
|Scan2：筛选/吸收旧消元元素|3.29%|3.26%|amd_2.rs:577–652|
|恢复度数桶|2.67%|2.87%|amd_2.rs:933–998|

此外13.72%/13.71%样本确定在amd_2，但优化后行号为0，未擅自归到某个循环。这意味着上表是**已能归属的热点**，不是对全部内核成本的完整分解。落在`Nv[j]>0`附近的样本不能全解释成这条比较指令的延迟；循环还有间接读取、筛选、累计和写回。细分结果保存在`operations-summary.json`和各批operations.json。

## 正式计时与构建影响

用户重跑后保存的9241原版函数时间，两批各400次中位数：

|构建|批1|批2|
|---|---:|---:|
|正常省略帧指针|2.852429ms|2.854628ms|
|为火焰图保留帧指针|2.848637ms|2.858585ms|

本批帧指针版本与普通版本的中位数接近（约-0.13%/+0.14%）。保留帧指针的AMD和测试入口均为opt3、debug2、overflow-checks=no、无LTO/target-cpu=native；未对所有循环强制禁止内联。原始源码/二进制SHA256、精确rustc命令及案例39/118时间也已保存。

两批采样期间函数中位数为3.069750ms/3.060681ms，比上述正式时间高约7%；这是观察到的跨运行差异，包含采样及运行状态影响，不用采样时间替代正式时间。

## 栈质量与数据边界

- 两批分别15364/15328样本，perf报告Lost Samples均为0。
- 第一批15343/15364样本包含driver main，15342包含AMD函数，15197包含标量order_unchecked入口。内联函数会出现在展开后的调用链中。
- 原始FP栈在C main以上的libc启动端出现未知/不可信地址。原始perf.data和folded完整保留；SVG仅裁去C main以上的启动帧，保留AMD到测试main的调用链，不改变样本数。
- perf对libc调试文件有`could not read first record`警告，保存在collect.log；AMD带调试信息，能映射到函数和源码。没有把未解析位置分摊到热点。
- SVG是实际调用栈按样本数宽度绘制的火焰图；此前的平面函数图未当作本次图使用。已渲染PNG检查图形。
- `perf-attributes.txt`确认`exclude_kernel=1, exclude_callchain_kernel=1`；没有请求内核调用栈。
- CPU0属于cpu_core的0–3集合；电源profile/EPP为performance，governor标签powersave。具体环境见environment.txt，结束快照见power-after.txt。

## 可复跑

二进制已在持久结果目录，重复采集即可：

```bash
cd /home/cts/workspace/rustpower-amd-supervariable
bash docs/experiments/amd-j-profile/collect.sh
```

`run.sh`会先重测正式耗时再调用collect；`build.sh`重新构建两份诊断二进制。数据均在`results-20261026-205936/`。本次仅完成正确对象的诊断，未修改AMD算法、生产默认路径，未提交推送。
