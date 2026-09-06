# LM完整性能报告

 

包含八条实现、四级消融、KLU/QDLDL后端比较、全部计时分项、算法参数、迭代次数和680次逐轮记录；CSV位于 `/home/cts/pp/data/`。

在仓库根目录运行 `python3 performance/python/report.py --output-root ~/pp`，同时更新Markdown、CSV、论文表格并把完整报告打印到stdout。

性能入口 `lm-assembly`、`lm-ablation` 也会直接打印每次运行的计时分项及每算例汇总。

## 性能入口

- `lm.rs`：LM方法配置、调用和正确性检查；A0不含V4，KLU/Cholesky后端适配保留在`linear_solvers.rs`。
- `opf.rs`：V1、V4、V5.0、V5.2、V5.3、V5.5、V5.6共用一份调用和计时。`audit_opf.rs`复用这里的版本选择。
- `bench.rs`：`timeit!`返回结果和毫秒数；统一预热/轮次顺序、JSON保存、逐轮输出及中位数/范围汇总。只计完整调用，内部阶段仍读取solver已有计时器。

在仓库根目录运行：

```bash
cargo bench --locked --features benchmark --bench comparison -- lm-assembly
cargo bench --locked --features benchmark --bench comparison -- lm-ablation
cargo bench --locked --features benchmark --bench comparison -- lm-solvers --klu
cargo bench --locked --features benchmark --bench comparison -- opf
```

默认预热1次、测量7次。可加`--case IEEE39 --repeats 2`快速检查；需要性能数字时请固定CPU，并保持相同线程设置。`opf-assembly`是`opf`的兼容入口，`opf-v4-v5`只选择V4与V5.0，均复用同一循环。

指定`--case`或`--repeats`时，默认写入该入口的`checks/`子目录，保留正式完整测量；显式设置输出目录环境变量时以该目录为准。

每个入口向stdout打印参数、状态、迭代/重试计数、全部已有分项计时和汇总；JSON与环境信息写到`target/research/performance/<入口>/`。时间汇总排除预热，失败运行保留，不以失败耗时计算加速比。

OPF入口使用原生CSV网络与`warm_x0`，不是论文中pandapower导出的同模型初值；同模型审计仍用`audit_opf`。OPF的“首次/后续求解区域”含包装工作，不是纯符号/数值分解，KLU计时器另列。完整调用包含模型缓存构造和首次分解。
