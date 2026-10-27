# 已验证预分组 AMD 代码交接

> 当前结构更新：AMD C/构建/缓存状态已归到 `rustpower-amd`；旧 KLU、DefaultSolver、
> 原 Native/ECS 与 NR 源码恢复基线。显式使用 `AmdNewtonSolver` 或安装
> `AmdOrderingPlugin` 才启用实验加速；feature 本身只提供新接口。
> 以 `docs/ybus_amd.md` 为当前入口，以下自动接入描述属于历史记录。

> 2026-10-27 更新：`klu_ybus_amd_rust` 已接入 Native/ECS/Python，默认选择 Rust；
> `klu_ybus_amd` 单独启用时仍选择 C。两者可运行时切换，普通 KLU 也保留。
> 当前实现及端到端结果见 `docs/experiments/amd-ybus/RUST_BACKEND_INTEGRATION_RESULTS.md`。
> 以下历史交接中的“生产接入仅 C”描述对应更新前的状态。

工作树：`/home/cts/workspace/rustpower-amd-supervariable`。分支`codex/ybus-amd-supervariable`，HEAD `1ad70a7a5834dd6d93c5c967115beb63e853f91b`。修改未提交、未推送。原仓库`/home/cts/workspace/rustpower`未改。本次仅保全交接，没有新基准。

## 应保留的加速实现

C：`crates/rustpower-sol-klu/vendor/amd/amd_super2.c`是预分组AMD内核，`ybus_order.c::rustpower_ybus_analyze`从已按PQ/PV/Slack排列的Ybus建立母线邻接，PQ权重2、PV权重1，运行内核并建立KLU Symbolic。不是对完整J做普通AMD后复制排列。许可证/来源文件在同目录，需连同代码保留。

Rust：`crates/rustpower-amd/src/powerflow.rs`提供`YbusPattern::analyze`、`OrderingWorkspace::order`/`order_unchecked`；`QuotientGraph::initialize`建立较小的邻接和加权外部度，`eliminate`调用`amd_2::<usize,FAST,false>`并传seed。`src/amd_2.rs:118`初始化PQ Nv=2、PV Nv=1，保留完整标量输出编号。`src/ybus.rs::order_ybus[_unchecked]`是包含Ybus分析的完整包装入口。Rust port来源、LICENSE、PROVENANCE.md必须保留。预合并不承诺与普通标量AMD排列逐项相同。

## 生产接入和开关

实际NR/KLU生产接入目前使用上述C实现，**不是Rust版本**。调用经过`src/basic/ybus_ordering.rs::prepare_klu` → `crates/rustpower-sol-klu/src/ybus.rs::KLUSolver::prepare_ybus` → C helper。native入口在`src/native/solver.rs:369`，ECS入口在`src/basic/ecs/network.rs:147`；`Solve` trait保持原样。

根feature `klu_ybus_amd` → `klu_dyn`及下层`ybus_amd`，下层通过可选cc构建C源码。根默认features只有rsparse/arrow，**默认未启用该优化**。显式启用feature后，KLU对象默认ybus_enabled=true，可用`set_ybus_ordering(false)`关闭。

缓存键包含精确Ybus列指针/行号、npq/npv、btf/ordering；变化则清理旧Symbolic/Numeric，调用者使J缓存失效。不支持结构、非AMD ordering、分配失败等情况保留普通KLU分析回退。具体资格条件在下层ybus.rs::eligible，含排序/对角/对称/非Slack连通等检查。不得删掉这些条件却继续声称单块Symbolic已验证。

BTF注意：优化C路径直接提供已确认的一个块，不执行通用BTF；即使common.btf=1、Symbolic.do_btf=1也如此。BTF1总收益包括跳过通用BTF；BTF0两边都不执行通用BTF。

## 已有有效数据

`docs/experiments/amd-ybus/INTEGRATION_RESULTS.md`及integration-results/cpu0-20261026-085534：实际NR路径，9241 BTF1首次18.004→16.279ms；BTF0首次16.998→16.188ms。复用NR无明确加速。39/118/9241收敛迭代4/5/6，最大复数电压误差4.54e-13。

`C_SEEDED_VS_RUST_SEEDED.md`：同批修改C与Rust预分组对照，9241准备2.168→2.328ms，首次NR16.319→16.494ms；准备差约7.36%包含Symbolic适配差异，不是纯内核差。两者预分组排列一致，实测L/U一致。

最新Rust Nv=2诊断：`docs/experiments/amd-nv2-profile/RESULTS.md`和results-20261026-211505。9241同批amd_2内核2.30→1.54ms；计时取自仅增加两次时钟的源码副本，生产源码未改。3.03→1.66ms是不同入口、且Ybus分析/J索引生成均在计时外，**不能称为首次完整准备成本或仅改Nv的收益**。火焰图四份及原始perf.data均已保存。

## 构建和复跑（仅交付命令，本次未执行）

```bash
cd /home/cts/workspace/rustpower-amd-supervariable
cargo build --offline --release --features klu_ybus_amd
cargo test --offline -p rustpower-sol-klu --features ybus_amd
cargo test --offline --features klu_ybus_amd --test ybus_ordering
cargo test --offline -p rustpower-amd
BENCH_CPU=0 REPEATS=100 bash docs/experiments/amd-ybus/run-rust-amd-bench.sh
BENCH_CPU=0 REPEATS=100 bash docs/experiments/amd-ybus/run-integration-bench.sh
# 最新Nv=2与标量内核诊断；依赖已有target/amd-profile的release依赖
bash docs/experiments/amd-nv2-profile/build.sh
bash docs/experiments/amd-nv2-profile/run.sh
```

需系统SuiteSparse/KLU/AMD库及头文件；build.rs支持SUITESPARSE_DIR。已有测试日志及命令保留在docs/experiments/amd-ybus。后续执行前核实CPU/电源，不把旧不同批数据直接相减。

## 与有效实现分开保存的研究

`src/exact.rs`、`src/on_demand.rs`、`src/lazy_core.rs`是完整初始表/按需标量研究，不能代替已验证Nv=2路径；按需版本未获得加速。`amd-ybus/baseline-profile`测错了用户所需原版J AMD范围，报告已标注纠正，不能沿用其结论。正确J入口火焰图在`amd-j-profile`。

列重排实验已按用户指示废弃，不恢复、不纳入交付方案。远程0.5.5的已知serde/changelog delta已本地应用，见INTEGRATION_RESULTS.md；未产生合并提交。保留全部现有文件和未提交修改，后继接管者自行审查，不自动提交推送或开始新研究。
