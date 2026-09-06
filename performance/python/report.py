"""从两组 LM 原始测量生成唯一最终报告，不把未收敛耗时用于速度比较。"""
import json
from pathlib import Path
from statistics import median
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'target/research/performance'
REPORT = ROOT / 'performance/REPORT.md'
LABELS = {'NE-cached':'正规方程：固定结构重算乘积', 'NE-rebuild':'正规方程：通用乘法',
          'AUG-upper':'增广：原上三角填充', 'AUG-operator':'增广：直接算子',
          'AUG-COO-legacy':'baseline：V4填J后COO组装',
          'AUG-FS-legacy':'baseline：全J裁剪后COO组装'}
CASES = ['IEEE39','IEEE118','pegase9241','6515rte_dc','6515rte_flat']

def read(name):
    folder=DATA/name
    return json.loads((folder/'measurements.json').read_text()), json.loads((folder/'environment.json').read_text())

def counts(rows,key):
    values=sorted({x[key] for x in rows})
    return str(values[0]) if len(values)==1 else f'{values[0]}–{values[-1]}'

def table(rows):
    out=['| 算法 | 收敛/测量 | 接受步数 | 线性求解次数 | 驱动初始化 ms | 潮流求解 ms | 最终残差∞ |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for method in dict.fromkeys(x['method'] for x in rows):
        samples=[x for x in rows if x['method']==method and x['round']>0]
        assert len(samples)==7, (method,len(samples))
        ok=sum(x.get('converged',x['residual_inf']<1e-8) for x in samples)
        elapsed=f'{median(x["solve_ms"] for x in samples):.3f}' if ok==7 else '未收敛'
        residual=max(x['residual_inf'] for x in samples)
        out.append(f'| {LABELS.get(method,method)} | {ok}/7 | {counts(samples,"iterations")} | {counts(samples,"linear_solves")} | {median(x["build_ms"] for x in samples):.3f} | {elapsed} | {residual:.2e} |')
    return out

def assembly_table(rows):
    out=['| 算法 | J或Jᵀ填充 ms | JᵀJ计算 ms | COO组装转换 ms | μ及右端项 ms | 准备合计 ms |',
         '|---|---:|---:|---:|---:|---:|']
    for method in dict.fromkeys(x['method'] for x in rows):
        samples=[x for x in rows if x['method']==method and x['round']>0]
        if not all(x['converged'] for x in samples):
            out.append(f'| {LABELS[method]} | 未收敛 | — | — | — | — |')
            continue
        values=[median(x['j_or_aug_fill_ms'] for x in samples),
                median(x['product_symbolic_ms']+x['product_numeric_ms'] for x in samples),
                median(x['coo_ms'] for x in samples),
                median(x['mu_ms'] for x in samples),
                median(x['matrix_preparation_ms'] for x in samples)]
        out.append('| '+LABELS[method]+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    return out

def main():
    assembly,env_a=read('lm-assembly'); solvers,env_s=read('lm-solvers')
    stamp=datetime.now(timezone.utc).isoformat(timespec='seconds')
    lines=['# 性能对比最终报告','',f'生成时间：{stamp}。由 `performance/python/report.py` 从原始记录生成。',
    '', '## 范围与运行方式','',
    '本报告汇总本轮 LM 装配和线性求解器实测。ACPF、OPF 的既有性能程序已迁入同一目录并独立编译；本轮未重测这些旧入口，不把历史数值冒充本轮结果。正确性单元测试保留在 src，性能程序通过 cargo bench 单独运行。',
    '', '```bash', 'export CARGO_TARGET_DIR=/home/cts/workspace/rustpower/target',
    'python performance/python/audit_lm_6515.py',
    'cargo bench --features benchmark --bench comparison -- lm-check',
    'taskset -c 1 cargo bench --features benchmark --bench comparison -- lm-assembly',
    'cc -O3 -fPIC -shared -I/usr/include/suitesparse performance/lm_cholmod_bench.c -o /tmp/liblm_cholmod.so -lcholmod',
    'OMP_NUM_THREADS=1 RUSTPOWER_CHOLMOD_THREADS=1 RUSTPOWER_CHOLMOD_LIBRARY=/tmp/liblm_cholmod.so taskset -c 1 cargo bench --features benchmark --bench comparison -- lm-solvers',
    'taskset -c 1 cargo bench --features benchmark --bench comparison -- v4vsoperator',
    'python performance/python/report.py','```','',
    'Cholesky 默认链接系统 BLAS；使用 OpenBLAS 时须指定 LD_PRELOAD。实际环境以下表为准。旧 `bench_*` 示例也已移入 performance，仍通过 `cargo run --release --example 名称` 运行。其他 ACPF/OPF 入口运行 `cargo bench --features benchmark --bench comparison -- --help` 查看。',
    '', '## 实测环境','', '| 项目 | 装配对比 | 求解器对比 |','|---|---|---|']
    for key,label in [('cpu_info','CPU'),('cpu_affinity','绑核'),('rustc','Rust'),('git_commit','基准提交'),('omp_threads','OMP线程'),('blas_preload','预加载BLAS'),('cholmod_threads','CHOLMOD线程'),('cholmod_library','CHOLMOD接口')]:
        lines.append(f'| {label} | {env_a.get(key) or "未设置"} | {env_s.get(key) or "未设置"} |')
    lines+=['','运行时存在未提交修改，完整状态记录在各组 environment.json。每种算法先预热1次，再独立求解7次；时间取中位数。当前四条装配路径在一次潮流内部复用线性求解器；两条历史baseline按原代码每次试步重新创建求解器。',
    '', '## 算法参数与初值','',
    'LM对比的路径均为 GN-LM：正规方程 `(JᵀJ+μI)δ=-Jᵀr`，或其等价增广方程。没有使用完整残差 Hessian，也没有施加 OPF 不等式约束。收敛判据是潮流残差无穷范数小于 1e-8，最多300个外层迭代。后文v4vsoperator是独立的纯填充测试，不运行LM。',
    '', '| 算例 | 初值来源 | 信赖域初始半径 |','|---|---|---:|']
    for case in CASES:
        row=next(x for x in solvers if x['case']==case)
        tr=row['options']['trust_region']
        init='pandapower DCPF 初值' if case.endswith('_dc') else 'pandapower 平启动，保留电压设定值' if case.endswith('_flat') else 'cases ZIP 经既有 PowerGrid 初始化得到的 v_bus_init'
        lines.append(f'| {case} | {init} | {tr["initial_radius"] if tr else "未启用显式半径"} |')
    opts=solvers[0]['options']
    lines+=['', '当前实现共用参数（直接取自实测记录；两条历史baseline的差异见下文）：','', '| 参数 | 值 |','|---|---|']
    for key,label in [('damping_metric','阻尼尺度'),('initial_mu','初始μ'),('min_mu','最小μ'),('max_mu','最大μ'),('max_trials','每步最多试探次数'),('mu_increase','拒绝后μ放大因子'),('failed_step_increase','线性求解失败后μ放大因子'),('mu_decrease','好步μ减小因子'),('acceptance_threshold','ρ接受阈值'),('good_step_threshold','好步ρ阈值'),('reject_nonpositive_voltage','拒绝非正电压幅值')]:
        assert all(x['options'][key]==opts[key] for x in assembly+solvers if x.get('policy')!='legacy'),key
        lines.append(f'| {label} | {opts[key]} |')
    tr=next(x['options']['trust_region'] for x in solvers if x['case']=='6515rte_flat')
    lines+=['','6515平启动的完整信赖域配置：','', '```json',json.dumps(tr,ensure_ascii=False,indent=2),'```','',
    '两条历史baseline原样运行：μ初值0.01，下限1e-12；拒绝后乘2，失败后乘10，超过1e12退出；每步最多30次试探。ρ大于1e-4接受，大于0.75时μ除以3。没有显式信赖域，也不拒绝非正幅值。其预测下降量是 `−gᵀδ/2`，当前实现则为 `(−gᵀδ+μ‖δ‖²)/2`（本次使用μI）。因此它们是历史完整流程对照，不能把总耗时差全归给矩阵组装。',
    '', '接受步数为外层迭代数；线性求解次数包含被拒绝的试步与求解失败重试。当前四条装配路径检查迭代数、求解次数和最终电压一致；历史baseline允许计数不同，收敛时独立验证残差并记录电压差。报告残差列取七次中的最大值。',
    '', '## 装配对比：四条当前路径和两条历史baseline','',
    '**旧报告的“构建 ms”只是驱动器初始化，不是最终矩阵组装。正规方程的首次JᵀJ结构和数值计算发生在求解阶段；用这一列判断正规方程组装更快是错误的。下表已改名，并补上求解期间的矩阵准备分项。**',
    '', '| 路径 | 实际代码与操作 |','|---|---|',
    '| 正规方程：固定结构重算乘积 | `src/lm/normal_eq/mod.rs`：V4填J；首次通用乘法建立JᵀJ，随后按固定结果结构重算数值。 |',
    '| 正规方程：通用乘法 | 同一NeDriver的`dumb_mode=true`：V4填J；每次外层迭代通用乘法重建JᵀJ。 |',
    '| 增广：原上三角填充 | `src/lm/gn_triu.rs`的`build`：`fill_jt_rows`直接填增广上三角中的Jᵀ。 |',
    '| 增广：直接算子 | 同一GnTriuDriver的`build_operator`：`JacobianOperator`直接填增广上三角中的Jᵀ。 |',
    '| baseline：V4填J后COO组装 | `src/lm/baseline/aug_coo.rs`：V4填J，每次试步把完整增广矩阵写入COO并转换为CSC。 |',
    '| baseline：全J裁剪后COO组装 | `src/lm/baseline/full_slice.rs`：计算全节点极坐标J，裁剪后组装完整增广COO，再转换为CSC。 |',
    '', '“固定结构重算乘积”是JᵀJ的另一种计算实现，不是V4的直接雅可比填充；两种正规方程都调用V4填J。它为每个乘积元素重新匹配J中的行号，没有预存乘法项的位置。首次通用乘法已得到数值后，还会再算一次数值。',
    '', '分项时间均为一次完整潮流求解内的累计时间，然后对七次运行取中位数，不是单次填充时间：',
    '', '- JᵀJ计算包含通用乘法的分配、结构和数值计算、格式转换，以及固定结构路径的后续数值更新；原始字段product_symbolic_ms并非纯符号分析时间。',
    '- COO组装转换包含μ和−I写入、裁剪（全J路径）、COO转CSC及数组提取。',
    '- 当前四条路径的μ计时还包含右端项写入；历史baseline的μ写入已计入COO，右端项写入计入其线性求解时间，因此其μ列为0。',
    '- 准备合计对每次运行先求和再取中位数，不包含驱动初始化、梯度Jᵀr、残差评估和线性求解。这是实际流程的准备阶段计时，各路径对功率计算和工作数组的复用不同，不是同一输入下的纯填充内核测试。',
    '- 所有路径使用QDLDL，但历史baseline还会重复符号分析；未收敛的历史运行不参与速度比较。','']
    lines+=['### 9241：每次填充与旧工作记录对照','',
    '旧工作记录的805–974 µs是每次迭代的填充时间。旧提交6077c28的baseline/bench.rs明确输出prof_fill_ns / iterations。当前收敛路径也是每次外层迭代填充一次，μ重试只修改对角；以下对每次运行先以累计填充时间除以迭代数，再取七次运行的中位数。',
    '', '| 来源与路径 | 填充次数 | 累计填充 ms | 平均每次填充 µs |',
    '|---|---:|---:|---:|',
    '| 旧工作记录：原上三角填充 | 11 | 约8.855–10.714（按均值换算） | 805–974 |']
    for method in ['AUG-upper','AUG-operator']:
        rows=[x for x in assembly if x['case']=='pegase9241' and x['method']==method and x['round']>0]
        assert len(rows)==7 and all(x['converged'] and x['iterations']==11 for x in rows)
        total=median(x['j_or_aug_fill_ms'] for x in rows)
        average=median(x['j_or_aug_fill_ms']*1000/x['iterations'] for x in rows)
        lines.append(f'| 本次实测：{LABELS[method]} | 11 | {total:.3f} | {average:.1f} |')
    lines+=['', '本表只计Jᵀ填充及其工作向量准备，不含μ和右端项更新。原上三角fill_jt_rows与历史提交6077c28逐段对照，仅有格式变化。旧代码还在fill计时内计算scalc；当前实现从残差阶段复用scalc。因此历史记录与本次结果保留各自的测量范围，同机的原上三角和直接算子则都复用scalc。CPU对差异的贡献未单独测量。','']
    for case in CASES:
        rows=[x for x in assembly if x['case']==case]
        lines+=['### '+case,'']+table(rows)+['','求解期间的矩阵准备：','']+assembly_table(rows)+['']
    if (DATA/'v4vsoperator/measurements.json').exists():
        fills,env_f=read('v4vsoperator')
        lines+=['## V4与operator直接填J：v4vsoperator','',
        '测试文件：performance/v4vsoperator.rs。两者直接填同一个CSC雅可比矩阵，使用同一Ybus、V、scalc和V4符号缓存；不涉及JᵀJ、增广矩阵或线性求解。',
        '', '复用既有v4.rs的非平坦电压：节点k的幅值为1+0.004cos(2.1k)，相角为0.03sin(1.3k)−0.01k弧度；scalc在计时前计算。纯填充预先准备Vnorm和1/|V|；第二组另外计入V4的Vnorm计算或operator的1/|V|计算。输出数组全程复用。',
        '', f'CPU：{env_f["cpu_info"]}；绑核：{env_f["cpu_affinity"]}。每组预热20次，测7轮，每轮各填2000次并交替先后顺序；表中为各轮平均单次耗时的中位数。没有潮流迭代、阻尼参数或收敛结果；正确性按两个输出的差检查。',
        '', '| 算例 | 范围 | V4 µs/次 | operator µs/次 | V4耗时/operator耗时 |',
        '|---|---|---:|---:|---:|']
        for case in ['IEEE39','IEEE118','pegase9241']:
            for scope in ['纯填充','电压向量准备+填充']:
                rows=[x for x in fills if x['case']==case and x['scope']==scope]
                assert len(rows)==7
                a=median(x['v4_us_per_fill'] for x in rows)
                b=median(x['operator_us_per_fill'] for x in rows)
                lines.append(f'| {case} | {scope} | {a:.3f} | {b:.3f} | {a/b:.3f} |')
        error=max(x['relative_inf_error'] for x in fills)
        lines+=['', f'相对误差定义为max|J_V4−J_operator|/max(1,max|J_V4|)。三个算例的最大值为{error:.3e}，检查阈值2e-13；同时检查全部输出有限、没有遗漏填充。原始记录：target/research/performance/v4vsoperator/。','']
    lines+=['## 求解器对比','', '正规方程各后端使用同一通用稀疏乘积。完整增广矩阵的 KLU/QDLDL 是相同布局对比，上三角 QDLDL 单列；不能把布局收益全部归因于分解算法。','']
    for case in CASES:
        lines+=['### '+case,'']+table([x for x in solvers if x['case']==case])+['']
    lines+=['## 未收敛结果与解释','',
    '6515平启动的正规方程KLU在第一步内耗尽30次试探。诊断中所有调用返回成功，但 refactor 路径的相对线性残差最终约为3.2e18；同一矩阵全新分解约为2.8e-16。每次完整分解的诊断路径24步、44次线性求解收敛。该额外诊断不用于性能比较。',
    '', '当前封装只检查状态码，未调用 rcond、rgrowth、condest；KLU官方说明 refactor沿用原主元顺序，应另行检查数值质量。这是复用路径的可靠性问题，不能写成正规方程本身不收敛。',
    '', '## 原始数据与文件组织','',
    '- `performance/`：全部性能程序、Python基准、审计程序、CHOLMOD测试接口和本报告。',
    '- `src/`：算法实现和正确性单元测试；不挂载性能测试。',
    '- `target/research/performance/lm-assembly/`、`lm-solvers/`：本轮 measurements.json 和 environment.json。',
    '- `target/research/lm_audit/linear_solvers/klu_diagnostic.log`：前述KLU诊断日志。',
    '', '迁移后103项库正确性测试通过，2项保留忽略，无性能测试混入；独立 lm-check 通过。原始数据目录由现有 target 忽略规则排除。此前的Markdown实验报告已归档到工作树外，本目录仅保留这份最终报告。']
    REPORT.write_text('\n'.join(lines)+'\n')
    print(REPORT)

if __name__=='__main__':main()
