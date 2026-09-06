"""输出完整LM实测报告到Markdown和stdout；读取已有数据，不运行求解器。"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

sys.dont_write_bytecode = True
from ablation import STAGES, BACKENDS, BACKEND_TIMES, backend_summaries, export

ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "AUG-FS-upper": "全J裁剪＋上三角COO（不用V4）",
    "AUG-FS": "全J裁剪＋完整COO（不用V4）",
    "NE-rebuild": "V4＋通用JᵀJ",
    "NE-cached": "V4＋固定结构JᵀJ",
    "AUG-COO-upper": "V4＋上三角COO（内部消融）",
    "AUG-COO": "V4＋完整COO（内部消融）",
    "AUG-upper": "原triu直接填充",
    "AUG-operator": "算子triu直接填充",
}
CASES = ["IEEE39", "IEEE118", "pegase9241", "6515rte_dc", "6515rte_flat"]
ASSEMBLY_FIELDS = [
    ("j_or_aug_fill_ms", "J/Jᵀ评估与填充"),
    ("product_symbolic_ms", "乘积构建"),
    ("product_numeric_ms", "固定乘积数值"),
    ("coo_ms", "COO"), ("mu_ms", "μ/右端"),
    ("matrix_preparation_ms", "组装准备合计"),
]
EXECUTION_FIELDS = [
    ("build_ms", "初始化"), ("solve_ms", "LM求解"),
    ("total_execution_ms", "总执行"), ("linear_total_ms", "其中线性求解"),
    ("solver_setup_ms", "其中符号分析"),
    ("solver_numeric_ms", "其中数值分解"),
    ("solver_backsolve_ms", "其中回代"),
]


def samples(records, case, method):
    return [r for r in records if r["case"] == case and r["method"] == method and r["round"] > 0]


def elapsed(records, case, method, key):
    return median(r[key] for r in samples(records, case, method))


def validate(records, methods):
    expected = {(c, m, i) for c in CASES for m in methods for i in range(8)}
    assert len(records) == len(expected)
    assert {(r["case"], r["method"], r["round"]) for r in records} == expected
    for case in CASES:
        rows = [r for r in records if r["case"] == case]
        assert len({(r["iterations"], r["linear_solves"]) for r in rows}) == 1
        assert all(r["options"] == rows[0]["options"] for r in rows)
    for r in records:
        assert r["policy"] == "current" and r["solver_reuse"]
        if r["method"].startswith(("AUG-COO", "AUG-FS")):
            assert r["coo_upper_only"] == r["method"].endswith("-upper")
        assert r["converged"] and r["residual_inf"] < r["tolerance_inf"] == 1e-8
        assert r["max_voltage_difference"] < 1e-6
        for key, _ in ASSEMBLY_FIELDS + EXECUTION_FIELDS:
            assert math.isfinite(r[key]) and r[key] >= 0
        assembly = sum(r[k] for k, _ in ASSEMBLY_FIELDS[:-1])
        assert math.isclose(assembly, r["matrix_preparation_ms"], abs_tol=1e-9)
        assert math.isclose(r["total_execution_ms"], r["build_ms"] + r["solve_ms"], abs_tol=1e-9)
        assert r["matrix_preparation_ms"] <= r["solve_ms"]


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"] + [
        "| " + " | ".join(str(v) for v in row) + " |" for row in rows
    ] + [""]


def summary(records, methods, title):
    lines = [f"## {title}", "", "以下分项均为7次测量的中位数，单位ms；第0轮预热不参与汇总。", ""]
    for case in CASES:
        first = next(r for r in records if r["case"] == case)
        lines += [f"### {case}", "",
                  f"节点{first['buses']}；状态变量{first['states']}；各方法接受{first['iterations']}步、线性求解{first['linear_solves']}次。", ""]
        for fields in [ASSEMBLY_FIELDS, EXECUTION_FIELDS]:
            lines += table(["方法"] + [name for _, name in fields], [
                [METHODS[m]] + [f"{elapsed(records, case, m, key):.6f}" for key, _ in fields]
                for m in methods
            ])
        rows = []
        for m in methods:
            group = samples(records, case, m)
            ranges = [f"{min(r[k] for r in group):.6f}–{max(r[k] for r in group):.6f}"
                      for k in ["matrix_preparation_ms", "total_execution_ms"]]
            rows.append([METHODS[m], f"{sum(r['converged'] for r in group)}/{len(group)}", *ranges,
                         f"{max(r['residual_inf'] for r in group):.3e}",
                         f"{max(r['max_voltage_difference'] for r in group):.3e}"])
        lines += table(["方法", "收敛/测量", "组装准备最小–最大", "总执行最小–最大", "最大残差∞", "最大电压差"], rows)
    return lines


def raw_tables(suite, records):
    lines = [f"## 逐轮原始数据：{suite}", "",
             "包含第0轮预热及第1–7轮测量，顺序与实际执行相同。时间单位ms，显示6位小数；CSV保留原始浮点精度。方法标识见前表，参数见各算例配置。", ""]
    for case in CASES:
        rows = [r for r in records if r["case"] == case]
        lines += ["<details>", f"<summary>{case}：{len(rows)}次运行</summary>", ""]
        for fields in [ASSEMBLY_FIELDS, EXECUTION_FIELDS]:
            lines += table(["轮次", "方法"] + [name for _, name in fields], [
                [r["round"], r["method"]] + [f"{r[key]:.6f}" for key, _ in fields] for r in rows
            ])
        lines += table(["轮次", "方法", "收敛", "接受步", "线性求解", "J评估", "COO组装", "残差∞", "最大电压差"], [
            [r["round"], r["method"], "是" if r["converged"] else "否", r["iterations"], r["linear_solves"],
             r["jacobian_evaluations"], r["coo_assemblies"], f"{r['residual_inf']:.6e}",
             f"{r['max_voltage_difference']:.6e}"] for r in rows
        ])
        lines += ["</details>", ""]
    return lines



def validate_backends(records):
    expected = {(c, m, i) for c in CASES for m in BACKENDS for i in range(8)}
    assert len(records) == len(expected)
    assert {(r['case'], r['method'], r['round']) for r in records} == expected
    for r in records:
        assert r['solver_reuse']
        for key in BACKEND_TIMES:
            assert math.isfinite(r[key]) and r[key] >= 0
        assert math.isclose(r['total_execution_ms'], r['build_ms'] + r['solve_ms'], abs_tol=1e-9)
        if r['converged']:
            assert r['residual_inf'] < 1e-8 and r['max_voltage_difference'] < 1e-6
        if r['backend'] == 'KLU':
            assert r['matrix_storage'] == 'full' and r['first_factor_count'] == 1


def backend_report(records):
    lines = ['## KLU与QDLDL性能', '',
             '独立运行lm-solvers --klu，五条路径使用同一模型、初值和LM参数。正规方程两条都使用V4及通用JᵀJ；完整增广两条都使用同一个算子填充。QDLDL上三角另列。KLU接收完整CSC，不能直接把上三角输入当作完整系统求解。', '',
             '每次完整潮流内复用求解器；初始化包括后端构造和驱动构建。分解计时包含KLU的首次factor、refactor和触发的factor fallback，次数取现有计数器。不同后端的浮点误差和失败重试可能改变迭代数，表中逐方法记录；未收敛运行的耗时不作为求解加速。', '']
    summary_rows = backend_summaries(records, CASES)
    for case in CASES:
        group = [r for r in summary_rows if r['case'] == case]
        lines += [f'### {case}：后端比较', '']
        def span(r, key):
            lo, hi = r[key + '_min'], r[key + '_max']
            if lo is None: return '—'
            return str(lo) if lo == hi else f'{lo}–{hi}'
        lines += table(['方法', '收敛/测量', '接受步', '线性求解', '组装准备ms', '线性求解ms', '总执行ms', '最大残差∞'], [
            [BACKENDS[r['method']], f"{r['converged_runs']}/{r['measured_runs']}", span(r, 'iterations'), span(r, 'linear_solves'),
             *[f"{r[k]:.6f}" for k in ['matrix_preparation_ms','linear_ms','total_execution_ms']], f"{r['residual_inf_max']:.3e}"] for r in group])
        lines += table(['方法', '符号ms', '分解ms', '回代ms', '首次factor次数', 'refactor次数', 'fallback次数'], [
            [BACKENDS[r['method']], *[f"{r[k]:.6f}" for k in ['analysis_ms','factor_ms','backsolve_ms']],
             *[span(r, k) for k in ['first_factor_count','refactor_count','factor_fallback_count']]] for r in group])
    return lines


def backend_raw(records):
    lines = ['## 逐轮原始数据：lm-solvers --klu', '', '时间单位ms，第0轮为预热。', '']
    for case in CASES:
        rows = [r for r in records if r['case'] == case]
        lines += ['<details>', f'<summary>{case}：{len(rows)}次运行</summary>', '']
        for fields in [BACKEND_TIMES[:5], BACKEND_TIMES[5:]]:
            lines += table(['轮次', '方法'] + fields, [
                [r['round'], BACKENDS[r['method']]] + [f"{r[k]:.6f}" for k in fields] for r in rows])
        lines += table(['轮次', '方法', '收敛', '接受步', '线性求解', '首次factor', 'refactor', 'fallback', '残差∞', '最大电压差'], [
            [r['round'], BACKENDS[r['method']], r['converged'], r['iterations'], r['linear_solves'],
             *[r.get(k, '—') for k in ['first_factor_count','refactor_count','factor_fallback_count']],
             f"{r['residual_inf']:.6e}", r.get('max_voltage_difference', '—')] for r in rows])
        lines += ['</details>', '']
    return lines

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "target/research/performance",
                        help="包含lm-assembly和lm-ablation原始JSON的目录")
    parser.add_argument("--output-root", type=Path, default=Path.home() / "pp",
                        help="输出paper/BENCHMARK_REPORT.md、data/及论文表格，默认~/pp")
    args = parser.parse_args()
    suites = {}
    for name, methods in [("lm-assembly", METHODS), ("lm-ablation", STAGES)]:
        folder = args.data_dir / name
        records = json.loads((folder / "measurements.json").read_text())
        env = json.loads((folder / "environment.json").read_text())
        validate(records, methods)
        suites[name] = (records, env)
    backend_folder = args.data_dir / "lm-solvers"
    backends = json.loads((backend_folder / "measurements.json").read_text())
    backend_env = json.loads((backend_folder / "environment.json").read_text())
    validate_backends(backends)
    records, env = suites["lm-assembly"]
    ablation, ablation_env = suites["lm-ablation"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    ablation_lines = export(args.output_root, records, env, ablation, ablation_env, METHODS, CASES, backends, backend_env)
    report = args.output_root / "paper/BENCHMARK_REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    count = sum(len(r) for r, _ in suites.values()) + len(backends)
    lines = ["# LM完整性能数据：方法比较与内部消融", "",
             f"生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}。读取已有实测记录，不是一次新的性能测量。", "",
             f"共{count}次运行：主对比320次（8条路径），独立消融160次（4条路径），KLU/QDLDL对比200次（5条路径），均包含预热。覆盖5个算例/初值配置。", "",
             "## 比较对象与结论范围", "",
             "V4＋COO是内部消融对照，已经使用作者的Jacobian方法。两条JᵀJ路径同样使用V4，比较的是线性系统构造方式；不能把它们标成未使用V4的常规完整实现。", "",
             "全J裁剪＋COO两条路径没有调用V4，按解析公式计算全节点Jacobian后裁剪。这是仓库内实现，尚不是pandapower或其他外部库的实测结果。常规Jacobian构建＋通用JᵀJ这条完整组合尚未测量，本报告不为它填入推算值。", ""]
    lines += table(["标识", "方法", "每次组装做什么"], [
        ["AUG-FS-upper / AUG-FS", "全J裁剪＋COO，不用V4", "解析公式计算全J；裁剪；写上三角/完整COO并转CSC"],
        ["NE-rebuild", "V4＋通用JᵀJ", "V4填J；通用稀疏乘法生成JᵀJ；写阻尼对角"],
        ["NE-cached", "V4＋固定结构JᵀJ", "首次通用乘法建立结构，随后逐乘积项匹配J行号重算数值"],
        ["AUG-COO-upper / AUG-COO", "V4＋COO，内部消融", "V4填J；写上三角/完整COO并转CSC"],
        ["AUG-upper", "原triu直填", "fill_jt_rows使用数值Yᵀ直接计算并填Jᵀ"],
        ["AUG-operator", "最终算子triu直填", "JacobianOperator读取当前Ybus，直接计算并填Jᵀ"],
    ])
    lines += ["八条路径均使用QDLDL，并在一次潮流内复用求解器及符号分析。COO三元组预留容量并跨迭代、重试复用；全J裁剪版本也复用全J三元组。首次reserve计入对应J或COO计时。COO→CSC仍包含内部临时分配、排序及格式转换，随后检查结构并复制数值到固定求解器输入。", "",
              "## 计时含义", "",
              "- J/Jᵀ：独立J求值和写入，或直接Jᵀ求值和填充。V4包含scalc/Vnorm准备；全J版本包含节点电流及三元组计算；直接路径复用残差阶段的scalc。导数计算并未被消除。",
              "- 乘积构建：通用JᵀJ乘法，包含结构、数值和格式转换，不是纯符号分析；固定乘积数值列记录随后按固定结构重算的时间。",
              "- COO：三元组写入、阻尼和−I对角、COO→CSC、结构检查及数值复制。μ/右端在COO路径只计右端，在其他路径还包含阻尼更新。",
              "- 组装准备合计：J/Jᵀ＋两项乘积＋COO＋μ/右端。总执行：初始化＋完整LM求解。每轮先求和，再取中位数；各分项中位数不保证相加等于合计中位数。",
              "- 线性求解包含符号分析、数值分解、回代及包装开销；后三项取已有QDLDL计时器，是子项，不应再次加到总执行中。未单独计时的残差、步长控制等留在LM求解总时间内。",
              "- 文件读取、网络初始化、初值复制和求解后的独立残差检查不计时；首次驱动初始化和首次线性分解计时。", "",
              "## 环境与算法参数", "",
              "每方法预热1次、测量7次；交替反转方法顺序。三组独立实验分别汇总，不跨组混算加速比。", ""]
    for name, (suite_records, environment) in {**suites, "lm-solvers": (backends, backend_env)}.items():
        lines += [f"### {name}环境", ""]
        lines += table(["项目", "值"], [[key, str(environment.get(key, "未记录")).replace("\t", " ")]
            for key in ["cpu_info", "cpu_affinity", "omp_threads", "rustc", "git_commit", "command"]])
        lines += ["输入文件SHA-256：", "", "```text", environment["input_sha256"].strip(), "```", ""]
    for case in CASES:
        row = next(r for r in records if r["case"] == case)
        other = next(r for r in ablation if r["case"] == case)
        assert row["options"] == other["options"]
        assert all(r["options"] == row["options"] for r in backends if r["case"] == case)
        lines += [f"### {case}参数（三组一致）", "",
                  "残差∞容差1e-8；最多300外层迭代；下列为实际序列化的全部LM参数：", "",
                  "```json", json.dumps(row["options"], ensure_ascii=False, indent=2), "```", ""]
    lines += ["IEEE39、IEEE118、pegase9241使用既有CSV ZIP解析及PowerGrid的v_bus_init；6515两种初值使用既有pandapower导出的同一模型，平启动保留设定幅值。仅6515平启动启用显式信赖域半径，所有算例均自适应调整μ。", ""]
    lines += summary(records, METHODS, "主对比：全部八条路径")
    lines += ["## 最终算子相对各实现的加速比", "",
              "每个值为该对照实现的中位数÷最终算子中位数。全部来自主对比同一批数据；大于1表示最终算子更快。", ""]
    for case in CASES:
        lines += [f"### {case}", ""]
        lines += table(["对照实现", "组装准备加速比", "总执行加速比"], [
            [METHODS[m]] + [f"{elapsed(records, case, m, k) / elapsed(records, case, 'AUG-operator', k):.3f}×"
                            for k in ["matrix_preparation_ms", "total_execution_ms"]]
            for m in METHODS if m != "AUG-operator"
        ])
    lines += ablation_lines
    lines += summary(ablation, STAGES, "独立消融：全部计时器及波动范围")
    lines += backend_report(backends)
    lines += ["## 正确性与解释", "",
              f"主对比320次、消融160次全部收敛，各算例同组内的接受步数和线性求解次数一致。KLU/QDLDL组收敛{sum(r['converged'] for r in backends)}/{len(backends)}次，其迭代和求解次数单独列出。主对比最大电压差{max(r['max_voltage_difference'] for r in records):.3e}；独立消融最大电压差{max(r['max_voltage_difference'] for r in ablation):.3e}。电压差相对各组同算例首条路径，残差在计时外独立重算。", "",
              "正规方程求解(JᵀJ＋μI)δ＝−Jᵀr，增广路径求解[μI Jᵀ; J −I][δ;s]＝[0;−r]。线性系统维数、稀疏结构及分解成本不同；总执行差异不能全部归于组装。", "",
              "V4＋COO到直接填充的差别还包含导数遍历和scalc复用；它是内部实现阶段的消融。固定乘积版本在首次通用乘法得到数值后还会重算一次，未预存所有乘法项位置。", "",
              "## 复现与输出", "", "在仓库根目录运行：", "", "```bash",
              "export CARGO_TARGET_DIR=/home/cts/workspace/rustpower/target",
              "cargo bench --locked --features benchmark --bench comparison -- lm-check",
              "OMP_NUM_THREADS=1 taskset -c 1 cargo bench --locked --features benchmark --bench comparison -- lm-assembly",
              "OMP_NUM_THREADS=1 taskset -c 1 cargo bench --locked --features benchmark --bench comparison -- lm-ablation",
              "OMP_NUM_THREADS=1 taskset -c 1 cargo bench --locked --features benchmark --bench comparison -- lm-solvers --klu",
              "python3 performance/python/report.py --output-root ~/pp", "```", "",
              "测试入口逐次打印参数、全部计时分项、收敛结果和每算例中位数汇总；报告脚本把本报告完整打印到stdout，同时保存Markdown、CSV及论文表格。", "",
              f"- 完整报告：`{report}`。",
              f"- 原始数据和汇总CSV：`{args.output_root / 'data'}`。",
              f"- 原始JSON及运行时源代码差异：`{args.data_dir}`下三组目录的measurements.json和environment.json。",
              "- 6515输入需要重新导出时，沿用performance/python/audit_lm_6515.py。", ""]
    for name, (suite_records, _) in suites.items():
        lines += raw_tables(name, suite_records)
    lines += backend_raw(backends)
    content = "\n".join(lines) + "\n"
    report.write_text(content)
    sys.stdout.write(content)
    print(f"已保存：{report}", file=sys.stderr)


if __name__ == "__main__":
    main()
