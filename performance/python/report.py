"""从同一轮LM实测生成唯一报告：组装时间、总执行时间及相对基线的加速比。"""
import json
import math
import sys
sys.dont_write_bytecode = True
from ablation import STAGES, export
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "target/research/performance/lm-assembly"
REPORT = ROOT / "performance/REPORT.md"
METHODS = {
    "AUG-COO-upper": "COO上三角：V4填J",
    "AUG-COO": "COO完整：V4填J",
    "NE-rebuild": "正规方程基线：通用乘法",
    "NE-cached": "正规方程：固定乘积结构",
    "AUG-FS-upper": "COO上三角：全J裁剪",
    "AUG-FS": "COO完整：全J裁剪",
    "AUG-upper": "增广：原上三角填充",
    "AUG-operator": "增广：直接算子",
}
CASES = ["IEEE39", "IEEE118", "pegase9241", "6515rte_dc", "6515rte_flat"]


def samples(records, case, method):
    return [r for r in records if r["case"] == case and r["method"] == method and r["round"] > 0]


def elapsed(records, case, method, key):
    return median(r[key] for r in samples(records, case, method))


def validate(records, methods=METHODS):
    assert len(records) == len(CASES) * len(methods) * 8
    keys = {(r["case"], r["method"], r["round"]) for r in records}
    expected = {(c, m, i) for c in CASES for m in methods for i in range(8)}
    assert keys == expected, "算例、方法或测量轮次不完整"
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
        for key in ["build_ms", "solve_ms", "matrix_preparation_ms", "total_execution_ms"]:
            assert math.isfinite(r[key]) and r[key] >= 0
        assembly = sum(r[k] for k in ["j_or_aug_fill_ms", "product_symbolic_ms", "product_numeric_ms", "mu_ms", "coo_ms"])
        assert math.isclose(assembly, r["matrix_preparation_ms"], abs_tol=1e-9)
        assert math.isclose(r["total_execution_ms"], r["build_ms"] + r["solve_ms"], abs_tol=1e-9)
        assert r["matrix_preparation_ms"] <= r["solve_ms"]


def main():
    records = json.loads((DATA / "measurements.json").read_text())
    environment = json.loads((DATA / "environment.json").read_text())
    validate(records)
    ablation_folder = DATA.parent / "lm-ablation"
    ablation = json.loads((ablation_folder / "measurements.json").read_text())
    ablation_env = json.loads((ablation_folder / "environment.json").read_text())
    validate(ablation, STAGES)
    ablation_lines = export(ROOT, records, environment, ablation, ablation_env, METHODS, CASES)
    options = records[0]["options"]
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# LM矩阵组装与总执行时间", "",
        f"生成时间：{stamp}。数据来自同一轮 `lm-assembly`，所有COO路径已修复求解器复用和步长控制。", "",
        "## 比较对象", "",
        "主要加速比选取两种现有实现作为比较基线：V4填J后以COO只组装增广上三角，以及使用通用稀疏乘法计算JᵀJ的正规方程。原上三角填充、固定结构乘积和全J裁剪COO同时列出；两条COO均保留upper_only=false完整模式和true上三角模式。", "",
        "| 方法 | 每轮实际组装操作 |",
        "|---|---|",
        "| COO：V4填J | V4填独立J；每次试步写增广COO、转CSC，再更新固定求解器输入缓冲区。upper_only=true时跳过下左J块，保留完整Jᵀ块。 |",
        "| 正规方程基线：通用乘法 | V4填J；每个外层迭代用nalgebra_sparse通用乘法重新生成JᵀJ，再写μ对角。没有COO组装步骤。 |",
        "| 正规方程：固定乘积结构 | V4填J；首次通用乘法建立乘积结构，随后按该结构重算数值，再写μ对角。 |",
        "| COO：全J裁剪 | 计算全节点极坐标Jacobian，裁剪保留方程，再以COO组装增广矩阵并转CSC。同样由upper_only控制是否写下左J块。 |",
        "| 增广：原上三角填充 | fill_jt_rows直接填上三角的Jᵀ块，使用预先构建的数值Yᵀ；试步更新μ对角。 |",
        "| 增广：直接算子 | JacobianOperator直接填上三角的Jᵀ块，通过符号镜像索引读取当前Ybus；试步更新μ对角。 |", "",
        "八条路径均使用QDLDL，在一次潮流内复用求解器及其符号分析。COO三元组按所需条目数预留容量并跨迭代、重试复用，全J裁剪版本也复用全J三元组；仍每次重新写三元组和转换格式。首次reserve计入对应填J或COO计时，COO转CSC的内部临时分配仍保留；CSC结构相同时只复制数值到固定输入缓冲区，结构改变时才重置求解器。两条正规方程也复用求解器输入缓冲区。", "",
        "正规方程求解 `(JᵀJ+μI)δ=−Jᵀr`；增广路径求解 `[μI Jᵀ; J −I][δ;s]=[0;−r]`。所有路径使用相同的未加阻尼二次模型下降量计算ρ；增广路径从解向量中的s计算该量，省去Jᵀr乘法。", "",
        "## 计时范围", "",
        "- **组装及右端准备**：一次完整潮流内，J/Jᵀ填充、JᵀJ乘积、COO写入与转换、μ对角及右端准备的累计时间。直接读取已有驱动计时器。μ和右端写入共用计时段，因此本列包含右端准备，不能解读为纯矩阵填充内核时间。",
        "- **总执行时间**：驱动及求解器初始化时间加完整LM求解时间，包含组装、残差计算、步长控制、重试和线性求解。逐次运行先求和，再取中位数。",
        "- 文件读取、网络模型初始化、输入电压复制和求解结束后的独立正确性检查均在计时外；首次驱动符号准备计入初始化，首次线性分解计入求解。",
        "- 组装列不含初始化、线性分解和残差评估。V4基线在填J计时内准备scalc/Vnorm，直接增广从残差阶段复用scalc；这是各实现实际流程的开销，未将其称为相同输入下的纯内核消融。", "",
        "预热1次，测量7次；每轮反转方法执行顺序，表中时间为7次的中位数，均以ms计。每次测量从相同初值独立求解。", "",
        "## 运行环境与参数", "",
        "| 项目 | 值 |", "|---|---|",
    ]
    for key, label in [("cpu_info", "CPU"), ("cpu_affinity", "绑核"), ("rustc", "Rust"), ("git_commit", "基准提交"), ("omp_threads", "OMP线程")]:
        value = str(environment.get(key) or "未设置").replace("\t", " ")
        lines.append(f"| {label} | {value} |")
    lines += ["", "运行时包含未提交修改，原始environment.json保留代码差异和输入文件SHA-256。", "",
              "| 参数 | 值 |", "|---|---|"]
    for key, label in [("damping_metric", "阻尼尺度"), ("initial_mu", "初始μ"), ("min_mu", "最小μ"), ("max_mu", "最大μ"), ("max_trials", "每外层迭代最多试步"), ("mu_increase", "拒绝后μ乘数"), ("failed_step_increase", "线性失败后μ乘数"), ("mu_decrease", "好步后μ除数"), ("acceptance_threshold", "ρ接受阈值"), ("good_step_threshold", "好步ρ阈值"), ("reject_nonpositive_voltage", "拒绝非正电压幅值")]:
        assert all(r["options"][key] == options[key] for r in records)
        lines.append(f"| {label} | {options[key]} |")
    lines += ["| 残差∞容差 | 1e-8 |", "| 最大外层迭代 | 300 |", "",
              "IEEE39、IEEE118、pegase9241使用既有CSV ZIP解析及PowerGrid初始化得到的v_bus_init；6515rte_dc和平启动使用既有pandapower导出的同一潮流模型。平启动保留电压设定幅值。", "",
              "除6515平启动外，不启用显式信赖域半径，均使用自适应μ。6515平启动的信赖域参数如下，八条路径一致：", "", "```json",
              json.dumps(next(r["options"]["trust_region"] for r in records if r["case"] == "6515rte_flat"), ensure_ascii=False, indent=2), "```", "",
              "## 组装与总执行时间", "",
              "每个算例的八条路径均为7/7次测量收敛。接受步数和线性求解次数列在各表前，线性求解次数包含被拒绝的试步。残差列取7次最大值。", ""]
    for case in CASES:
        row = next(r for r in records if r["case"] == case)
        lines += [f"### {case}", "",
                  f"节点{row['buses']}，状态变量{row['states']}；八条路径均接受{row['iterations']}步，线性求解{row['linear_solves']}次。", "",
                  "| 方法 | 组装及右端准备 ms | 总执行 ms | 其中初始化 ms | 最终残差∞ |",
                  "|---|---:|---:|---:|---:|"]
        for method, label in METHODS.items():
            measured = samples(records, case, method)
            times = [elapsed(records, case, method, k) for k in ["matrix_preparation_ms", "total_execution_ms", "build_ms"]]
            lines.append(f"| {label} | " + " | ".join(f"{t:.3f}" for t in times) + f" | {max(r['residual_inf'] for r in measured):.2e} |")
        lines += [""]
    lines += ["## 直接算子相对两条主要基线的加速比", "",
              "加速比 = 基线耗时中位数 ÷ 直接算子耗时中位数；大于1表示直接算子更快。组装与总执行分别计算。", "",
              "| 算例 | 相对COO上三角：组装 | 相对COO上三角：总执行 | 相对通用JᵀJ：组装 | 相对通用JᵀJ：总执行 |",
              "|---|---:|---:|---:|---:|"]
    for case in CASES:
        ratios = [elapsed(records, case, baseline, key) / elapsed(records, case, "AUG-operator", key)
                  for baseline in ["AUG-COO-upper", "NE-rebuild"] for key in ["matrix_preparation_ms", "total_execution_ms"]]
        lines.append(f"| {case} | " + " | ".join(f"{r:.2f}×" for r in ratios) + " |")
    lines += [""] + ablation_lines
    lines += ["", "## 结果解释", "",
              "直接增广组装省去独立J到COO的写入、格式转换和数值复制；相对正规方程，还省去JᵀJ乘积。两种正规方程都已经使用V4填J，因此与它们的对比衡量的是最终线性系统构造方式，并非用普通Jacobian算法替代V4。", "",
              "原上三角填充也已直接写入Jᵀ块，没有独立J和COO中间矩阵。它与直接算子的对比，才反映这两种填充实现的差异。", "",
              "总执行时间还受矩阵维数、稀疏结构和线性分解影响：正规方程为n阶，增广为2n阶；主要COO基线和直接路径现在均只存增广上三角；完整COO模式另列，用于观察多写下左J块的代价。因此总加速比表示完整实现的收益，不能全归给填充内核。", "",
              "固定结构乘积保留为另一种实现对照。它每次重新匹配J列中的行号以计算乘积项，并未预存所有乘法项位置；首次通用乘法已得到数值后还会重算一次。", "",
              f"320次运行（含预热）全部收敛；每个算例的八条路径迭代数和线性求解次数一致。最终电压相对同算例正规方程参考的最大差为{max(r['max_voltage_difference'] for r in records):.3e}；残差在计时外重新用Ybus、Sbus和最终电压计算。", "",
              "## 复现与数据", "", "在仓库根目录运行，沿用已有6515导出文件：", "", "```bash",
              "export CARGO_TARGET_DIR=/home/cts/workspace/rustpower/target",
              "cargo bench --locked --features benchmark --bench comparison -- lm-check",
              "OMP_NUM_THREADS=1 taskset -c 1 cargo bench --locked --features benchmark --bench comparison -- lm-assembly",
              "OMP_NUM_THREADS=1 taskset -c 1 cargo bench --locked --features benchmark --bench comparison -- lm-ablation",
              "python3 performance/python/report.py", "```", "",
              "6515输入需要重新导出时，在含pandapower的Python环境执行 `python performance/python/audit_lm_6515.py --output target/research/lm_audit`。", "",
              "- 测试入口：`performance/lm.rs`；报告生成：`performance/python/report.py`。",
              "- 原始测量：`target/research/performance/lm-assembly/measurements.json`，含完整参数、7轮时间、组装分项和求解器已有计时器数据。",
              "- 环境及代码差异：同目录`environment.json`。",
              "- 主对比与独立消融分别来自lm-assembly和lm-ablation；两组参数与逐轮时间随CSV保存，不把两组时间混算加速比。", ""]
    REPORT.write_text("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
