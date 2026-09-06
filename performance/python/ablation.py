"""消融分项与CSV导出；只处理驱动器已有计时，不推算未测的独立内核耗时。"""
import csv
import json
from statistics import median

STAGES = {
    'AUG-COO-upper': ('A0', 'V4 + 上三角COO'),
    'AUG-upper': ('A1', '原triu直填'),
    'AUG-operator': ('A2', '算子triu直填'),
}
TIME_FIELDS = ['build_ms', 'solve_ms', 'total_execution_ms', 'j_or_aug_fill_ms',
               'coo_ms', 'product_symbolic_ms', 'product_numeric_ms', 'mu_ms',
               'matrix_preparation_ms', 'linear_total_ms']


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten(prefix, value, output):
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f'{prefix}_{key}' if prefix else key, item, output)
    else:
        output[prefix] = value


def summaries(records, methods, cases):
    result = []
    for case in cases:
        for method in methods:
            rows = [r for r in records if r['case'] == case and r['method'] == method and r['round'] > 0]
            first = rows[0]
            row = {k: first[k] for k in ['case', 'method', 'buses', 'states', 'iterations', 'linear_solves']}
            row.update(measured_runs=len(rows), converged_runs=sum(r['converged'] for r in rows),
                       residual_inf_max=max(r['residual_inf'] for r in rows),
                       max_voltage_difference=max(r['max_voltage_difference'] for r in rows))
            for field in TIME_FIELDS:
                row[field] = median(r[field] for r in rows)
            for field in ['matrix_preparation_ms', 'total_execution_ms']:
                row[field + '_min'] = min(r[field] for r in rows)
                row[field + '_max'] = max(r[field] for r in rows)
            row['product_ms'] = median(r['product_symbolic_ms'] + r['product_numeric_ms'] for r in rows)
            flatten('options', first['options'], row)
            result.append(row)
    return result


def export(root, primary, primary_env, ablation, ablation_env, methods, cases):
    folder = root / 'performance/data'
    folder.mkdir(exist_ok=True)
    raw = []
    for suite, records, env in [('lm-assembly', primary, primary_env), ('lm-ablation', ablation, ablation_env)]:
        for record in records:
            row = {'suite': suite}
            flatten('', record, row)
            row.update(git_commit=env['git_commit'], cpu=env['cpu_info'],
                       cpu_affinity=env['cpu_affinity'], rustc=env['rustc'], omp_threads=env['omp_threads'],
                       input_sha256=env['input_sha256'])
            raw.append(row)
    write_csv(folder / 'measurements.csv', raw)
    report_rows = summaries(primary, methods, cases)
    for row in report_rows:
        for baseline, name in [('AUG-COO-upper', 'coo_upper'), ('NE-rebuild', 'normal_generic')]:
            reference = next(r for r in report_rows if r['case'] == row['case'] and r['method'] == baseline)
            row[f'assembly_speedup_vs_{name}'] = reference['matrix_preparation_ms'] / row['matrix_preparation_ms']
            row[f'total_speedup_vs_{name}'] = reference['total_execution_ms'] / row['total_execution_ms']
    write_csv(folder / 'report.csv', report_rows)
    stage_rows = summaries(ablation, STAGES, cases)
    for row in stage_rows:
        baseline = next(r for r in stage_rows if r['case'] == row['case'] and r['method'] == 'AUG-COO-upper')
        independent = row['method'] == 'AUG-COO-upper'
        row['stage'] = STAGES[row['method']][0]
        row['standalone_jacobian_ms'] = row['j_or_aug_fill_ms'] if independent else 0.0
        row['direct_jt_evaluation_and_fill_ms'] = 0.0 if independent else row['j_or_aug_fill_ms']
        first = next(r for r in ablation if r['case'] == row['case'] and r['method'] == row['method'])
        row['jacobian_evaluations'] = first['jacobian_evaluations']
        row['coo_assemblies'] = first['coo_assemblies']
        row['assembly_speedup_vs_a0'] = baseline['matrix_preparation_ms'] / row['matrix_preparation_ms']
        row['total_speedup_vs_a0'] = baseline['total_execution_ms'] / row['total_execution_ms']
    write_csv(folder / 'ablation.csv', stage_rows)
    # Compact manifests preserve provenance without embedding a large source diff in every row.
    manifests = {name: {k: v for k, v in env.items() if k != 'source_diff'}
                 for name, env in [('lm-assembly', primary_env), ('lm-ablation', ablation_env)]}
    (folder / 'environment.json').write_text(json.dumps(manifests, indent=2, ensure_ascii=False) + '\n')
    write_latex(root, stage_rows, cases)
    return markdown(stage_rows, cases)


def markdown(rows, cases):
    lines = ['## 消融：从上三角COO到KKT直接填充', '',
             '独立入口`lm-ablation`运行现有三条驱动路径，各预热1次、测量7次。A0、A1、A2都使用KktPattern、相同增广上三角、相同初值与LM参数，并复用QDLDL。', '',
             '- A0：V4计算独立J，然后COO组装上三角并转CSC。',
             '- A1：原fill_jt_rows直接计算并写入最终Jᵀ块，消除独立J缓冲区和COO转换。',
             '- A2：JacobianOperator直接填入同一Jᵀ块，复用V3/V4的边导数关系，并读取当前Ybus。', '',
             '导数计算没有被消除：A1/A2把导数求值和最终矩阵写入合并，所以“独立J评估”列为0，但“直接Jᵀ评估与填充”仍有耗时。COO列包含三元组写入、μ及−I对角、转CSC、结构检查和数值复制；μ/右端列在A0只计右端，在A1/A2还计μ更新。准备合计逐轮求和后取中位数，不能把各分项中位数直接相加。', '',
             'A0评估还包含scalc/Vnorm准备；A1/A2复用残差阶段的scalc。A0→A1同时改变导数遍历和数据复用，分项计时能展示消失的COO阶段，但总差值不是“只删除COO、其他指令逐项相同”的实验。A1→A2进一步比较两种直接填充实现。', '']
    for case in cases:
        group = [r for r in rows if r['case'] == case]
        first = group[0]
        lines += [f'### {case}：组装分项', '',
                  f"三条路径均接受{first['iterations']}步，求解{first['linear_solves']}次。Jacobian评估{first['jacobian_evaluations']}次；A0的COO组装{first['coo_assemblies']}次，A1/A2为0次。", '',
                  '| 阶段 | 独立J评估 ms | 直接Jᵀ评估与填充 ms | COO ms | μ/右端 ms | 准备合计 ms | 总执行 ms |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in group:
            fields = ['standalone_jacobian_ms', 'direct_jt_evaluation_and_fill_ms', 'coo_ms',
                      'mu_ms', 'matrix_preparation_ms', 'total_execution_ms']
            lines.append(f"| {r['stage']} {STAGES[r['method']][1]} | " + ' | '.join(f'{r[k]:.3f}' for k in fields) + ' |')
        final = group[-1]
        lines += ['', f"A2相对A0：组装准备{final['assembly_speedup_vs_a0']:.2f}×，总执行{final['total_speedup_vs_a0']:.2f}×。", '']
    lines += ['CSV：`performance/data/measurements.csv`保留逐轮原始字段和参数；`report.csv`保存主报告表格；`ablation.csv`保存上述消融分项和加速比。最小/最大耗时也保留在汇总CSV中。', '']
    return lines


def write_latex(root, stages, cases):
    folder = root / 'paper/tables'
    folder.mkdir(parents=True, exist_ok=True)
    out = [r'\begin{table*}[t]', r'\centering', r'\caption{Upper-triangular KKT assembly ablation. Median cumulative times in ms over seven solves; A0 is V4 plus COO, A1 is the original direct row fill, and A2 is the borrowed operator.}',
           r'\label{tab:ablation}', r'\begin{tabular}{llrrrrrr}', r'\toprule',
           r'Case & Stage & Standalone $J$ & Direct $J^\top$ & COO & $\mu$/RHS & Preparation & Total \\', r'\midrule']
    for case in cases:
        for row in (r for r in stages if r['case'] == case):
            values = [row[k] for k in ['standalone_jacobian_ms', 'direct_jt_evaluation_and_fill_ms',
                                      'coo_ms', 'mu_ms', 'matrix_preparation_ms', 'total_execution_ms']]
            out.append(case.replace('_', r'\_') + ' & ' + row['stage'] + ' & ' + ' & '.join(f'{v:.3f}' for v in values) + r' \\')
        out.append(r'\addlinespace')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (folder / 'ablation.tex').write_text('\n'.join(out) + '\n')
    out = [r'\begin{table}[t]', r'\centering', r'\caption{A2/A0 comparison expressed as baseline time divided by final time. Counts include all accepted steps and linear solves.}',
           r'\label{tab:speedup}', r'\begin{tabular}{lrrrr}', r'\toprule',
           r'Case & Steps & Solves & Prep. & Total \\', r'\midrule']
    for r in stages:
        if r['stage'] == 'A2':
            out.append(r['case'].replace('_', r'\_') + f" & {r['iterations']} & {r['linear_solves']} & {r['assembly_speedup_vs_a0']:.2f}$\\times$ & {r['total_speedup_vs_a0']:.2f}$\\times$" + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (folder / 'speedup.tex').write_text('\n'.join(out) + '\n')
    r = next(r for r in stages if r['case'] == 'pegase9241' and r['stage'] == 'A2')
    base = next(r for r in stages if r['case'] == 'pegase9241' and r['stage'] == 'A0')
    macros = {'AssemblySpeedup': r['assembly_speedup_vs_a0'], 'TotalSpeedup': r['total_speedup_vs_a0'],
              'BaselineJacobian': base['standalone_jacobian_ms'], 'BaselineCoo': base['coo_ms'],
              'FinalFill': r['direct_jt_evaluation_and_fill_ms'], 'FinalTotal': r['total_execution_ms'],
              'BaselineTotal': base['total_execution_ms']}
    (folder / 'numbers.tex').write_text('\n'.join('\\newcommand{\\' + name + '}{' + f'{value:.2f}' + '}' for name, value in macros.items()) + '\n')
