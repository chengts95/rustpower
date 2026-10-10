"""Export pandapower matched-input references for the rustpower lm-robustness sweep.

For each case and each loading factor alpha, this script:
  1. builds a fresh pandapower network, scales non-slack injections by alpha
     (load p/q, sgen p/q, gen p — exactly the net-injection scaling S*alpha
     the Rust sweep applies);
  2. runs pandapower NR from flat and from its own DC start, capturing the
     exact internal inputs (Ybus CSC in [PQ|PV|slack] order, S, V) via a
     monkey-patch of pypower's newtonpf, and the exact pandapower results;
  3. writes one base record (structure, base injections, flat start) plus
     one line per alpha (DC start + pandapower outcomes) to
     target/research/lm_robustness/<case>.jsonl.

The Rust side (`lm-robustness --import <case>`) replays the same grid with
identical inputs and compares walls, starts, and converged voltages.

Run with the pandapower environment, e.g.:
  .venv/bin/python performance/python/robustness_pp.py
  .venv/bin/python performance/python/robustness_pp.py --case case300

No reactive-limit switching, distributed slack, or ZIP loads; the stopping
tolerance is passed as 1e-8 (pandapower's tolerance_mva field) on both sides.
"""
import argparse
import copy
import importlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makeBdc import makeBdc

DEFAULT_CASES = ["case39", "case118", "case300", "case6470rte", "case6495rte", "case6515rte"]
ALPHA_STEP = 0.05
ALPHA_MAX = 3.0
TOL = 1e-8
MAX_ITER = 300

parser = argparse.ArgumentParser()
parser.add_argument("--case", action="append", help="pandapower.networks case name; repeatable")
parser.add_argument("--mpc", type=Path, help="MATPOWER .m file (e.g. case2383wp.m); overrides --case")
parser.add_argument("--alpha-max", type=float, default=ALPHA_MAX)
parser.add_argument("--alpha-step", type=float, default=ALPHA_STEP)
parser.add_argument("--alpha-min", type=float, default=1.0,
                    help="start alpha; base injections are recovered by linear division")
parser.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parents[2] / "target/research/lm_robustness")
cli = parser.parse_args()
cli.output.mkdir(exist_ok=True, parents=True)

mod = importlib.import_module("pandapower.pf.run_newton_raphson_pf")
original = mod.newtonpf


def scale_injections(net, alpha):
    """Net non-slack injection x alpha: matches the Rust sweep's S_spec x alpha."""
    if len(net.load):
        net.load["p_mw"] *= alpha
        net.load["q_mvar"] *= alpha
    if len(net.sgen):
        net.sgen["p_mw"] *= alpha
        net.sgen["q_mvar"] *= alpha
    if len(net.gen):
        net.gen["p_mw"] *= alpha  # PV reactive output is not a specification


def run_once(net, init):
    """Run pandapower NR; capture exact reduced inputs and the result."""
    captured = {}

    def capture(Y, S, V, ref, pv, pq, ppci, options, *args):
        order = np.r_[pq, pv, ref].astype(int)
        assert len(np.unique(order)) == Y.shape[0]
        y = Y.tocsc()[order, :][:, order].tocsc()
        y.sort_indices()
        captured.update(dict(
            nb=len(order), npq=len(pq), npv=len(pv),
            cp=y.indptr.tolist(), ri=y.indices.tolist(),
            y_re=y.data.real.tolist(), y_im=y.data.imag.tolist(),
            s_re=S[order].real.tolist(), s_im=S[order].imag.tolist(),
            v_start_re=V[order].real.tolist(), v_start_im=V[order].imag.tolist(),
        ))
        # DC 相移注入（与 rustpower ECS build_dcpf_model 同公式），供 import 模式
        # 复现 pandapower 的 DC 启动；同一 permuted 母线顺序。
        _, _, pbusinj, _, _ = makeBdc(ppci["bus"], ppci["branch"])
        captured["p_shift"] = np.asarray(pbusinj).ravel()[order].tolist()
        t = perf_counter()
        result = original(Y, S, V, ref, pv, pq, ppci, options, *args)
        elapsed = perf_counter() - t
        mis = result[0] * np.conj(Y @ result[0]) - S
        residual = np.max(np.abs(np.r_[mis[np.r_[pq, pv]].real, mis[pq].imag]))
        captured["result"] = dict(
            converged=bool(result[1]), iterations=int(result[2]),
            residual_inf=float(residual) if np.isfinite(residual) else None,
            core_ms=elapsed * 1000,
            v_re=[float(x) if np.isfinite(x) else None for x in result[0][order].real],
            v_im=[float(x) if np.isfinite(x) else None for x in result[0][order].imag],
        )
        return result

    mod.newtonpf = capture
    try:
        try:
            pp.runpp(net, init=init, max_iteration=MAX_ITER, tolerance_mva=TOL,
                     enforce_q_lims=False, distributed_slack=False, numba=False)
        except pp.LoadflowNotConverged:
            pass  # capture() 已记录 converged=False 的完整结果
    finally:
        mod.newtonpf = original
    return captured


def build_net(case):
    if cli.mpc is not None:
        # MATPOWER .m → CaseFrames 解析 → from_ppc；np.array 显式拷贝
        # （matpowercaseframes 返回只读数组，pandapower 内部要原地改索引）。
        from matpowercaseframes import CaseFrames
        from pandapower.converter.matpower import from_ppc
        cf = CaseFrames(str(cli.mpc))
        ppc = {
            "version": "2",
            "baseMVA": float(cf.baseMVA),
            "bus": np.array(cf.bus.values, dtype=float),
            "gen": np.array(cf.gen.values, dtype=float),
            "branch": np.array(cf.branch.values, dtype=float),
            "gencost": np.array(cf.gencost.values, dtype=float),
        }
        return from_ppc(ppc)
    return getattr(pn, case)()


if cli.mpc is not None:
    cases = [cli.mpc.stem]
else:
    cases = cli.case or DEFAULT_CASES

for case in cases:
    net0 = build_net(case)
    base = None
    lines = []
    alpha = cli.alpha_min
    print(f"=== {case} ===")
    while alpha <= cli.alpha_max + 1e-9:
        row = {"alpha": round(alpha, 4)}
        for init in ["flat", "dc"]:
            net = copy.deepcopy(net0)
            scale_injections(net, alpha)
            cap = run_once(net, init)
            if base is None and init == "flat":
                base = {k: cap[k] for k in
                        ["nb", "npq", "npv", "cp", "ri", "y_re", "y_im", "s_re", "s_im", "p_shift"]}
                base["case"] = case
                base["v_flat_re"] = cap["v_start_re"]
                base["v_flat_im"] = cap["v_start_im"]
                if cli.alpha_min != 1.0:
                    # 注入随 α 线性缩放，除回 α=1.0 基值（Rust 端按行 α 重新缩放）
                    base["s_re"] = [x / cli.alpha_min for x in base["s_re"]]
                    base["s_im"] = [x / cli.alpha_min for x in base["s_im"]]
            if init == "dc":
                row["dc_start_re"] = cap["v_start_re"]
                row["dc_start_im"] = cap["v_start_im"]
            row[init] = cap["result"]
        lines.append(row)
        f, d = row["flat"], row["dc"]
        print(f"  a={alpha:5.2f} | flat: {'ok' if f['converged'] else 'XX'} it={f['iterations']:3}"
              f" | dc: {'ok' if d['converged'] else 'XX'} it={d['iterations']:3}")
        if not f["converged"] and not d["converged"] and alpha > 1.05:
            break
        alpha += cli.alpha_step

    path = cli.output / f"{case}.jsonl"
    with path.open("w") as fh:
        fh.write(json.dumps({"base": base}) + "\n")
        for row in lines:
            fh.write(json.dumps(row) + "\n")
    print(f"  -> {path} ({len(lines)} alpha rows)")
