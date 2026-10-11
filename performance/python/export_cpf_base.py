"""Export a CPF base record (alpha=1) for a MATPOWER .m case.

Self-contained: parses the .m directly (no matpowercaseframes dependency),
converts via pandapower's from_ppc, captures pypower's internal reduced
inputs (Ybus CSC in [PQ|PV|slack] order, S, flat V) with the same
monkey-patch as robustness_pp.py, and writes a single-base-record jsonl:

  target/research/lm_robustness/<case>.jsonl   (first line: {"base": {...}})

The Rust side (src/cpf/tests_jsonl.rs) replays CPF on this record.

Usage:
  python performance/python/export_cpf_base.py target/research/case2383wp.m
"""
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandapower as pp
from pandapower.converter.matpower import from_ppc

TOL = 1e-8
MAX_ITER = 300


def parse_matpower_m(path: Path) -> dict:
    """Minimal MATPOWER v2 .m parser: numeric mpc.{bus,gen,branch,gencost} blocks."""
    text = path.read_text(encoding="utf-8", errors="replace")

    def block(name):
        m = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.S)
        if m is None:
            return None
        body = re.sub(r"%[^\n]*", "", m.group(1))  # strip comments
        body = body.replace("...", " ")            # line continuations
        rows = []
        for chunk in re.split(r"[;\n]", body):
            chunk = chunk.strip()
            if chunk:
                rows.append([float(x) for x in chunk.split()])
        return np.array(rows, dtype=float)

    base = re.search(r"mpc\.baseMVA\s*=\s*([\d.eE+-]+)", text)
    ppc = {
        "version": "2",
        "baseMVA": float(base.group(1)),
        "bus": block("bus"),
        "gen": block("gen"),
        "branch": block("branch"),
    }
    gencost = block("gencost")
    if gencost is not None:
        ppc["gencost"] = gencost
    return ppc


def capture_base(net):
    """Run one flat-start NR at alpha=1 and capture the exact reduced inputs."""
    import importlib
    mod = importlib.import_module("pandapower.pf.run_newton_raphson_pf")
    original = mod.newtonpf
    captured = {}

    def capture(Y, S, V, ref, pv, pq, ppci, options, *args):
        order = np.r_[pq, pv, ref].astype(int)
        y = Y.tocsc()[order, :][:, order].tocsc()
        y.sort_indices()
        captured.update(dict(
            nb=len(order), npq=len(pq), npv=len(pv),
            cp=y.indptr.tolist(), ri=y.indices.tolist(),
            y_re=y.data.real.tolist(), y_im=y.data.imag.tolist(),
            s_re=S[order].real.tolist(), s_im=S[order].imag.tolist(),
            v_flat_re=V[order].real.tolist(), v_flat_im=V[order].imag.tolist(),
        ))
        t = perf_counter()
        result = original(Y, S, V, ref, pv, pq, ppci, options, *args)
        captured["nr_converged"] = bool(result[1])
        captured["nr_iterations"] = int(result[2])
        captured["nr_core_ms"] = (perf_counter() - t) * 1000
        return result

    mod.newtonpf = capture
    try:
        pp.runpp(net, init="flat", max_iteration=MAX_ITER, tolerance_mva=TOL,
                 enforce_q_lims=False, distributed_slack=False, numba=False)
    except pp.LoadflowNotConverged:
        pass
    finally:
        mod.newtonpf = original
    return captured


def main():
    m_path = Path(sys.argv[1])
    case = m_path.stem
    ppc = parse_matpower_m(m_path)
    net = from_ppc(ppc)
    cap = capture_base(net)
    base_keys = ["nb", "npq", "npv", "cp", "ri", "y_re", "y_im",
                 "s_re", "s_im", "v_flat_re", "v_flat_im"]
    base = {k: cap[k] for k in base_keys}
    base["case"] = case
    out = Path(__file__).resolve().parents[2] / "target/research/lm_robustness" / f"{case}.jsonl"
    out.parent.mkdir(exist_ok=True, parents=True)
    with out.open("w") as fh:
        fh.write(json.dumps({"base": base}) + "\n")
    print(f"{case}: nb={base['nb']} npv={base['npv']} npq={base['npq']} "
          f"nnz={len(base['y_re'])} | base NR converged={cap['nr_converged']} "
          f"it={cap['nr_iterations']} {cap['nr_core_ms']:.1f}ms -> {out}")


if __name__ == "__main__":
    main()
