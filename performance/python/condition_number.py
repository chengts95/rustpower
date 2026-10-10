"""Condition-number certificate for the reduced power-flow Jacobian at the
converged solution, as a function of loading factor alpha.

For each alpha: build the net (MATPOWER .m via CaseFrames or
pandapower.networks), scale non-slack injections, solve with pandapower NR,
rebuild the exact reduced Jacobian J* = d(P,Q)/d(theta,|V|) at the solution
with pypower's dSbus_dV, then estimate sigma_max by power iteration and
sigma_min by inverse power iteration (sparse LU, KLU-class cost).

Usage:
  PYTHONPATH=target/research/pylibs python performance/python/condition_number.py \
      --mpc target/research/case2383wp.m --alphas 1.0 1.4 1.6 1.8 1.85
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from scipy import sparse
from scipy.sparse import linalg as spla

parser = argparse.ArgumentParser()
parser.add_argument("--mpc", type=Path)
parser.add_argument("--case", default=None)
parser.add_argument("--alphas", type=float, nargs="+", required=True)
parser.add_argument("--tol", type=float, default=1e-10)
cli = parser.parse_args()

import importlib
mod = importlib.import_module("pandapower.pf.run_newton_raphson_pf")
original = mod.newtonpf
from pandapower.pypower.dSbus_dV import dSbus_dV


def build_net():
    if cli.mpc is not None:
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
    return getattr(pn, cli.case)()


def scale_injections(net, alpha):
    if len(net.load):
        net.load["p_mw"] *= alpha
        net.load["q_mvar"] *= alpha
    if len(net.sgen):
        net.sgen["p_mw"] *= alpha
        net.sgen["q_mvar"] *= alpha
    if len(net.gen):
        net.gen["p_mw"] *= alpha


def solve_and_jacobian(net):
    """Solve NR (flat start); return (converged, J_reduced at solution)."""
    captured = {}

    def capture(Y, S, V, ref, pv, pq, ppci, options, *args):
        result = original(Y, S, V, ref, pv, pq, ppci, options, *args)
        v_star = result[0]
        dS_dVm, dS_dVa = dSbus_dV(Y, v_star)
        pvpq = np.r_[pq, pv]
        j11 = dS_dVa[np.ix_(pvpq, pvpq)].real
        j12 = dS_dVm[np.ix_(pvpq, pq)].real
        j21 = dS_dVa[np.ix_(pq, pvpq)].imag
        j22 = dS_dVm[np.ix_(pq, pq)].imag
        captured["J"] = sparse.vstack(
            [sparse.hstack([j11, j12]), sparse.hstack([j21, j22])]).tocsc()
        captured["converged"] = bool(result[1])
        return result

    mod.newtonpf = capture
    try:
        try:
            pp.runpp(net, init="flat", max_iteration=300, tolerance_mva=1e-8,
                     enforce_q_lims=False, distributed_slack=False, numba=False)
        except pp.LoadflowNotConverged:
            pass
    finally:
        mod.newtonpf = original
    return captured.get("converged", False), captured.get("J")


def sigma_max(J, iters=200):
    n = J.shape[0]
    x = np.random.default_rng(0).standard_normal(n)
    for _ in range(iters):
        x = J.T @ (J @ x)
        x /= np.linalg.norm(x)
    return float(np.linalg.norm(J @ x))


def sigma_min(J, iters=200):
    n = J.shape[0]
    lu = spla.splu(J)
    x = np.random.default_rng(1).standard_normal(n)
    for _ in range(iters):
        x = lu.solve(x)          # J^{-1} x
        nrm = np.linalg.norm(x)
        x /= nrm
        # Rayleigh 商估计 1/sigma_min：取上次增长因子即可，最终用一次 J^T 修正
    return 1.0 / float(np.linalg.norm(lu.solve(x)) / np.linalg.norm(x))


net0 = build_net()
print(f"alpha  converged  sigma_max      sigma_min      cond(J*)        dim")
for alpha in cli.alphas:
    net = copy.deepcopy(net0)
    scale_injections(net, alpha)
    conv, J = solve_and_jacobian(net)
    if not conv or J is None:
        print(f"{alpha:5.2f}  no")
        continue
    smax = sigma_max(J)
    smin = sigma_min(J)
    print(f"{alpha:5.2f}  yes       {smax:12.4e}  {smin:12.4e}  {smax/smin:12.4e}  {J.shape[0]}")
