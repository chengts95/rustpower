"""Poor-man's continuation past the NR wall on case2383wp.

Solves alpha0 with pandapower (captures exact reduced inputs), then chains
a plain Newton corrector in original-bus-order form from the previous
solution through increasing alphas. Reports per-alpha: converged, Newton
iterations from previous solution, and sigma_min(J*) / cond at the solution.

Purpose: decide whether the alpha>=1.9 failure is a saddle-node (no
solution) or a convergence-domain failure (solution exists, NR from
flat/dc cannot reach it).

  PYTHONPATH=target/research/pylibs python performance/python/continuation_probe.py \
      --mpc target/research/case2383wp.m --alpha0 1.85 --chain 1.87 1.90 1.92 1.95 2.0
"""
import argparse
import copy
import importlib
from pathlib import Path

import numpy as np
import pandapower as pp
from scipy import sparse
from scipy.sparse import linalg as spla

parser = argparse.ArgumentParser()
parser.add_argument("--mpc", type=Path)
parser.add_argument("--case", help="pandapower.networks case name (e.g. case6470rte)")
parser.add_argument("--init", default="flat", choices=["flat", "dc"],
                    help="start for the alpha0 anchor solve")
parser.add_argument("--alpha0", type=float, required=True)
parser.add_argument("--chain", type=float, nargs="+", required=True)
cli = parser.parse_args()

from pandapower.pypower.dSbus_dV import dSbus_dV

mod = importlib.import_module("pandapower.pf.run_newton_raphson_pf")
original = mod.newtonpf

if cli.mpc is not None:
    from matpowercaseframes import CaseFrames
    from pandapower.converter.matpower import from_ppc
    cf = CaseFrames(str(cli.mpc))
    ppc = {"version": "2", "baseMVA": float(cf.baseMVA),
           "bus": np.array(cf.bus.values, dtype=float),
           "gen": np.array(cf.gen.values, dtype=float),
           "branch": np.array(cf.branch.values, dtype=float),
           "gencost": np.array(cf.gencost.values, dtype=float)}
    net0 = from_ppc(ppc)
else:
    import pandapower.networks as pn
    net0 = getattr(pn, cli.case)()


def scale_injections(net, alpha):
    if len(net.load):
        net.load["p_mw"] *= alpha
        net.load["q_mvar"] *= alpha
    if len(net.sgen):
        net.sgen["p_mw"] *= alpha
        net.sgen["q_mvar"] *= alpha
    if len(net.gen):
        net.gen["p_mw"] *= alpha


def solve_capture(net):
    captured = {}

    def capture(Y, S, V, ref, pv, pq, ppci, options, *args):
        order = np.r_[pq, pv, ref].astype(int)
        result = original(Y, S, V, ref, pv, pq, ppci, options, *args)
        captured.update(Y=Y.tocsr(), S=S.copy(), pq=pq, pv=pv, order=order,
                        v=result[0].copy(), converged=bool(result[1]))
        return result

    mod.newtonpf = capture
    try:
        try:
            pp.runpp(net, init=cli.init, max_iteration=300, tolerance_mva=1e-8,
                     enforce_q_lims=False, distributed_slack=False, numba=False)
        except pp.LoadflowNotConverged:
            pass
    finally:
        mod.newtonpf = original
    return captured


def reduced_jacobian(Y, V, pq, pvpq):
    dS_dVm, dS_dVa = dSbus_dV(Y, V)
    j11 = dS_dVa[np.ix_(pvpq, pvpq)].real
    j12 = dS_dVm[np.ix_(pvpq, pq)].real
    j21 = dS_dVa[np.ix_(pq, pvpq)].imag
    j22 = dS_dVm[np.ix_(pq, pq)].imag
    return sparse.vstack([sparse.hstack([j11, j12]),
                          sparse.hstack([j21, j22])]).tocsc()


def newton(Y, S, V0, pq, pvpq, tol=1e-10, maxit=60):
    V = V0.copy()
    for it in range(1, maxit + 1):
        mis = V * np.conj(Y @ V) - S
        F = np.r_[mis[pvpq].real, mis[pq].imag]
        if np.max(np.abs(F)) < tol:
            return V, it - 1, True
        J = reduced_jacobian(Y, V, pq, pvpq)
        dx = spla.spsolve(J, F)
        if not np.all(np.isfinite(dx)):
            return V, it, False
        V[pvpq] *= np.exp(-1j * dx[:len(pvpq)])
        V[pq] = V[pq] / np.abs(V[pq]) * (np.abs(V[pq]) - dx[len(pvpq):])
        if np.min(np.abs(V)) < 1e-6 or np.max(np.abs(V)) > 10:
            return V, it, False
    mis = V * np.conj(Y @ V) - S
    F = np.r_[mis[pvpq].real, mis[pq].imag]
    return V, maxit, bool(np.max(np.abs(F)) < tol)


def sigma_min(J, iters=150):
    lu = spla.splu(J)
    x = np.random.default_rng(1).standard_normal(J.shape[0])
    for _ in range(iters):
        x = lu.solve(x)
        x /= np.linalg.norm(x)
    return 1.0 / float(np.linalg.norm(lu.solve(x)))


def sigma_max(J, iters=150):
    x = np.random.default_rng(0).standard_normal(J.shape[0])
    for _ in range(iters):
        x = J.T @ (J @ x)
        x /= np.linalg.norm(x)
    return float(np.linalg.norm(J @ x))


net = copy.deepcopy(net0)
scale_injections(net, cli.alpha0)
cap = solve_capture(net)
assert cap["converged"], f"alpha0={cli.alpha0} 未收敛，无法启动延拓"
Y, pq, pv = cap["Y"], cap["pq"], cap["pv"]
pvpq = np.r_[pq, pv]
S0 = cap["S"]
v = cap["v"]
print(f"alpha0={cli.alpha0:.2f} ok；链式延拓（上一步解作初值，tol=1e-10）")
print(f"alpha   conv  nrit  smin(J*)       cond(J*)       min|V|")
for alpha in cli.chain:
    S = S0 * (alpha / cli.alpha0)  # 注入线性缩放（slack 行不参与失配，无影响）
    v, it, conv = newton(Y, S, v, pq, pvpq)
    if conv:
        J = reduced_jacobian(Y, v, pq, pvpq)
        smin, smax = sigma_min(J), sigma_max(J)
        print(f"{alpha:5.2f}   yes   {it:3d}  {smin:12.4e}  {smax/smin:12.4e}  {np.min(np.abs(v)):6.3f}")
    else:
        print(f"{alpha:5.2f}   NO    {it:3d}  — 延拓失败（从上一步解出发也不收敛）")
        break
