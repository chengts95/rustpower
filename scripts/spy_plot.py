"""Sparsity-pattern isomorphism figure for the paper.

Loads IEEE 39, runs a power flow, reorders the buses into [PQ | PV | slack]
(completeness order: PQ first, PV next, slack last), and renders Ybus,
dS/dtheta, P*Ybus*P^T, and the reduced Jacobian J_red on a 2x2 spy plot. By
Theorem 1 the top row matrices share the same nonzero pattern; by
Corollary 1 the bottom row matrices share the same (N_pq + 2 N_pv)-square
pattern.

All bus-type reordering and projection are expressed as explicit sparse matrix
products (T @ Ybus @ T.T and P @ Jfull @ P.T) to match the paper formulation.
"""
import os
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as pn


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "paper", "sparsity_isomorphism.png"))


def dSbus_dV(Ybus, V):
    """MATPOWER complex sensitivities."""
    Ibus = Ybus @ V
    Vnorm = V / np.abs(V)
    diagV = sp.diags(V)
    diagIbus = sp.diags(Ibus)
    diagVnorm = sp.diags(Vnorm)
    dS_dVm = diagV @ (Ybus @ diagVnorm).conj() + diagIbus.conj() @ diagVnorm
    dS_dVa = 1j * diagV @ (diagIbus - Ybus @ diagV).conj()
    return dS_dVm.tocsc(), dS_dVa.tocsc()


def main():
    net = pn.case39()
    pp.runpp(net)

    ppc = net._ppc
    Ybus = ppc["internal"]["Ybus"].tocsc()

    Vm = ppc["bus"][:, 7]
    Va = np.deg2rad(ppc["bus"][:, 8])
    V = Vm * np.exp(1j * Va)

    btype = ppc["bus"][:, 1].astype(int)
    pv    = np.where(btype == 2)[0]
    pq    = np.where(btype == 1)[0]
    slack = np.where(btype == 3)[0]

    n     = Ybus.shape[0]
    npv   = len(pv)
    npq   = len(pq)
    npvpq = npv + npq
    m     = npvpq + npq          # dimension of Jred: (Npvpq + Npq)

    # ── Permutation T: reorders buses to [PQ | PV | slack] ──────────────────
    # T_mat is an n×n sparse permutation matrix; T_mat @ x permutes bus vector x.
    T_perm = np.concatenate([pq, pv, slack])
    T_mat  = sp.eye(n, format='csc')[T_perm, :]
    TYT    = (T_mat @ Ybus @ T_mat.T).tocsc()   # T·Ybus·T^T
    V_p    = T_mat @ V                           # T·V (complex)

    # ── Full 2n×2n Jacobian assembled on the permuted system ─────────────────
    dS_dVm_p, dS_dVa_p = dSbus_dV(TYT, V_p)
    Jfull = sp.bmat([[dS_dVa_p.real, dS_dVm_p.real],
                     [dS_dVa_p.imag, dS_dVm_p.imag]], format='csc')

    # ── Projection P: m×2n ───────────────────────────────────────────────────
    # Selects the active rows/cols from Jfull to form Jred:
    #   rows 0..npvpq-1     ← Jfull rows 0..npvpq-1    (∂P equations, pvpq buses)
    #   rows npvpq..m-1     ← Jfull rows n..n+npq-1    (∂Q equations, pq buses)
    # Same selection applies to columns (θ for pvpq, |V| for pq).
    col_P = np.concatenate([np.arange(npvpq), n + np.arange(npq)])
    P     = sp.csc_matrix((np.ones(m), (np.arange(m), col_P)), shape=(m, 2 * n))
    Jred  = (P @ Jfull @ P.T).tocsc()

    # ── Structural projection P_Y: m×n ───────────────────────────────────────
    # Same logical row/column selection applied to the n×n space of TYT:
    #   rows 0..npvpq-1     ← TYT rows 0..npvpq-1
    #   rows npvpq..m-1     ← TYT rows 0..npq-1   (pq is the leading block after T)
    col_PY = np.concatenate([np.arange(npvpq), np.arange(npq)])
    P_Y    = sp.csc_matrix((np.ones(m), (np.arange(m), col_PY)), shape=(m, n))
    Yred   = (P_Y @ TYT @ P_Y.T).tocsc()

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))
    panels = [
        (TYT,      r"$\mathbf{T}\mathbf{Y}_{\mathrm{bus}}\mathbf{T}^{\!\top}$"),
        (dS_dVa_p, r"$\partial \mathbf{S}_{\mathrm{bus}} / \partial \boldsymbol{\theta}$"),
        (Yred,     r"$\mathbf{P}\,(\mathbf{T}\mathbf{Y}_{\mathrm{bus}}\mathbf{T}^{\!\top})\,\mathbf{P}^{\!\top}$"),
        (Jred,     r"$\mathbf{J}_{\mathrm{red}}$"),
    ]
    for ax, (M, t) in zip(axes.flat, panels):
        ax.spy(M, markersize=4, color="black")
        ax.set_title(t, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT}")
    print(f"  n={n}  npq={npq}  npv={npv}  m={m}  (Jred: {m}×{m})")
    print(f"  TYT nnz={TYT.nnz}  Jred nnz={Jred.nnz}  Yred nnz={Yred.nnz}")


if __name__ == "__main__":
    main()
