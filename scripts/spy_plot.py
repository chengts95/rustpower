"""Sparsity-pattern isomorphism figure for the paper.

Loads IEEE 39, runs a power flow, reorders the buses into [PQ | PV | slack]
(completeness order: PQ first, PV next, slack last), and renders Ybus,
dS/dtheta, P*Ybus*P^T, and the reduced Jacobian J_red on a 2x2 spy plot. By
Theorem 1 the top row matrices share the same nonzero pattern; by
Corollary 1 the bottom row matrices share the same (N_pq + 2 N_pv)-square
pattern.
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
    pv = np.where(btype == 2)[0]
    pq = np.where(btype == 1)[0]
    slack = np.where(btype == 3)[0]

    # Apply T: reorder buses into [PQ, PV, slack] (completeness order, the
    # paper's Section II anchor).
    T = np.concatenate([pq, pv, slack])
    Ybus_p = Ybus[T, :][:, T]
    V_p = V[T]

    npv = len(pv)
    npq = len(pq)

    dS_dVm, dS_dVa = dSbus_dV(Ybus_p, V_p)

    # After permutation: pq buses are 0..npq-1, pv are npq..npq+npv-1.
    pq_i = np.arange(0, npq)
    pvpq_i = np.arange(0, npq + npv)

    j11 = dS_dVa[pvpq_i, :][:, pvpq_i].real
    j12 = dS_dVm[pvpq_i, :][:, pq_i].real
    j21 = dS_dVa[pq_i, :][:, pvpq_i].imag
    j22 = dS_dVm[pq_i, :][:, pq_i].imag
    Jred = sp.bmat([[j11, j12], [j21, j22]], format="csc")

    # Y_red: the same 2x2 stacking/projection applied to Y_bus's pattern. By
    # Corollary 1 this is the structural reference Jred should be compared to,
    # not Y_bus itself (Jred is not n*n and is not symmetric the same way).
    y11 = Ybus_p[pvpq_i, :][:, pvpq_i]
    y12 = Ybus_p[pvpq_i, :][:, pq_i]
    y21 = Ybus_p[pq_i, :][:, pvpq_i]
    y22 = Ybus_p[pq_i, :][:, pq_i]
    Yred = sp.bmat([[y11, y12], [y21, y22]], format="csc")

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))
    panels = [
        (Ybus_p, r"$\mathbf{Y}_{\mathrm{bus}}$"),
        (dS_dVa, r"$\partial \mathbf{S}_{\mathrm{bus}} / \partial \boldsymbol{\theta}$"),
        (Yred,   r"$\mathcal{P}\,\mathbf{Y}_{\mathrm{bus}}\,\mathcal{P}^{\!\top}$"),
        (Jred,   r"$\mathbf{J}_{\mathrm{red}}$"),
    ]
    for ax, (M, t) in zip(axes.flat, panels):
        ax.spy(M, markersize=4, color="black")
        ax.set_title(t, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
