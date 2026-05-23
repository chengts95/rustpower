#!/usr/bin/env python3
"""
scripts/plot_solve_breakdown.py

Per-solve cost breakdown: Assembly vs KLU-init vs KLU-iterative.

Three panels (IEEE 39, IEEE 118, PEGASE 9241).  Each panel shows V0/V1/V2
as horizontal stacked bars:

    Assembly   (dark, hatch)  — N_iter × per-iter assembly + symbolic build (V2)
    KLU init   (medium, none) — klu_l_analyze + klu_l_factor  (once per solve)
    KLU steady (light, none)  — (N_iter-1) × refactor + N_iter × back-sub

KLU init and KLU steady are version-invariant (same bars for V0/V1/V2);
only the Assembly bar shrinks V0→V2.

Data source: paper/solve_breakdown.csv (written by bench_jacobian_fill Rust test).

Usage:
    python scripts/plot_solve_breakdown.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path

df = pd.read_csv('paper/solve_breakdown.csv')

SYSTEMS   = ['IEEE 39', 'IEEE 118', 'PEGASE 9241']
VERSIONS  = ['V0', 'V1', 'V2']
SYS_TITLE = {'IEEE 39': 'IEEE 39', 'IEEE 118': 'IEEE 118', 'PEGASE 9241': 'PEGASE 9241'}

VER_LABEL_FULL  = {
    'V0': 'V0  (MATPOWER)',
    'V1': "V1  (Sch\xe4fer '18)",
    'V2': 'V2  (this work)',
}
VER_LABEL_SHORT = {'V0': 'V0', 'V1': 'V1', 'V2': 'V2'}

# ─── Styles ────────────────────────────────────────────────────────────────────
STYLE_ASM  = dict(facecolor='0.20', hatch='////', edgecolor='white',  linewidth=0.0)
STYLE_INIT = dict(facecolor='0.55', hatch='',     edgecolor='0.40',   linewidth=0.6)
STYLE_ITER = dict(facecolor='0.82', hatch='',     edgecolor='0.40',   linewidth=0.6)

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         8,
    'axes.linewidth':    0.55,
    'xtick.major.width': 0.55,
    'ytick.major.width': 0.55,
    'xtick.major.size':  3,
    'hatch.linewidth':   0.55,
})

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6))
fig.subplots_adjust(wspace=0.32, bottom=0.35)

BAR_H = 0.50
Y = {'V0': 2, 'V1': 1, 'V2': 0}

for idx, (ax, sys_name) in enumerate(zip(axes, SYSTEMS)):
    sys_df = df[df['system'] == sys_name].set_index('version')

    # Per-solve segment totals (ms) for each version.
    segs = {}
    for ver in VERSIONS:
        row = sys_df.loc[ver]
        n   = row['n_iter']
        # Assembly: N×per-iter + one-time symbolic build (sym_us=0 for V0/V1).
        asm_ms   = (n * row['asm_us'] + row['sym_us']) / 1e3
        # KLU one-time per solve: analyze + first full factor.
        init_ms  = (row['analyze_us'] + row['factor_us']) / 1e3
        # KLU iterative: (N-1) refactors + N back-subs.
        iter_ms  = ((n - 1) * row['refactor_us'] + n * row['backsolve_us']) / 1e3
        segs[ver] = (asm_ms, init_ms, iter_ms)

    x_max = max(sum(s) for s in segs.values())

    for ver in VERSIONS:
        y              = Y[ver]
        asm, init, itr = segs[ver]
        total          = asm + init + itr

        # Assembly bar.
        ax.barh(y, asm, height=BAR_H, left=0, zorder=3, **STYLE_ASM)
        # KLU init bar.
        ax.barh(y, init, height=BAR_H, left=asm, zorder=3, **STYLE_INIT)
        # KLU steady bar.
        ax.barh(y, itr, height=BAR_H, left=asm + init, zorder=3, **STYLE_ITER)
        # Outer border.
        ax.barh(y, total, height=BAR_H,
                facecolor='none', edgecolor='0.30', linewidth=0.6, zorder=4)

        # Assembly % label (white, inside bar, only when wide enough).
        pct = asm / total * 100
        if asm > x_max * 0.08:
            ax.text(asm / 2, y, f'{pct:.0f}%',
                    ha='center', va='center', fontsize=5.8,
                    color='white', fontweight='bold', zorder=5)

    ax.set_title(SYS_TITLE[sys_name], fontsize=8.5, fontweight='bold', pad=3)
    ax.set_yticks([0, 1, 2])
    if idx == 0:
        ax.set_yticklabels(
            [VER_LABEL_FULL['V2'], VER_LABEL_FULL['V1'], VER_LABEL_FULL['V0']],
            fontsize=6.8)
    else:
        ax.set_yticklabels(
            [VER_LABEL_SHORT['V2'], VER_LABEL_SHORT['V1'], VER_LABEL_SHORT['V0']],
            fontsize=7)
    ax.set_xlabel('Per-solve time (ms)', fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlim(0, x_max * 1.08)
    ax.grid(axis='x', linestyle=':', linewidth=0.4, color='0.65', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ─── Legend ────────────────────────────────────────────────────────────────────
h_asm  = mpatches.Patch(facecolor='0.20', hatch='////', edgecolor='0.40',
                         linewidth=0.55, label='Jacobian assembly  (N \xd7 per-iter + symbolic build)')
h_init = mpatches.Patch(facecolor='0.55', hatch='',    edgecolor='0.40',
                         linewidth=0.55, label='KLU init: analyze + factor  (once per solve, version-invariant)')
h_iter = mpatches.Patch(facecolor='0.82', hatch='',    edgecolor='0.40',
                         linewidth=0.55, label='KLU iterative: re-factor + back-sub  (version-invariant)')

fig.legend(handles=[h_asm, h_init, h_iter],
           loc='lower center', ncol=1, fontsize=6.5,
           frameon=False, bbox_to_anchor=(0.5, 0.01),
           handlelength=2.4, handletextpad=0.6)

fig.text(
    0.015, -0.02,
    'Per-solve time reconstructed from directly instrumented component averages '
    '(bench_jacobian_fill, export_solve_breakdown_csv).  '
    'KLU segments are identical for V0/V1/V2.',
    fontsize=5.5, color='0.45')

# ─── Save ──────────────────────────────────────────────────────────────────────
out = Path('paper/solve_breakdown')
fig.savefig(str(out) + '.pdf', bbox_inches='tight', dpi=300)
fig.savefig(str(out) + '.png', bbox_inches='tight', dpi=300)
print(f'Saved {out}.pdf and {out}.png')

# ─── Console summary ───────────────────────────────────────────────────────────
print()
print('Assembly share of estimated per-solve time:')
for sys_name in SYSTEMS:
    sys_df = df[df['system'] == sys_name].set_index('version')
    parts = {}
    for ver in VERSIONS:
        row = sys_df.loc[ver]
        n   = row['n_iter']
        asm = (n * row['asm_us'] + row['sym_us']) / 1e3
        tot = asm + (row['analyze_us'] + row['factor_us']) / 1e3 \
                  + ((n-1)*row['refactor_us'] + n*row['backsolve_us']) / 1e3
        parts[ver] = asm / tot * 100
    print(f'  {sys_name}: V0={parts["V0"]:.0f}%  V1={parts["V1"]:.0f}%  V2={parts["V2"]:.0f}%')
