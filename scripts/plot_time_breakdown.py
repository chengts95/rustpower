#!/usr/bin/env python3
"""
scripts/plot_time_breakdown.py

Per-Newton-iteration cost split: Assembly vs LU re-factorisation.

Three panels (IEEE 39, IEEE 118, PEGASE 9241).  Each panel shows V0 / V1 / V2
as horizontal stacked bars:
    Assembly  (dark, hatch) — Jacobian fill, shrinks V0→V2
    LU        (light, none) — KLU numeric re-factorisation, constant across versions

The LU bar stays the same width; the Assembly bar shrinks dramatically, making
LU the dominant cost by V2.

N_iter per solve is back-computed from the measured total times and per-call
assembly times under the assumption that LU cost is version-invariant:
    N_iter = (T_total_V0 − T_total_V2) / (ASM_V0 − ASM_V2)     [ms / ms]

Usage (from repo root):
    python scripts/plot_time_breakdown.py

Outputs:
    paper/time_breakdown.pdf
    paper/time_breakdown.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─── Directly instrumented data ────────────────────────────────────────────────
# Source: compare_klu_breakdown() in src/basic/bench_jacobian_fill.rs
# Each value is the average over 300 warm calls (30 for PEGASE 9241).
#
# Assembly (μs): SpMV (Ybus*v) + fill_jacobian_ultimate, KLU excluded.
ASM_US = {
    '39':   {'V0': 29.68, 'V1': 6.19,  'V2': 1.84},
    '118':  {'V0': 101.8, 'V1': 16.71, 'V2': 6.03},
    '9241': {'V0': 13745, 'V1': 4635,  'V2': 641},
}
# KLU per-iteration (μs): refactor + back-substitution, measured directly.
# Version-invariant (same KLU instance), so V0/V1/V2 share the same value.
KLU_US = {
    '39':   3.64 + 0.76,    # refactor 3.64  back-sub 0.76
    '118':  8.73 + 1.82,    # refactor 8.73  back-sub 1.82
    '9241': 2889 + 353,     # refactor 2889  back-sub 353
}

SYSTEMS  = ['39', '118', '9241']
VERSIONS = ['V0', 'V1', 'V2']
SYS_TITLE = {'39': 'IEEE 39', '118': 'IEEE 118', '9241': 'PEGASE 9241'}

# Per-iteration breakdown (ms): assembly + KLU (version-invariant)
iter_ms = {}
for s in SYSTEMS:
    iter_ms[s] = {}
    for v in VERSIONS:
        asm = ASM_US[s][v] / 1000.0    # μs → ms
        lu  = KLU_US[s]    / 1000.0
        iter_ms[s][v] = {'asm': asm, 'lu': lu, 'total': asm + lu}

# ─── Style ─────────────────────────────────────────────────────────────────────
STYLE_ASM = dict(facecolor='0.20', hatch='',     edgecolor='0.30')
STYLE_LU  = dict(facecolor='0.82', hatch='',     edgecolor='0.40')

VER_LABEL_FULL  = {
    'V0': 'V0  (MATPOWER)',
    'V1': "V1  (Sch\xe4fer '18)",
    'V2': 'V2  (Proposed)',
}
VER_LABEL_SHORT = {'V0': 'V0', 'V1': 'V1', 'V2': 'V2'}

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
fig.subplots_adjust(wspace=0.32, bottom=0.32)

BAR_H = 0.50
Y = {'V0': 2, 'V1': 1, 'V2': 0}

for idx, (ax, s) in enumerate(zip(axes, SYSTEMS)):
    x_max = iter_ms[s]['V0']['total']

    # Consistent unit for this panel: determined by V0 (largest bar).
    v0_total = iter_ms[s]['V0']['total']
    if v0_total >= 0.5:      # ≥ 0.5 ms → use ms  (PEGASE)
        unit, scale = 'ms', 1.0
    else:                    # < 0.5 ms → use μs  (IEEE 39, IEEE 118)
        unit, scale = 'μs', 1000.0

    for v in VERSIONS:
        y   = Y[v]
        asm = iter_ms[s][v]['asm']
        lu  = iter_ms[s][v]['lu']

        # Assembly segment (left, dark)
        ax.barh(y, asm, height=BAR_H, left=0,
                facecolor=STYLE_ASM['facecolor'], hatch=STYLE_ASM['hatch'],
                edgecolor=STYLE_ASM['edgecolor'], linewidth=0.0, zorder=3)
        # LU segment (right, light)
        ax.barh(y, lu, height=BAR_H, left=asm,
                facecolor=STYLE_LU['facecolor'], hatch=STYLE_LU['hatch'],
                edgecolor=STYLE_LU['edgecolor'], linewidth=0.6, zorder=3)
        # outer border
        ax.barh(y, asm + lu, height=BAR_H,
                facecolor='none', edgecolor='0.30', linewidth=0.6, zorder=4)

        # Assembly fraction: inside bar when wide enough, else annotate below bar
        pct = asm / (asm + lu) * 100
        if asm > x_max * 0.10:
            ax.text(asm / 2, y, f'{pct:.0f}%',
                    ha='center', va='center', fontsize=5.8,
                    color='white', fontweight='bold', zorder=5)
        # else: bar too short — percentage stated in paper text, omit label here

        # Total time label — consistent unit per panel, 3 significant figures.
        total = asm + lu
        val   = total * scale
        ax.text(total + x_max * 0.015, y, f'{val:.3g} {unit}',
                ha='left', va='center', fontsize=5.5, color='0.25', zorder=5)

    ax.set_title(SYS_TITLE[s], fontsize=8.5, fontweight='bold', pad=3)
    ax.set_yticks([0, 1, 2])
    if idx == 0:
        ax.set_yticklabels(
            [VER_LABEL_FULL['V2'], VER_LABEL_FULL['V1'], VER_LABEL_FULL['V0']],
            fontsize=6.8)
    else:
        ax.set_yticklabels(
            [VER_LABEL_SHORT['V2'], VER_LABEL_SHORT['V1'], VER_LABEL_SHORT['V0']],
            fontsize=7)
    ax.set_xlabel('Per-iteration time (ms)', fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlim(0, x_max * 1.28)
    ax.grid(axis='x', linestyle=':', linewidth=0.4, color='0.65', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ─── Legend ────────────────────────────────────────────────────────────────────
h_asm = mpatches.Patch(
    facecolor=STYLE_ASM['facecolor'], hatch=STYLE_ASM['hatch'], edgecolor='0.40',
    linewidth=0.55, label='Jacobian assembly')
h_lu = mpatches.Patch(
    facecolor=STYLE_LU['facecolor'], hatch=STYLE_LU['hatch'], edgecolor='0.40',
    linewidth=0.55, label='KLU re-factorisation  (constant across V0/V1/V2)')

fig.legend(handles=[h_asm, h_lu], loc='lower center', ncol=2, fontsize=6.8,
           frameon=False, bbox_to_anchor=(0.5, 0.04),
           columnspacing=1.2, handlelength=2.4, handletextpad=0.6)

fig.text(0.015, -0.04,
         'Per-iteration time = assembly + KLU (refactor + back-sub).  '
         'KLU cost directly instrumented (version-invariant); assembly from compare_assembly().',
         fontsize=5.8, color='0.45')

# ─── Save ──────────────────────────────────────────────────────────────────────
out = Path('paper/time_breakdown')
fig.savefig(str(out) + '.pdf', bbox_inches='tight', dpi=300)
fig.savefig(str(out) + '.png', bbox_inches='tight', dpi=300)
print(f'Saved {out}.pdf and {out}.png')
print()
print('Assembly fraction per version (directly instrumented):')
for s in SYSTEMS:
    fracs = {v: iter_ms[s][v]["asm"] / iter_ms[s][v]["total"] * 100 for v in VERSIONS}
    print(f'  {SYS_TITLE[s]}: V0={fracs["V0"]:.0f}%  V1={fracs["V1"]:.0f}%  V2={fracs["V2"]:.0f}%')
