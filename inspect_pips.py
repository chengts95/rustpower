"""Inspect PYPOWER PIPS source and run case39 with detailed per-iter output."""
import pandapower as pp
import pandapower.networks as pn
import numpy as np, inspect, importlib

# ── Find bundled PIPS ────────────────────────────────────────────────────────
try:
    from pandapower.pypower import pips as pips_mod
    src = inspect.getsource(pips_mod.pips)
    print("=== PIPS key parameters ===")
    for i, line in enumerate(src.split('\n')):
        stripped = line.strip()
        if any(k in stripped for k in [
            'sigma', 'SIGMA', 'step_control', 'feastol', 'gradtol',
            'comptol', 'costtol', 'max_it', 'alpha', 'gamma',
            'def pips', 'opt.get', "'max_it'", "'feastol'",
        ]):
            print(f"  {i:4d}: {line.rstrip()}")
except Exception as e:
    print(f"pips import error: {e}")

# ── Run case39 OPF and capture ext_grid ─────────────────────────────────────
net = pn.case39()
pp.runopp(net, verbose=False)

print(f"\n=== case39 OPF result: {net.res_cost:.4f} EUR ===")
print("\ngen dispatch:")
cols_gen = [c for c in net.res_gen.columns if c in ['p_mw','q_mvar']]
print(net.res_gen[cols_gen].to_string())
print("\next_grid dispatch:")
cols_eg = [c for c in net.res_ext_grid.columns if c in ['p_mw','q_mvar']]
print(net.res_ext_grid[cols_eg].to_string())
print("\nBus vm_pu range:", net.res_bus['vm_pu'].min(), "to", net.res_bus['vm_pu'].max())

# ── Check x0 (initial point) in PYPOWER ─────────────────────────────────────
print("\n=== PYPOWER initial point (from _ppc) ===")
ppc = net._ppc
if ppc:
    gen = ppc['gen']
    bus = ppc['bus']
    nb = len(bus)
    ng = len(gen)
    print(f"nb={nb}, ng={ng}")
    print("bus Va(deg) first 5:", bus[:5, 8].tolist())
    print("bus Vm(pu)  first 5:", bus[:5, 7].tolist())
    print("gen Pg(pu)  first 5:", (gen[:5, 1]/100).tolist())
    print("gen Qg(pu)  first 5:", (gen[:5, 2]/100).tolist())
    print("gen Pgmin(pu):", (gen[:, 9]/100).tolist())
    print("gen Pgmax(pu):", (gen[:, 8]/100).tolist())
