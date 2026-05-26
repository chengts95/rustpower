"""Compare pandapower OPF vs our PIPS solver on case39."""
import pandapower as pp
import pandapower.networks as pn
import numpy as np

net = pn.case39()
print("=== Case39 poly_cost ===")
print(net.poly_cost[['element','et','cp0_eur','cp1_eur_per_mw','cp2_eur_per_mw2']].to_string())

# ── 1. Pandapower OPF ───────────────────────────────────────────────────────
try:
    pp.runopp(net, verbose=True)
    print(f"\n=== Pandapower OPF result ===")
    print(f"Objective: {net.res_cost:.4f} EUR")
    print(f"\ngen dispatch (MW):")
    print(net.res_gen[['p_mw','q_mvar','vm_pu']].to_string())
    print(f"\next_grid dispatch (MW):")
    print(net.res_ext_grid[['p_mw','q_mvar','vm_pu']].to_string())
    print(f"\nBus voltages (first 5):")
    print(net.res_bus[['vm_pu','va_degree']].head().to_string())
    total_gen_mw = net.res_gen['p_mw'].sum() + net.res_ext_grid['p_mw'].sum()
    total_load_mw = net.load['p_mw'].sum()
    print(f"\nTotal gen: {total_gen_mw:.2f} MW,  Total load: {total_load_mw:.2f} MW")
except Exception as e:
    print(f"OPF failed: {e}")
    import traceback; traceback.print_exc()

# ── 2. PYPOWER solver parameters ────────────────────────────────────────────
print("\n=== PYPOWER default OPF options ===")
from pypower import ppoption
opt = ppoption.ppoption()
for k in ['OPF_ALG','OPF_VIOLATION','OPF_FLOW_LIM','OPF_IGNORE_ANG_LIM',
          'PIPS_MAX_IT','PIPS_STEP_CONTROL','PIPS_FEASTOL','PIPS_GRADTOL',
          'PIPS_COMPTOL','PIPS_COSTTOL']:
    if k in opt:
        print(f"  {k} = {opt[k]}")

# ── 3. PIPS parameters (from pypower.pips) ──────────────────────────────────
print("\n=== PIPS default parameters ===")
try:
    import inspect
    from pypower import pips as pips_mod
    src = inspect.getsource(pips_mod.pips)
    # Find SIGMA/sigma usage
    for line in src.split('\n'):
        if any(kw in line for kw in ['sigma','SIGMA','step_control','feastol',
                                      'gradtol','comptol','costtol','max_it',
                                      'alpha_min','alpha']):
            print(' ', line.rstrip())
except Exception as e:
    print(f"  Could not inspect pips: {e}")

# ── 4. Initial point ────────────────────────────────────────────────────────
print("\n=== Initial point (flat start in pypower?) ===")
try:
    from pypower import opf_execute, ext2int, makeYbus, opf_setup
    import pypower.api as pa
    ppc = net._ppc  # internal pypower case
    if ppc is not None:
        print("gen Pg initial (p.u.):", ppc['gen'][:5, 1] / 100)
        print("bus Va initial (deg):", ppc['bus'][:5, 8])
        print("bus Vm initial (p.u.):", ppc['bus'][:5, 7])
except Exception as e:
    print(f"  {e}")
