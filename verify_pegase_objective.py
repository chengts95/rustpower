import pandapower as pp
import pandapower.networks as pn
import numpy as np

# Try to run with a more robust solver or higher max_it
net = pn.case9241pegase()
try:
    pp.runopp(net, verbose=True, max_it=300)
    print(f"PEGASE 9241 Objective: {net.res_cost:.4f}")
except Exception as e:
    print(f"PEGASE 9241 OPF failed to converge in Python: {e}")

# Check initial load
total_load_mw = net.load.p_mw.sum()
print(f"Total Load: {total_load_mw:.2f} MW")
