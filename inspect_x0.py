"""Check what x0 PYPOWER passes to PIPS before solving."""
import pandapower as pp
import pandapower.networks as pn
import numpy as np

net = pn.case39()

# Run power flow first (what pandapower does before OPF)
pp.runpp(net, verbose=False)
print("=== Power flow solution (used as OPF x0) ===")
print("Bus vm_pu first 5:", net.res_bus['vm_pu'][:5].tolist())
print("Bus va_degree first 5:", net.res_bus['va_degree'][:5].tolist())
print("Gen p_mw:", net.res_gen['p_mw'].tolist())
print("Gen q_mvar:", net.res_gen['q_mvar'].tolist())
print("Ext_grid p_mw:", net.res_ext_grid['p_mw'].tolist())
print("Ext_grid q_mvar:", net.res_ext_grid['q_mvar'].tolist())

# Inspect opf_setup
try:
    from pandapower.pypower import opf_setup
    import inspect
    src = inspect.getsource(opf_setup.opf_setup)
    print("\n=== opf_setup x0 lines ===")
    for i, line in enumerate(src.split('\n')):
        if any(k in line for k in ['x0','Va0','Vm0','Pg0','initial','bus[:,VA]','gen[:,PG]']):
            print(f"  {i:4d}: {line.rstrip()}")
except Exception as e:
    print(f"opf_setup: {e}")

# Check ppc state after runpp
net2 = pn.case39()
pp.runpp(net2, verbose=False)
ppc2 = net2._ppc
bus = ppc2['bus']
gen = ppc2['gen']
print(f"\n=== ppc after runpp (before runopp): this is the OPF x0 ===")
print("Bus Va(deg) [0..4]:", bus[:5, 8].tolist())
print("Bus Vm(pu)  [0..4]:", bus[:5, 7].tolist())
print("Gen Pg(pu):", (gen[:, 1]/100).tolist())
print("Gen Qg(pu):", (gen[:, 2]/100).tolist())
print("Gen Pgmin(pu):", (gen[:, 9]/100).tolist())
print("Gen Pgmax(pu):", (gen[:, 8]/100).tolist())
