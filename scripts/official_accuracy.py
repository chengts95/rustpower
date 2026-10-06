"""
Official Accuracy Benchmark: Pandapower vs RustPower
"""

import warnings
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import rustpower

warnings.filterwarnings('ignore')

TOL = 1e-6

def check_accuracy(net_name, net):
    print(f"\n=========================================================")
    print(f" ACCURACY CHECK: {net_name} ({len(net.bus)} buses)")
    print(f"=========================================================")
    
    # Ground Truth: Pandapower
    pp.runpp(net, init="flat", recycle=False, tolerance_mva=TOL)
    pp_vm = net.res_bus.vm_pu.values
    pp_va = net.res_bus.va_degree.values
    
    # RustPower
    rp_model = rustpower.PowerGrid.from_pandapower(net)
    num_buses = len(net.bus)
    V_init_flat = np.ones(num_buses, dtype=np.complex128)
    rp_model.solve(V_init_flat)
    
    rp_vm = rp_model.res_bus.vm_pu.values
    rp_va = rp_model.res_bus.va_degree.values
    
    # Compare
    vm_diff = np.abs(pp_vm - rp_vm)
    va_diff = np.abs(pp_va - rp_va)
    
    print(f"VM Error (pu):     Max = {np.max(vm_diff):.2e}, Mean = {np.mean(vm_diff):.2e}")
    print(f"VA Error (degree): Max = {np.max(va_diff):.2e}, Mean = {np.mean(va_diff):.2e}")
    
    if np.max(vm_diff) < 1e-5 and np.max(va_diff) < 1e-4:
        print("-> [PASS] Accuracy is virtually identical to Pandapower.")
    else:
        print("-> [FAIL] Significant divergence found.")

if __name__ == "__main__":
    cases = {
        "Case 39": pn.case39,
        "Case 118": pn.case118,
        "Case 1354 PEGASE": pn.case1354pegase,
        "Case 9241 PEGASE": pn.case9241pegase
    }
    
    for name, case_fn in cases.items():
        net = case_fn()
        check_accuracy(name, net)

