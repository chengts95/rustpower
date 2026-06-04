import numpy as np
import pandapower as pp
import pandapower.networks as nw
from rustpower import NewtonSolver
import time

def solve_ppci_verified_demo(net_name="case118"):
    print(f"--- Testing {net_name} with Verified Double-Transpose Logic ---")
    if net_name == "case118":
        net = nw.case118()
    elif net_name == "case9241":
        net = nw.case9241pegase()
    else:
        net = nw.case14()
        
    pp.runpp(net) # Original pandapower result
    v_pp = net["_ppc"]["internal"]["V"]
    
    ppc = net["_ppc"]
    internal = ppc["internal"]
    Ybus = internal["Ybus"] # CSC matrix
    Sbus = internal["Sbus"]
    
    # Construct a proper flat start:
    # - PQ buses: 1.0 + 0j
    # - PV buses: magnitude from setpoint, angle 0
    # - Slack buses: exact complex voltage from setpoint
    # Note: internal["V"] here is the converged result from runpp, so its magnitudes 
    # for PV and complex values for Slack match the problem specification.
    Vinit_flat = np.ones(Ybus.shape[0], dtype=np.complex128)
    ref = internal["ref"]
    pv = internal["pv"]
    pq = internal["pq"]
    
    Vinit_flat[pv] = np.abs(internal["V"][pv]) # Keep magnitude, reset angle
    Vinit_flat[ref] = internal["V"][ref]       # Keep exact slack voltage
    
    p_vec = np.concatenate([pq, pv, ref]).astype(np.int64)
    p_inv = np.zeros(len(p_vec), dtype=np.int64)
    p_inv[p_vec] = np.arange(len(p_vec), dtype=np.int64)

    # 1. Create Solver Context
    solver = NewtonSolver()
    
    # 2. setup_context (Double-Transpose Trick)
    t0 = time.perf_counter()
    solver.setup_context(
        y_indptr=Ybus.indptr,
        y_indices=Ybus.indices,
        y_data=Ybus.data,
        s_bus=Sbus,
        v_init=Vinit_flat, # Flat start!
        p_vec=p_vec.tolist(),
        p_inv=p_inv.tolist(),
        npv=len(pv),
        npq=len(pq)
    )
    t_setup = (time.perf_counter() - t0) * 1000

    # 3. Solve
    t1 = time.perf_counter()
    converged = solver.solve()
    t_solve = (time.perf_counter() - t1) * 1000

    # 4. Get Results
    v_rust = solver.get_voltage()
    
    # Extract metrics via Python logic since we removed the metrics dict from solve()
    # Wait, solve() just returns bool now. We need an iterations getter.
    # Let's add it or just check if it converged.
    
    # 5. Accuracy Check
    diff = np.linalg.norm(v_rust - v_pp)
    
    print(f"Setup Context: {t_setup:.2f} ms")
    print(f"Solve:         {t_solve:.2f} ms")
    print(f"Converged:     {converged}")
    print(f"Voltage Diff:  {diff:.2e}")
    
    if diff < 1e-6 and converged:
        print("✅ VERIFICATION SUCCESSFUL")
    else:
        print("❌ VERIFICATION FAILED - Result discrepancy or no convergence")

if __name__ == "__main__":
    solve_ppci_verified_demo("case118")
    print("\n" + "="*40 + "\n")
    solve_ppci_verified_demo("case9241")
