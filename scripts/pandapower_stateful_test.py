import numpy as np
import pandapower as pp
import pandapower.networks as nw
from rustpower import NewtonSolver
import time

def solve_ppci_stateful_demo(net_name="case118"):
    print(f"--- Testing {net_name} with Stateful Solver Context ---")
    if net_name == "case118":
        net = nw.case118()
    elif net_name == "case9241":
        net = nw.case9241pegase()
    else:
        net = nw.case14()
        
    pp.runpp(net)
    ppc = net["_ppc"]
    internal = ppc["internal"]
    
    Ybus = internal["Ybus"]
    Sbus = internal["Sbus"]
    Vinit = internal["V"]
    
    ref = internal["ref"]
    pv = internal["pv"]
    pq = internal["pq"]
    p_vec = np.concatenate([pq, pv, ref]).astype(np.int64) # Rust handles conversion
    p_inv = np.zeros(len(p_vec), dtype=np.int64)
    p_inv[p_vec] = np.arange(len(p_vec), dtype=np.int64)

    # 1. Create Solver Context
    solver = NewtonSolver()
    
    # 2. setup_context (Pour matrices into Rust)
    t0 = time.perf_counter()
    solver.setup_context(
        y_indptr=Ybus.indptr,
        y_indices=Ybus.indices,
        y_data=Ybus.data,
        s_bus=Sbus,
        v_init=Vinit,
        p_vec=p_vec.tolist(),
        p_inv=p_inv.tolist(),
        npv=len(pv),
        npq=len(pq)
    )
    t_setup = (time.perf_counter() - t0) * 1000
    print(f"Setup Context (Once): {t_setup:.2f} ms")

    # 3. Solve (Pure State Transition)
    t1 = time.perf_counter()
    solver.solve()
    t_solve = (time.perf_counter() - t1) * 1000
    print(f"Solve (Stateful):     {t_solve:.2f} ms")

    # 4. Get Results (Lightweight Restore)
    t2 = time.perf_counter()
    v_res = solver.get_voltage()
    t_get = (time.perf_counter() - t2) * 1000
    print(f"Get & Restore Result: {t_get:.2f} ms")
    
    print(f"Total Workflow Time:  {t_setup + t_solve + t_get:.2f} ms")

if __name__ == "__main__":
    solve_ppci_stateful_demo("case118")
    print("\n" + "="*40 + "\n")
    solve_ppci_stateful_demo("case9241")
