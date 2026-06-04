import numpy as np
import pandapower as pp
import pandapower.networks as nw
from rustpower import NewtonSolver
import time

def solve_ppci_optimized_demo(net_name="case118"):
    print(f"--- Testing {net_name} with Optimized In-Rust Permutation ---")
    if net_name == "case118":
        net = nw.case118()
    elif net_name == "case9241":
        net = nw.case9241pegase()
    else:
        net = nw.case14()
        
    # 1. Get raw PPCI
    pp.runpp(net)
    ppc = net["_ppc"]
    internal = ppc["internal"]
    
    Ybus = internal["Ybus"] # CSC matrix
    Sbus = internal["Sbus"]
    Vinit = internal["V"]
    
    # 2. Get Permutation Vectors
    # Rustpower expects: [PQ | PV | Slack]
    ref = internal["ref"]
    pv = internal["pv"]
    pq = internal["pq"]
    p_vec = np.concatenate([pq, pv, ref]).astype(np.int32)
    p_inv = np.zeros(len(p_vec), dtype=np.int32)
    p_inv[p_vec] = np.arange(len(p_vec), dtype=np.int32)

    # 3. Create Solver
    solver = NewtonSolver()
    
    # 4. Solve with zero-copy-like mapping and in-Rust permutation
    # Note: We send raw (non-permuted) matrices to Rust
    t0 = time.perf_counter()
    res = solver.solve_ppci_optimized(
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
    t_total_py = (time.perf_counter() - t0) * 1000

    if res["converged"]:
        m = res["metrics"]
        print(f"Results: Converged, Iterations={res['iterations']}")
        print(f"Breakdown (Microseconds):")
        print(f"  - Rust Data Map (FFI): {m['map_us']:.1f} us")
        print(f"  - Rust Permute (P*Y*PT): {m['perm_us']:.1f} us")
        print(f"  - Rust Core Calc:     {m['calc_us']:.1f} us")
        print(f"  - Rust Restore order: {m['restore_us']:.1f} us")
        print(f"  - Total (Rust Side):  {m['total_us']:.1f} us")
        print(f"  - Total (Python Wall): {t_total_py:.2f} ms")
        
        overhead = (m['map_us'] + m['perm_us'] + m['restore_us'])
        calc = m['calc_us']
        print(f"Overhead/Calc Ratio: {overhead/calc:.2f}x")
    else:
        print(f"Failed: {res.get('error')}")

if __name__ == "__main__":
    solve_ppci_optimized_demo("case118")
    print("\n" + "="*40 + "\n")
    solve_ppci_optimized_demo("case9241")
