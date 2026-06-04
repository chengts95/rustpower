import numpy as np
import pandapower as pp
import pandapower.networks as nw
from rustpower import NewtonSolver
import time

def extract_ppci_data(net):
    """
    Extracts and prepares data from pandapower's PPCI structure.
    """
    # 1. Run pp to ensure internal ppci structure is built and populated
    pp.runpp(net)
    
    ppc = net["_ppc"]
    internal = ppc["internal"]
    
    # 2. Key components in PPCI
    Ybus = internal["Ybus"] # CSC matrix
    Sbus = internal["Sbus"] # np.ndarray
    Vinit = internal["V"]   # np.ndarray
    
    # 3. Bus type indices (contiguous 0..N-1 in internal PPCI)
    ref = internal["ref"]
    pv = internal["pv"]
    pq = internal["pq"]
    
    return Ybus, Sbus, Vinit, ref, pv, pq

def get_permutation_to_rustpower(ref, pv, pq, n_bus):
    """
    Creates a permutation vector that reorders buses to:
    [PQ buses | PV buses | Slack buses]
    as expected by rustpower core.
    """
    # Order: PQ first, then PV, then Slack
    p = np.concatenate([pq, pv, ref]).astype(np.int32)
    
    # Inverse permutation to map results back
    p_inv = np.zeros(n_bus, dtype=np.int32)
    p_inv[p] = np.arange(n_bus, dtype=np.int32)
    
    return p, p_inv

def solve_ppci_with_rust(net):
    print(f"Integrating with pandapower network: {net.name}")
    
    # 1. Extract raw PPCI from internal keys
    Ybus, Sbus, Vinit, ref, pv, pq = extract_ppci_data(net)
    n_bus = Ybus.shape[0]
    
    # 2. Handle Permutation (Mathematical optimization)
    # Rustpower core expects [PQ | PV | Slack]
    p, p_inv = get_permutation_to_rustpower(ref, pv, pq, n_bus)
    
    # Permute Ybus: Y_rust = P * Ybus * P^T
    # In SciPy, this is efficient if we do it right
    t_perm_start = time.perf_counter()
    Y_rust = Ybus[p, :][:, p] 
    
    # CRITICAL: Rust's CscMatrix requires sorted row indices.
    # SciPy slicing might leave indices unsorted.
    Y_rust.sort_indices()
    
    S_rust = Sbus[p]
    V_rust = Vinit[p]
    t_perm = (time.perf_counter() - t_perm_start) * 1000
    
    # 3. Call Rust Solver
    solver = NewtonSolver()
    
    # Using solve_ppci_profiled to see the breakdown
    res = solver.solve_ppci_profiled(
        y_indptr=Y_rust.indptr.astype(np.int32),
        y_indices=Y_rust.indices.astype(np.int32),
        y_data=Y_rust.data.astype(np.complex128),
        s_bus=S_rust.astype(np.complex128),
        v_init=V_rust.astype(np.complex128),
        pv_bus=np.arange(len(pq), len(pq) + len(pv)).tolist(), # Relative to p
        pq_bus=np.arange(len(pq)).tolist()
    )
    
    if res["converged"]:
        # 4. Map results back to PPCI order
        V_res_rust = res["v"]
        V_res_ppci = V_res_rust[p_inv]
        
        # Write back to pandapower's internal structure
        net["_ppc"]["internal"]["V"] = V_res_ppci
        
        m = res["metrics"]
        print(f"--- Performance Breakdown (IEEE 118) ---")
        print(f"Python Permutation: {t_perm:.2f} ms")
        print(f"Rust Data Map:      {m['map_us']/1000:.2f} ms")
        print(f"Rust Core Calc:     {m['calc_us']/1000:.2f} ms")
        print(f"Total (Full Trip):  {(t_perm + m['total_us']/1000):.2f} ms")
        
        overhead = t_perm + m['map_us']/1000
        calc = m['calc_us']/1000
        print(f"Overhead/Calc Ratio: {overhead/calc:.2f}x")
        return True
    else:
        print("Rustpower failed to converge.")
        return False

if __name__ == "__main__":
    # Test with IEEE 118
    net118 = nw.case118()
    solve_ppci_with_rust(net118)
    
    print("\n" + "="*40 + "\n")
    
    # Test with a much larger case
    try:
        net9241 = nw.case9241pegase()
        solve_ppci_with_rust(net9241)
    except Exception as e:
        print(f"Skipping large case: {e}")
