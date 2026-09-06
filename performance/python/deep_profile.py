"""
Deep Profiling Benchmark: RustPower vs Pandapower

This script breaks down the exact execution time of pandapower's `runpp` to 
reveal where the time is actually spent: Data Prep vs. Pure Solving vs. Post-processing.
"""

import numpy as np
import pandapower as pp
import pandapower.networks as nw
from rustpower.solver import NewtonSolver
import time
import copy

def deep_profile_benchmark(net_name="case118"):
    print(f"\n{'='*50}")
    print(f"🔍 Deep Profiling Benchmark: {net_name}")
    print(f"{'='*50}")

    if net_name == "case9241":
        net = nw.case9241pegase()
    else:
        net = nw.case118()

    # ---------------------------------------------------------
    # 1. Profile Native Pandapower
    # ---------------------------------------------------------
    # We use pandapower's internal timing mechanisms if possible, 
    # but we can also manually measure the stages.
    
    from pandapower.pd2ppc import _pd2ppc, _ppc2ppci
    from pandapower.results import _copy_results_ppci_to_ppc, _extract_results, init_results

    # Stage A: Warm-up to eliminate JIT/Cold-start overhead
    # Run a full successful solve to ensure Numba is fully compiled
    net_warmup = copy.deepcopy(net)
    try:
        pp.runpp(net_warmup, algorithm='nr', init='flat')
    except:
        pass

    # The safest way to profile this without internal hacks is running runpp 
    # and tracking the total time vs the solver time recorded in net._ppc['iterations'] 
    
    # Actually, pandapower records the solver time internally!
    t0 = time.perf_counter()
    pp.runpp(net, algorithm='nr', init='flat')
    t_total_pp = (time.perf_counter() - t0) * 1000
    
    ppc = net["_ppc"]
    internal = ppc["internal"]
    
    # Pandapower stores the pure solver time in internal['et']
    t_pure_solve_pp = ppc.get("et", 0.0) * 1000 # et is in seconds
    
    # If 'et' is not accurate, we calculate overhead:
    t_overhead_pp = t_total_pp - t_pure_solve_pp

    print("--- Native Pandapower Breakdown ---")
    print(f"Total runpp Time:     {t_total_pp:>8.2f} ms")
    print(f"  ├─ Pure SciPy NR:   {t_pure_solve_pp:>8.2f} ms")
    print(f"  └─ Data Prep & I/O: {t_overhead_pp:>8.2f} ms ({(t_overhead_pp/t_total_pp)*100:.1f}%)")

    # ---------------------------------------------------------
    # 2. RustPower Solve (Context already prepared)
    # ---------------------------------------------------------
    Ybus = internal["Ybus"]
    Sbus = internal["Sbus"]
    Vinit = np.ones(Ybus.shape[0], dtype=np.complex128)
    Vinit[internal["pv"]] = np.abs(internal["V"][internal["pv"]])
    Vinit[internal["ref"]] = internal["V"][internal["ref"]]
    
    pq = internal["pq"]
    pv = internal["pv"]
    ref = internal["ref"]
    p_vec = np.concatenate([pq, pv, ref]).astype(np.int64)
    p_inv = np.zeros(len(p_vec), dtype=np.int64)
    p_inv[p_vec] = np.arange(len(p_vec), dtype=np.int64)

    solver = NewtonSolver()
    solver.setup_context(
        y_indptr=Ybus.indptr, y_indices=Ybus.indices, y_data=Ybus.data,
        s_bus=Sbus, v_init=Vinit, p_vec=p_vec.tolist(), p_inv=p_inv.tolist(),
        npv=len(pv), npq=len(pq)
    )
    
    # Warmup rust solver
    solver.solve()
    
    # Actual timed run
    # Note: since the solver updates V internally, we should reset it to get a fair iteration count,
    # but since we are just measuring raw computation speed of an iteration, we can just solve again,
    # though it might converge in 0 iterations. To be completely fair and measure the NR loop, 
    # we MUST reset the context state or re-feed the flat Vinit.
    
    # Re-setup to ensure a flat start for the timed run
    solver.setup_context(
        y_indptr=Ybus.indptr, y_indices=Ybus.indices, y_data=Ybus.data,
        s_bus=Sbus, v_init=Vinit, p_vec=p_vec.tolist(), p_inv=p_inv.tolist(),
        npv=len(pv), npq=len(pq)
    )
    
    t1 = time.perf_counter()
    solver.solve()
    t_pure_solve_rust = (time.perf_counter() - t1) * 1000
    
    print("\n--- RustPower Breakdown ---")
    print(f"Pure Rust Core Solve: {t_pure_solve_rust:>8.2f} ms")
    
    print("\n--- Ultimate Comparison ---")
    print(f"Pure Algorithm Speedup (SciPy NR vs Rust KLU): {t_pure_solve_pp / t_pure_solve_rust:.1f}x")

if __name__ == "__main__":
    deep_profile_benchmark("case118")
    deep_profile_benchmark("case9241")
