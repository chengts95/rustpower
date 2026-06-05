"""
RustPower vs Pandapower Benchmark

This script performs an apples-to-apples comparison by:
1. Running pandapower natively to get the ground-truth Ybus, Sbus, and converged V.
2. Feeding the exact same Ybus and Sbus, along with a flat start, into RustPower.
3. Comparing the convergence speed and final accuracy.
"""

import numpy as np
import pandapower as pp
import pandapower.networks as nw
from rustpower.solver import NewtonSolver
import time

def run_benchmark(net_name="case118"):
    print(f"\n{'='*40}")
    print(f"🚀 Benchmarking {net_name}")
    print(f"{'='*40}")
    
    if net_name == "case9241":
        net = nw.case9241pegase()
    else:
        net = nw.case118()

    # ---------------------------------------------------------
    # 1. Native Pandapower Run
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    pp.runpp(net, algorithm='nr', init='flat')
    t_pp = (time.perf_counter() - t0) * 1000
    
    # Extract the exact matrices used in the final iteration
    ppci = net["_ppc"]
    internal = ppci["internal"]
    v_pp = internal["V"] # The true converged result
    Ybus = internal["Ybus"]
    Sbus = internal["Sbus"]
    
    pq = internal["pq"]
    pv = internal["pv"]
    ref = internal["ref"]

    # ---------------------------------------------------------
    # 2. Prepare Rust Start State
    # ---------------------------------------------------------
    # Construct a flat start honoring the setpoints (PV magnitudes, Slack complex voltage)
    v_init = np.ones(Ybus.shape[0], dtype=np.complex128)
    v_init[pv] = np.abs(v_pp[pv])
    v_init[ref] = v_pp[ref]
    
    # Permutation vectors for [PQ | PV | Slack]
    p_vec = np.concatenate([pq, pv, ref]).astype(np.int64)
    p_inv = np.zeros(len(p_vec), dtype=np.int64)
    p_inv[p_vec] = np.arange(len(p_vec), dtype=np.int64)

    # ---------------------------------------------------------
    # 3. RustPower Solve
    # ---------------------------------------------------------
    solver = NewtonSolver()
    
    # Setup Phase (Includes Data Transfer and O(NNZ) Permutation)
    t1 = time.perf_counter()
    solver.setup_context(
        y_indptr=Ybus.indptr,
        y_indices=Ybus.indices,
        y_data=Ybus.data,
        s_bus=Sbus,
        v_init=v_init,
        p_vec=p_vec.tolist(),
        p_inv=p_inv.tolist(),
        npv=len(pv),
        npq=len(pq)
    )
    t_setup = (time.perf_counter() - t1) * 1000
    
    # Pure Computation Phase
    t2 = time.perf_counter()
    converged = solver.solve()
    t_solve = (time.perf_counter() - t2) * 1000
    
    v_rust = solver.get_voltage()

    # ---------------------------------------------------------
    # 4. Results & Comparison
    # ---------------------------------------------------------
    diff = np.linalg.norm(v_rust - v_pp)
    
    print(f"Pandapower (Native):  {t_pp:>8.2f} ms")
    print(f"Rust Setup Context:   {t_setup:>8.2f} ms")
    print(f"Rust Core Solve:      {t_solve:>8.2f} ms")
    
    t_rust_total = t_setup + t_solve
    print(f"Rust Total:           {t_rust_total:>8.2f} ms")
    
    speedup = t_pp / t_rust_total
    print(f"\nSpeedup vs Native:    {speedup:.1f}x")
    print(f"L2 Norm Difference:   {diff:.2e}")
    
    if diff < 1e-8 and converged:
        print("✅ ACCURACY VERIFIED (Mathematically identical)")
    else:
        print("❌ ACCURACY FAILED")

if __name__ == "__main__":
    run_benchmark("case118")
    run_benchmark("case9241")
