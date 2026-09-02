import time
import numpy as np
import pandapower as pp
import pandapower.networks as nw

def test_newton_solver_api():
    print("Testing NewtonSolver API with PEGASE 9241...")
    
    # 1. Load the network
    net = nw.case9241pegase()
    
    # 2. Warm up and solve natively to get PPC internal matrices
    # Use 1 iteration just to initialize matrices and return immediately
    try:
        pp.runpp(net, algorithm='nr', max_iteration=20, init='flat', tolerance_mva=1e-6)
    except:
        pass
        
    ppci = net["_ppc"]
    internal = ppci["internal"]
    v_pp = internal["V"]
    Ybus = internal["Ybus"] # Unpermuted CSR
    Sbus = internal["Sbus"] # Unpermuted Sbus
    
    pq = internal["pq"]
    pv = internal["pv"]
    ref = internal["ref"]
    
    # Permutation vectors: PQ buses first, then PV, then Slack/Ref
    p_vec = np.concatenate([pq, pv, ref]).astype(np.int64)
    p_inv = np.zeros(len(p_vec), dtype=np.int64)
    p_inv[p_vec] = np.arange(len(p_vec), dtype=np.int64)
    
    # Flat start: set slack voltage setpoint and PV magnitude setpoints
    v_init = np.ones(Ybus.shape[0], dtype=np.complex128)
    v_init[pv] = np.abs(v_pp[pv])
    v_init[ref] = v_pp[ref]
    
    # 3. Import rustpower and initialize solver
    import sys
    try:
        rp_solver = sys.modules["rustpower.solver"].NewtonSolver()
    except KeyError:
        import rustpower
        rp_solver = sys.modules.get("rustpower.solver").NewtonSolver() if "rustpower.solver" in sys.modules else getattr(rustpower, "solver").NewtonSolver()
        
    # Warmup / Setup Context
    start = time.perf_counter()
    rp_solver.setup_context(
        Ybus.indptr,
        Ybus.indices,
        Ybus.data,
        Sbus,
        v_init.copy(),
        p_vec.tolist(),
        p_inv.tolist(),
        len(pv),
        len(pq)
    )
    setup_time = time.perf_counter() - start
    print(f"RustPower setup_context took {setup_time * 1000:.3f} ms")
    
    # Solve (Cold Start - builds j_pattern and symbolics)
    start = time.perf_counter()
    converged = rp_solver.solve()
    print(f"Iterations taken: {rp_solver.get_iterations()}")
    cold_time = time.perf_counter() - start
    print(f"RustPower cold solve took {cold_time * 1000:.3f} ms. Converged: {converged}")
    
    # Fetch result
    v_rp = rp_solver.get_voltage()
    
    # Compare
    diff = np.linalg.norm(v_rp - v_pp)
    print(f"L2 Norm Difference vs Pandapower: {diff:.2e}")
    if diff < 1e-6:
        print("[PASS] Results match Pandapower!")
    else:
        print("[FAIL] Results diverge!")

    # Hot Loop Benchmark
    rp_solver.enable_cache(True)
    rp_solver.solve() # prime cache
    times_rp = []
    for _ in range(100):
        start = time.perf_counter()
        rp_solver.solve()
        times_rp.append(time.perf_counter() - start)
        
    print(f"\nRustPower Hot Loop (100 runs): {np.mean(times_rp) * 1000:.3f} ms / solve")

if __name__ == "__main__":
    test_newton_solver_api()
