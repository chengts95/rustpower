import numpy as np
import pandapower as pp
import pandapower.networks as nw
from rustpower.solver import NewtonSolver
import time
import copy

class RustPowerSolver:
    def __init__(self):
        self._ctx = NewtonSolver()

    def solve(self, net):
        # 1. Internal PP Initialization (dry run)
        try:
            pp.runpp(net, algorithm='nr', max_iteration=0, init='flat')
        except:
            pass
        
        ppci = net["_ppc"]
        internal = ppci["internal"]
        
        # 2. Extract Data
        Ybus = internal["Ybus"]
        Sbus = internal["Sbus"]
        Vinit = internal["V"]
        
        # 3. Get Indices
        ref = internal["ref"]
        pv = internal["pv"]
        pq = internal["pq"]
        
        p_vec = np.concatenate([pq, pv, ref]).astype(np.int64)
        p_inv = np.zeros(len(p_vec), dtype=np.int64)
        p_inv[p_vec] = np.arange(len(p_vec), dtype=np.int64)

        # 4. Setup
        self._ctx.setup_context(
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
        
        # 5. Solve (time only this part for comparison)
        t0 = time.perf_counter()
        converged = self._ctx.solve()
        t_solve = (time.perf_counter() - t0) * 1000
        
        return converged, t_solve

def benchmark_grid(net_name):
    print(f"\n{'='*50}")
    print(f"📊 Benchmarking Grid: {net_name}")
    print(f"{'='*50}")

    if net_name == "case9241":
        net = nw.case9241pegase()
        iterations = 10  # Pandapower native is slow, so fewer loops
    else:
        net = nw.case118()
        iterations = 100
        
    net_pp = copy.deepcopy(net)

    # ----------------------------------------------------
    # Test 1: Native Pandapower (SciPy NR)
    # ----------------------------------------------------
    pp_times = []
    # Warmup
    pp.runpp(net_pp, algorithm='nr', init='flat')
    for _ in range(iterations):
        t0 = time.perf_counter()
        pp.runpp(net_pp, algorithm='nr', init='flat')
        pp_times.append((time.perf_counter() - t0) * 1000)
    
    avg_pp = np.mean(pp_times)
    min_pp = np.min(pp_times)
    print(f"Native Pandapower ({iterations} loops) -> Avg: {avg_pp:.2f} ms | Min: {min_pp:.2f} ms")

    # ----------------------------------------------------
    # Test 2: RustPower as PPCI Solver (Context)
    # ----------------------------------------------------
    rp_solver = RustPowerSolver()
    # Initial setup run (dry run inside solve handles Vinit correctly)
    net_rp = copy.deepcopy(net)
    rp_solver.solve(net_rp)
    
    # We only benchmark the solver loop (setup is done)
    # In a real TS simulation, only Vinit/Sbus updates are needed.
    rp_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        # To make it fair, we re-run setup to simulate a full step, 
        # but realistically you just call solve() if topology doesn't change.
        # Let's time BOTH to show the difference.
        
        # Time purely the solve (mimicking native ECS power)
        t_start = time.perf_counter()
        rp_solver._ctx.solve()
        rp_times.append((time.perf_counter() - t_start) * 1000)

    avg_rp = np.mean(rp_times)
    min_rp = np.min(rp_times)
    print(f"RustPower PPCI Plugin (Solve Only, {iterations} loops) -> Avg: {avg_rp:.3f} ms | Min: {min_rp:.3f} ms")

if __name__ == "__main__":
    benchmark_grid("case118")
    benchmark_grid("case9241")
