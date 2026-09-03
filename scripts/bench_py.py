"""
Official Performance Benchmark: Pandapower vs LightSim2Grid vs RustPower
"""

import time
import warnings
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from lightsim2grid.gridmodel import init_from_pandapower
from lightsim2grid.solver import KLUSolver, AlgorithmType
import rustpower
import lightsim2grid

warnings.filterwarnings('ignore')

TOL = 1e-6
MAX_ITER = 20
NUM_TRIALS_COLD = 10
NUM_TRIALS_HOT = 20


print("Warming up Pandapower Numba JIT compiler...")
dummy_net = pn.case39()
pp.runpp(dummy_net)

def run_benchmark(net_name, net):
    print(f"\n=========================================================")
    print(f" BENCHMARKING NETWORK: {net_name} ({len(net.bus)} buses)")
    print(f"=========================================================")
    
    num_buses = len(net.bus)
    V_init_flat = np.ones(num_buses, dtype=np.complex128)

    # Calculate V_init that satisfies PV and Slack bus magnitude setpoints for LS2G pure solver
    V_init_compensated = np.ones(num_buses, dtype=np.complex128)
    for _, row in net.ext_grid.iterrows():
        V_init_compensated[int(row['bus'])] = row['vm_pu']
    for _, row in net.gen.iterrows():
        V_init_compensated[int(row['bus'])] = row['vm_pu']

    # ---------------------------------------------------------
    # ROUND 1: Cold Start Performance (Includes Symbolic Analysis)
    # ---------------------------------------------------------
    
    # Pandapower
    times_pp_cold = []
    for _ in range(NUM_TRIALS_COLD):
        start = time.perf_counter()
        pp.runpp(net, init="flat", recycle=False, tolerance_mva=TOL)
        times_pp_cold.append(time.perf_counter() - start)
    pp_cold_ms = np.mean(times_pp_cold) * 1000
    pp_iters = net._ppc['iterations']

    # LS2G Pure Solver
    ls_model = init_from_pandapower(net)
    ls_model.change_solver(AlgorithmType.NRSing_KLU)
    ls_model.ac_pf(V_init_flat.copy(), MAX_ITER, TOL) 

    Ybus = ls_model.get_Ybus_solver()
    Sbus = ls_model.get_Sbus_solver()
    slack_ids = ls_model.get_slack_ids_solver()
    slack_weights = ls_model.get_slack_weights_solver()
    pv = ls_model.get_pv_solver()
    pq = ls_model.get_pq_solver()
    ls_solver = KLUSolver()

    # Warmup
    ls_solver.compute_pf(Ybus, V_init_compensated.copy(), Sbus, slack_ids, slack_weights, pv, pq, MAX_ITER, TOL)
    ls2g_iters = ls_solver.get_nb_iter()

    times_ls2g_cold = []
    for _ in range(NUM_TRIALS_COLD):
        start = time.perf_counter()
        ls_solver.compute_pf(Ybus, V_init_compensated.copy(), Sbus, slack_ids, slack_weights, pv, pq, MAX_ITER, TOL)
        times_ls2g_cold.append(time.perf_counter() - start)
    ls2g_cold_ms = np.mean(times_ls2g_cold) * 1000

    # RustPower
    rp_model = rustpower.PowerGrid.from_pandapower(net)
    report = rp_model.solve(V_init_flat.copy(), max_iter=MAX_ITER, tol=TOL)
    rp_iters = report.iterations

    times_rp_cold = []
    for _ in range(NUM_TRIALS_COLD):
        rp_model.init_pf()
        start = time.perf_counter()
        rp_model.solve(V_init_flat.copy(), max_iter=MAX_ITER, tol=TOL)
        times_rp_cold.append(time.perf_counter() - start)
    rp_cold_ms = np.mean(times_rp_cold) * 1000

    print(f"--- COLD START (Iterations: PP={pp_iters}, LS2G={ls2g_iters}, RP={rp_iters}) ---")
    print(f"[Pandapower] {pp_cold_ms:.3f} ms")
    print(f"[LS2G (KLU)] {ls2g_cold_ms:.3f} ms")
    print(f"[RustPower]  {rp_cold_ms:.3f} ms")

    # ---------------------------------------------------------
    # ROUND 2: Hot Loop Performance (Maximum Cached State)
    # ---------------------------------------------------------
    
    # Pandapower
    pp.runpp(net, init="flat", recycle=True, tolerance_mva=TOL)
    times_pp_hot = []
    for _ in range(NUM_TRIALS_HOT):
        # Force flat start in PPC to ensure Newton iterations actually run
        net._ppc['bus'][:, 7] = 1.0 # vm
        net._ppc['bus'][:, 8] = 0.0 # va
        start = time.perf_counter()
        pp.runpp(net, init="flat", recycle=dict(ppc=True, Ybus=True, bus_pq=False, trafo=False, gen=False), tolerance_mva=TOL)
        times_pp_hot.append(time.perf_counter() - start)
    pp_hot_ms = np.mean(times_pp_hot) * 1000
    pp_hot_iters = net._ppc['iterations']

    # LS2G GridModel
    
    ls_model.ac_pf(V_init_flat.copy(), MAX_ITER, TOL)
    times_ls2g_hot = []
    for _ in range(NUM_TRIALS_HOT):
        start = time.perf_counter()
        ls_model.ac_pf(V_init_flat.copy(), MAX_ITER, TOL)
        times_ls2g_hot.append(time.perf_counter() - start)
    ls2g_hot_ms = np.mean(times_ls2g_hot) * 1000
    ls2g_hot_iters = ls_model.get_solver().get_nb_iter()

    # RustPower
    rp_model.init_pf()
    rp_model.enable_cache(True)
    rp_model.solve(V_init_flat.copy(), max_iter=MAX_ITER, tol=TOL)
    times_rp_hot = []
    for _ in range(NUM_TRIALS_HOT):
        start = time.perf_counter()
        report = rp_model.solve(V_init_flat.copy(), max_iter=MAX_ITER, tol=TOL)
        times_rp_hot.append(time.perf_counter() - start)
    rp_hot_ms = np.mean(times_rp_hot) * 1000
    rp_hot_iters = report.iterations

    print(f"\n--- HOT LOOP (Iterations: PP={pp_hot_iters}, LS2G={ls2g_hot_iters}, RP={rp_hot_iters}) ---")
    print(f"[Pandapower] {pp_hot_ms:.3f} ms")
    print(f"[LS2G (KLU)] {ls2g_hot_ms:.3f} ms")
    print(f"[RustPower]  {rp_hot_ms:.3f} ms")


if __name__ == "__main__":
    print("--- Environment ---")
    print(f"Pandapower Version: {pp.__version__}")
    print(f"LightSim2Grid Version: {lightsim2grid.__version__ if hasattr(lightsim2grid, '__version__') else 'Unknown'}")
    print(f"RustPower Version: {rustpower.version()}")
    print(f"RustPower Active Features: {rustpower.features()}")
    
    cases = {
        "Case 39": pn.case39,
        "Case 118": pn.case118,
        "Case 1354 PEGASE": pn.case1354pegase,
        "Case 9241 PEGASE": pn.case9241pegase
    }
    
    for name, case_fn in cases.items():
        net = case_fn()
        run_benchmark(name, net)

