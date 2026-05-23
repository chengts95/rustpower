import pandapower as pp
import pandapower.networks as pn
import time
import numpy as np
try:
    import lightsim2grid
    from lightsim2grid.gridmodel import init_from_pandapower
    from lightsim2grid import SolverType
    LS2G_AVAILABLE = True
except ImportError:
    LS2G_AVAILABLE = False

try:
    import rustpower
    RP_AVAILABLE = True
except ImportError:
    RP_AVAILABLE = False

def benchmark(name, net_func, iterations=100):
    print(f"\nBenchmarking {name}...")
    net = net_func()
    
    # Pandapower (Default)
    pp.runpp(net)
    vm_pp = net.res_bus.vm_pu.values
    pp_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        pp.runpp(net)
        pp_times.append(time.perf_counter() - start)
    print(f"Pandapower (Default): Avg: {np.mean(pp_times)*1000:.3f}ms, Iterations: {net._ppc['iterations']}")
    print(f"  Vm range: [{np.min(vm_pp):.4f}, {np.max(vm_pp):.4f}]")

    # Native LightSim2Grid KLU
    if LS2G_AVAILABLE:
        try:
            model = init_from_pandapower(net)
            v_ones = np.ones(model.total_bus(), dtype=np.complex128)
            model.change_solver(SolverType.KLU)
            
            # Warmup and get results
            v_res = model.ac_pf(v_ones.copy(), 10, 1e-8)
            vm_ls = np.abs(v_res)
            
            ls_times = []
            for _ in range(iterations):
                v_init = v_ones.copy()
                start = time.perf_counter()
                model.ac_pf(v_init, 10, 1e-8)
                ls_times.append(time.perf_counter() - start)
            print(f"LightSim2Grid (KLU Native): Avg: {np.mean(ls_times)*1000:.3f}ms")
            if len(vm_ls) > 0:
                print(f"  Vm range: [{np.min(vm_ls):.4f}, {np.max(vm_ls):.4f}]")
            else:
                print("  LightSim2Grid did not converge (empty result)")
        except Exception as e:
            print(f"LightSim2Grid KLU failed: {e}")

def benchmark_rp(case_path, iterations=100):
    """Time RustPower via Python wrapper to quantify FFI overhead."""
    if not RP_AVAILABLE:
        return
    grid = rustpower.PowerGrid(case_path=case_path)
    grid.init_pf()
    grid.run_pf()  # warmup
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        grid.run_pf()
        times.append(time.perf_counter() - start)
    print(f"RustPower (Python wrapper): Avg: {np.mean(times)*1000:.3f} ms  "
          f"min: {np.min(times)*1000:.3f} ms  (n={iterations})")
    print(f"  [compare with Rust-native benchmark to isolate FFI overhead]")

if __name__ == "__main__":
    benchmark("IEEE 39", pn.case39, 100)
    benchmark("IEEE 118", pn.case118, 100)
    benchmark("PEGASE 9241", pn.case9241pegase, 150)

    print("\n--- RustPower FFI overhead check ---")
    print("IEEE 118:")
    benchmark_rp('cases/IEEE118/data.zip', iterations=300)
    print("PEGASE 9241:")
    benchmark_rp('cases/pegase9241/data.zip', iterations=30)
