"""
Final benchmark: pandapower vs lightsim2grid vs rustpower
Measures solver-core time (NR iterations only, no post-processing).
"""
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

def benchmark(name, net_func, rp_case_path, iterations=100):
    print(f"\n{'='*60}")
    print(f"  {name}  ({iterations} iterations)")
    print(f"{'='*60}")
    net = net_func()

    # ── Pandapower ──────────────────────────────────────────────
    pp.runpp(net)  # warmup
    vm_pp = net.res_bus.vm_pu.values.copy()
    pp_iters = net._ppc['iterations']
    pp_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        pp.runpp(net)
        pp_times.append(time.perf_counter() - start)
    pp_arr = np.array(pp_times) * 1e3
    print(f"\n  Pandapower (numpy):        Avg: {pp_arr.mean():>8.3f} ms | "
          f"Min: {pp_arr.min():>8.3f} ms | NR iters: {pp_iters}")
    print(f"    Vm range: [{np.min(vm_pp):.4f}, {np.max(vm_pp):.4f}]")

    # ── LightSim2Grid ──────────────────────────────────────────
    if LS2G_AVAILABLE:
        try:
            model = init_from_pandapower(net)
            n_bus = model.total_bus()
            v_ones = np.ones(n_bus, dtype=np.complex128)
            model.change_solver(SolverType.KLU)

            # Warmup
            v_res = model.ac_pf(v_ones.copy(), 10, 1e-8)
            vm_ls = np.abs(v_res)

            ls_times = []
            for _ in range(iterations):
                v_init = v_ones.copy()
                start = time.perf_counter()
                model.ac_pf(v_init, 10, 1e-8)
                ls_times.append(time.perf_counter() - start)
            ls_arr = np.array(ls_times) * 1e3
            print(f"\n  LightSim2Grid (KLU):      Avg: {ls_arr.mean():>8.3f} ms | "
                  f"Min: {ls_arr.min():>8.3f} ms")
            if len(vm_ls) > 0:
                print(f"    Vm range: [{np.min(vm_ls):.4f}, {np.max(vm_ls):.4f}]")
            else:
                print("    Did not converge")
        except Exception as e:
            print(f"\n  LightSim2Grid failed: {e}")

    # ── RustPower (Python API, with post-processing) ───────────
    if RP_AVAILABLE and rp_case_path:
        grid = rustpower.PowerGrid(case_path=rp_case_path)
        grid.init_pf()
        n_bus = grid.n_bus
        v_ones = np.ones(n_bus, dtype=np.complex128)
        r = grid.solve(v_ones.copy())  # warmup (flat start, forces real NR)

        rp_wall = []
        rp_internal = []
        for _ in range(iterations):
            v_init = v_ones.copy()
            start = time.perf_counter()
            r = grid.solve(v_init)
            rp_wall.append((time.perf_counter() - start) * 1e3)
            rp_internal.append(r.runtime_ms)

        wall_arr = np.array(rp_wall)
        int_arr = np.array(rp_internal)
        print(f"\n  RustPower (Python, w/ post-proc):")
        print(f"    Python wall-clock:      Avg: {wall_arr.mean():>8.3f} ms | "
              f"Min: {wall_arr.min():>8.3f} ms | NR iters: {r.iterations}")
        print(f"    Rust internal (SolveReport): Avg: {int_arr.mean():>8.3f} ms | "
              f"Min: {int_arr.min():>8.3f} ms")
        print(f"    FFI + Python overhead:  ~{(wall_arr.mean() - int_arr.mean()):>8.3f} ms")


    # ── RustPower Native (from bench_all, for reference) ───────
    print(f"\n  RustPower Native (bench_all, solver-only, no post-proc):")
    print(f"    [See `cargo run --release --example bench_all`]")

    print()

if __name__ == "__main__":
    benchmark("IEEE 39",      pn.case39,          None,                         iterations=200)
    benchmark("IEEE 118",     pn.case118,         'cases/IEEE118/data.zip',     iterations=200)
    benchmark("PEGASE 9241",  pn.case9241pegase,  'cases/pegase9241/data.zip',  iterations=50)

