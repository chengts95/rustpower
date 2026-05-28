import pandapower as pp
import pandapower.networks as pn
import time
import numpy as np

def benchmark_opf(name, net_func, iterations=20):
    print(f"\nBenchmarking OPF for {name}...")
    net = net_func()
    
    # Warmup
    try:
        pp.runopp(net)
        print(f"  Converged: {net.OPF_converged}")
    except Exception as e:
        print(f"  Warmup failed: {e}")
        return

    times = []
    for i in range(iterations):
        start = time.perf_counter()
        pp.runopp(net)
        times.append(time.perf_counter() - start)
        # print(f"    Iter {i+1}: {times[-1]*1000:.2f}ms")

    print(f"Pandapower OPF ({name}): Avg: {np.mean(times)*1000:.3f}ms, min: {np.min(times)*1000:.3f}ms, max: {np.max(times)*1000:.3f}ms")

if __name__ == "__main__":
    benchmark_opf("IEEE 39", pn.case39, iterations=30)
    benchmark_opf("IEEE 118", pn.case118, iterations=30)
