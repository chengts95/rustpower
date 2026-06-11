"""
Focused profiling of PowerGrid.solve() on pegase9241.
Measures:
  1. Solver core (newton_pf) via internal DEBUG prints
  2. Post-processing via internal DEBUG prints
  3. Total Python-measured wall time per solve()

Run with: python -u scripts/profile_solve.py
(requires release build: maturin develop --release)
"""
import rustpower
import time
import numpy as np
import sys

case = "cases/pegase9241/data.zip"
N_WARMUP = 3
N_ITERS = 20

print(f"=== RustPower solve() profile: {case} ===")
print(f"Warmup: {N_WARMUP}, Iterations: {N_ITERS}")
print()

grid = rustpower.PowerGrid(case_path=case)
print("Grid loaded.\n")

# Warmup
print("--- Warmup ---")
for i in range(N_WARMUP):
    r = grid.solve()
    print(f"  warmup {i}: converged={r.converged}, iters={r.iterations}, runtime_ms={r.runtime_ms:.3f}")
print()

# Timed runs
print("--- Timed runs (Python wall clock) ---")
py_times = []
for i in range(N_ITERS):
    sys.stdout.flush()
    t0 = time.perf_counter()
    r = grid.solve()
    t1 = time.perf_counter()
    dt = (t1 - t0) * 1e3
    py_times.append(dt)
    # DEBUG lines from Rust are printed automatically

print()
print(f"--- Summary (N={N_ITERS}) ---")
arr = np.array(py_times)
print(f"  Python wall-clock per solve():")
print(f"    mean:   {arr.mean():.3f} ms")
print(f"    median: {np.median(arr):.3f} ms")
print(f"    min:    {arr.min():.3f} ms")
print(f"    max:    {arr.max():.3f} ms")
print(f"    std:    {arr.std():.3f} ms")
print()
print("Compare the Python wall-clock with the Rust-internal timings above")
print("to isolate FFI/post-processing overhead.")
