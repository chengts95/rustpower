# Python API Guide

RustPower provides a high-performance, transactional Python API designed for both batch analysis, complex iterative workflows (e.g., time-series or optimization), and direct solver acceleration for libraries like `pandapower`.

## Installation

```bash
pip install rustpower
```

---

## Core Workflows

1. **Scenario A: Batch Analysis** — Ingest a network from pandapower or a CSV-ZIP file, solve, and analyze DataFrame results.
2. **Scenario B: Parameter Loops & Sweeps** — Modify element properties (`load.p_mw = ...`) in place and re-solve. Runs an incremental warm-start solve with cached matrices.
3. **Scenario C: Transactional Topology Research** — Add or remove buses, lines, loads, etc., using the `grid.edit()` context manager with atomicity and fused batch insertions.
4. **Scenario D: Direct SciPy / Pandapower Solver Backend** — Use `rustpower.solver.NewtonSolver` with $O(\text{nnz})$ CSR-to-CSC permutation to accelerate external power flow pipelines.

---

## Usage Examples

### 1. Ingestion from Pandapower & Solving (Scenario A)

```python
import pandapower.networks as nw
import rustpower as rp

# Ingest directly from an in-memory pandapower network
net = nw.case118()
grid = rp.PowerGrid.from_pandapower(net)

# Or load from a CSV-ZIP archive
# grid = rp.PowerGrid("cases/IEEE118/data.zip")

# Enable solver optimizations (optional)
grid.enable_cache(True)       # Reuses symbolic LU factorization
grid.enable_dcpf_init(True)   # Warm-starts voltage angles via DCPF

# Solve power flow
report = grid.solve(tol=1e-8, max_iter=20)

if report.converged:
    print(f"Converged in {report.iterations} iterations, runtime: {report.runtime_ms:.2f} ms")
    print("Bus results:\n", grid.res_bus.head())
    print("Line results:\n", grid.res_line.head())
    print("Transformer results:\n", grid.res_trafo.head())
else:
    print("Power flow diverged!")
```

### 2. Fast Parameter Sweeps (Scenario B)

Modifying element properties directly takes an **incremental path**, updating internal state while reusing pre-factored sparsity structures:

```python
load = grid.load(bus=5)  # Get handle to load at bus 5

for p in [50.0, 60.0, 70.0, 80.0]:
    load.p_mw = p         # Immediate property update
    report = grid.solve() # Incremental solve with warm start
    v5 = grid.bus(5).vm_pu
    print(f"P={p:5.1f} MW -> Bus 5 Vm={v5:.4f} p.u. (iters: {report.iterations})")
```

### 3. Transactional Topology Modifications (Scenario C)

All topology additions and removals go through `grid.edit()`. Fused batch insertion creates entities in a single step without archetype fragmentation:

```python
with grid.edit() as e:
    # 1. Add a new 110 kV bus
    new_bus_id, bus_handle = e.add_bus(vn_kv=110.0, name="Substation_West")
    
    # 2. Connect it to existing bus 10 via a new line
    e.add_line(from_bus=10, to_bus=new_bus_id, length_km=12.5, r_ohm_per_km=0.08, x_ohm_per_km=0.35)
    
    # 3. Add a load and generator
    e.add_load(bus=new_bus_id, p_mw=25.0, q_mvar=8.0)
    e.add_gen(bus=new_bus_id, p_mw=10.0, vm_pu=1.02)

# Exiting the 'with' block commits the transaction and marks topology as dirty.
# The subsequent solve() automatically triggers a full matrix rebuild.
report = grid.solve()
print("Solved after expansion:", report.converged)
```

### 4. Advanced Convergence: DCPF Warm Start & Iwamoto Multiplier

For stressed networks, ill-conditioned lines, or heavy loading conditions:

```python
# Option 1: DCPF warm start
# Solves DC power flow linear system first to initialize voltage angles
grid.enable_dcpf_init(True)
grid.solve()

# Option 2: Compute DCPF solution vector standalone
v_dcpf = grid.compute_dcpf_v()  # Complex numpy array in bus-id order

# Option 3: Iwamoto Optimal Multiplier
# Computes optimal deceleration step-size mu at each Newton step to guarantee convergence
grid.enable_iwamoto(True)
grid.solve(max_iter=30)
```

### 5. Low-Level Stateful Solver (`rustpower.solver.NewtonSolver`)

For direct integration with pandapower's `ppci` or custom SciPy code:

```python
from rustpower.solver import NewtonSolver
import numpy as np

solver = NewtonSolver()
# Cache is always enabled by default for maximum performance (10-25 us solves)

# y_indptr, y_indices, y_data: CSR format from SciPy
# p_vec, p_inv: Permutations mapping [PQ | PV | Slack]
solver.setup_context(
    y_indptr, y_indices, y_data,
    s_bus, v_init,
    p_vec, p_inv,
    npv=num_pv, npq=num_pq
)

converged = solver.solve(max_iter=15, tol=1e-8)
if converged:
    v_final = solver.get_voltage()   # Numpy complex128 array in original bus order
    iterations = solver.get_iterations()
    s_calc = solver.get_scalc()      # Exact bus power injections
```

---

## API Reference

### `PowerGrid`

#### Constructors & Ingestion
- `PowerGrid(case_path=None, qlim=False, f_hz=50.0, sn_mva=100.0)`: Initialize a grid. Can load from a CSV-ZIP case file or start empty.
- `PowerGrid.from_pandapower(net) -> PowerGrid`: Classmethod to construct a `PowerGrid` directly from a pandapower `net` object.
- `grid.from_pp_net(net)`: Ingest a pandapower `net` into an existing `PowerGrid` instance.
- `grid.load_network(net)`: Ingest an internal `rustpower.Network` struct.

#### Solvers & Algorithms
- `solve(max_iter=10, tol=1e-8, v_init=None) -> SolveReport`: Run power flow analysis.
  - `v_init`: Optional complex array in **bus-id order** for custom warm-start.
- `enable_cache(enable=True)`: Enable or disable Jacobian pattern and symbolic LU factorization caching across solves.
- `enable_dcpf_init(enable=True)`: Enable or disable automatic DC power flow angle initialization prior to Newton-Raphson.
- `compute_dcpf_v() -> np.ndarray`: Compute DCPF voltage vector (complex, bus-id ordered) without running full AC power flow.
- `apply_dcpf_init()`: Manually solve DCPF and update internal `v_bus_init`.
- `enable_iwamoto(enable=True)`: Enable or disable Iwamoto optimal multiplier method for robust convergence under heavy loads.
- `reset_state()`: Reset internal solver state and caches.

#### Element Queries & Handles
- `grid.bus(id: int) -> Optional[BusHandle]`: Get handle to a bus.
- `grid.load(bus=None, name=None) -> Optional[LoadHandle]`: Get first load matching criteria.
- `grid.loads(bus=None) -> list[LoadHandle]`: Get all loads (optionally filtered by bus ID).
- `grid.gen(bus=None, name=None) -> Optional[GenHandle]`: Get generator handle.
- `grid.line(from_bus: int, to_bus: int) -> Optional[LineHandle]`: Get line handle.
- `grid.describe() -> pd.DataFrame`: Summary of grid elements and counts.
- `grid.display_case_buses() -> pd.DataFrame`: Overview of all bus parameters.
- `grid.display_case_loads() -> pd.DataFrame`: Overview of all load parameters.

#### Results
- `grid.res_bus -> pd.DataFrame`: Bus results (`vm_pu`, `va_degree`, `p_mw`, `q_mvar`).
- `grid.res_line -> pd.DataFrame`: Line results (`p_from_mw`, `q_from_mvar`, `p_to_mw`, `q_to_mvar`, `pl_mw`, `ql_mvar`, `i_ka`, `vm_from_pu`, `va_from_degree`, `vm_to_pu`, `va_to_degree`, `loading_percent`).
- `grid.res_trafo -> pd.DataFrame`: Transformer results.
- `grid.v -> np.ndarray`: Complex voltage vector in **bus-id order**. Assignable for warm-starts (`grid.v = custom_v`).
- `grid.y_bus() -> scipy.sparse.csc_matrix`: Admittance matrix.
- `grid.converged -> bool`: Whether the last solve converged.
- `grid.iterations -> int`: Number of iterations taken.
- `grid.post_process()`: Manually trigger result extraction (normally executed lazily).

#### Persistence (Apache Arrow / Parquet)
- `grid.get_parquet_case() -> bytes`: Export grid topology and parameters to an in-memory ZIP of Parquet files.
- `grid.get_parquet_results() -> bytes`: Export simulation results to an in-memory ZIP of Parquet files.
- `grid.load_parquet_case(zip_bytes: bytes)`: Load and reconstruct ECS state from a ZIP of Parquet files.

#### Transactional Editing (`grid.edit()`)
- `add_bus(vn_kv, name=None, vm_min=0.9, vm_max=1.1, zone=0) -> (int, BusHandle)`
- `add_line(from_bus, to_bus, length_km, std_type=None, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, g_us_per_km=0.0, parallel=1, max_i_ka=0.0, name=None) -> LineHandle`
- `add_load(bus, p_mw=0.0, q_mvar=0.0, const_z_percent=0.0, const_i_percent=0.0, sn_mva=None, scaling=1.0, in_service=True, name=None) -> LoadHandle`
- `add_gen(bus, p_mw=0.0, vm_pu=1.0, sn_mva=None, min_q_mvar=-1e9, max_q_mvar=1e9, min_p_mw=-1e9, max_p_mw=1e9, scaling=1.0, in_service=True, name=None) -> GenHandle`
- `add_ext_grid(bus, vm_pu=1.0, va_degree=0.0, name=None) -> ExtGridHandle`
- `add_trafo(hv_bus, lv_bus, sn_mva=100.0, vn_hv_kv=110.0, vn_lv_kv=10.0, vk_percent=10.0, vkr_percent=1.0, pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0, tap_pos=0.0, tap_neutral=0.0, tap_step_percent=1.0, tap_min=-10.0, tap_max=10.0, in_service=True, name=None) -> TrafoHandle`
- `add_shunt(bus, q_mvar=0.0, p_mw=0.0, vn_kv=110.0, step=1, max_step=1, in_service=True, name=None) -> ShuntHandle`
- `remove(handle)`: Mark an element for deletion.

---

### `rustpower.solver.NewtonSolver`

Low-level stateful Newton-Raphson solver designed for high-throughput iterations and custom numerical pipelines (e.g. accelerating pandapower's `ppci` directly).

- `__init__()`: Initialize a new `NewtonSolver` instance.
- `setup_from_nodes(y_indptr, y_indices, y_data, s_bus, v_init, pq, pv, ref)`: Zero-boilerplate setup. Automatically concatenates `[pq, pv, ref]` partitions to build internal solver permutation `p_vec`, `p_inv`, `npv`, and `npq`, and automatically invalidates any stale cache.
- `setup_context(y_indptr, y_indices, y_data, s_bus, v_init, p_vec, p_inv, npv, npq)`: Ingest CSR Ybus matrix and explicit permutation vectors. Converts CSR to CSC in $O(\text{nnz})$ without sorting and sets up permuted matrices.
- `set_v_init(v_init)` / `v_init = ...`: Update the entire initial voltage vector (in **original bus order**) without re-running matrix transformations.
- `set_v_init_at(bus_idx, v)`: In-place update of initial voltage for a single bus.
- `update_v_init_batch(bus_indices, v_values)`: Batch update of initial voltages for specified buses.
- `v_init -> np.ndarray`: Read the current initial voltage vector in original bus order.
- `set_s_bus(s_bus)` / `s_bus = ...`: Update the entire bus power injection vector $S_{bus}$ (in **original bus order**).
- `set_s_bus_at(bus_idx, s)`: In-place update of power injection for a single bus (e.g., dynamic load / generator adjustment).
- `update_s_bus_batch(bus_indices, s_values)`: Batch update of power injections for specified buses.
- `s_bus -> np.ndarray`: Read the current bus power injection vector in original bus order.
- `clear_cache()` / `reset_cache()`: Explicitly clear internal Jacobian sparsity patterns and reset linear solver symbolic factorization.
- `solve(max_iter=10, tol=1e-6) -> bool`: Execute Newton-Raphson iterations. Returns `True` if converged.
- `extract_results(v=None, vm=None, va=None, va_deg=None, scalc=None, p_calc=None, q_calc=None)`: Zero-allocation in-place scatter extraction into pre-allocated 1D NumPy arrays in original bus order. Any parameter passed as `None` is skipped.
- `get_voltage() -> np.ndarray`: Get final complex bus voltages (in **original bus order**).
- `vm -> np.ndarray` / `get_voltage_magnitude() -> np.ndarray`: Extract voltage magnitudes $|V|$ in p.u. in original bus order.
- `va -> np.ndarray`: Extract voltage angles in radians in original bus order.
- `va_deg -> np.ndarray`: Extract voltage angles in degrees in original bus order.
- `get_voltage_angle(deg=False) -> np.ndarray`: Extract voltage angles (radians or degrees).
- `get_iterations() -> int`: Get iteration count of the last solve.
- `get_scalc() -> np.ndarray`: Extract exact bus power injections ($S_{calc} = V \cdot (Y_{bus} \cdot V)^*$) in original bus order.
- `p_calc -> np.ndarray` / `get_p_injections() -> np.ndarray`: Active power injections $P_{calc}$ (MW / p.u.) in original bus order.
- `q_calc -> np.ndarray` / `get_q_injections() -> np.ndarray`: Reactive power injections $Q_{calc}$ (MVAr / p.u.) in original bus order.
- `residual_norm -> float` / `get_residual() -> float`: Final maximum power mismatch $\|F\|_\infty$ after solve.
- `converged -> bool`: Boolean flag indicating whether the last solve converged.
- `n_buses -> int`: Number of buses configured in the solver context.

---

### Module-Level Functions

- `rustpower.version() -> str`: Returns the crate/package version (e.g. `"0.5.2"`).
- `rustpower.features() -> list[str]`: Returns list of compiled features (`"klu"`, `"faer"`, `"rsparse"`, `"arrow"`, etc.).
- `rustpower.load_csv_zip(path: str) -> Network`: Parse a pandapower CSV-ZIP archive into a `Network` data structure.
