# Python API Guide

RustPower provides a high-performance, transactional Python API designed for both batch analysis, complex iterative workflows (e.g., time-series or optimization), and direct solver acceleration for libraries like `pandapower`.

## Pandapower Integration Scope

RustPower is designed primarily for native Rust modeling and simulation. Its Python API offers two integration paths:

- **Numerical solver replacement (`NewtonSolver`)**: Seamlessly replace the numerical Newton-Raphson stage using pandapower's `ppc`/`ppci` matrices and vectors. Pandapower prepares the electrical model and manages the surrounding workflow; RustPower solves the supplied numerical system. The Python interface provides zero-copy access to compatible NumPy buffers and zero-allocation result extraction directly into pre-allocated output arrays. The solver still allocates internal storage and workspaces. See the [solver integration report](pandapower_integration_report.md).
- **Network conversion (`PowerGrid.from_pandapower()`)**: Convert supported basic elements into RustPower's native ECS model, where RustPower prepares and solves the network. Basic elements include buses, lines, two-winding transformers, loads, generators, external grids, shunts, and static generators. This does not imply support for every pandapower parameter, element model, option, controller, or analysis feature.

AC power-flow comparisons on selected IEEE benchmark systems validate the tested cases and configurations. Comparisons using pandapower-generated matrices validate the numerical solver; they do not establish feature compatibility for the network importer.

Networks containing switches require an explicit `switch_model="merge"` (native Type A) or `switch_model="admittance"` (native Type B). RustPower detects the need for a choice and raises `ValueError` if none is configured; it does not choose an electrical model implicitly. Both choices enable line and two-winding transformer terminal switching. Bus-bus switches (`et="b"`) and line switches (`et="l"`) are supported. Two-winding transformer switches (`et="t"`) support closed, HV-open, LV-open, and both-open states. An open winding carries no terminal current; the connected winding retains magnetizing current and core losses. Three-winding transformer switches and unknown types are rejected before replacing the grid. Networks without switches need no selection.

```python
grid = rp.PowerGrid.from_pandapower(net, switch_model="merge")
# Or configure the grid before importing/loading:
grid = rp.PowerGrid(switch_model="admittance")
grid.from_pp_net(net)
```

`"merge"` exactly merges closed ideal bus-bus switches. `"admittance"` approximates ideal bus-bus switches with high conductance, which can worsen numerical conditioning. Both model finite switch impedance as pure resistance; this does not reproduce pandapower's configurable switch R/X ratio. Choose the model according to your native modeling requirements. Once selected, the model is reused by imports and fixed for that `PowerGrid`; create another grid to change it. `from_pp_net()`, `load_network()`, and `load_parquet_case()` also accept the keyword to make the first selection. Archive loading requires an explicit choice when switches are present, because solver plugin configuration is not stored in case snapshots.

Public voltage arrays (`v`, `v_init`, `compute_dcpf_v()`) retain dense physical bus order and one entry per original bus, including merged buses. Initial voltages for a merged group use its selected representative (slack, then PV, then PQ; lowest physical index breaks ties).

The CIGRE MV example can be used without removing its closed transformer switches:

```python
import pandapower.networks as pn

net = pn.create_cigre_network_mv(with_der=False)
grid = rp.PowerGrid.from_pandapower(net, switch_model="merge")
grid.solve()
```

### Bus identifiers during conversion

The importer assigns internal `BusID` values `0..n-1` in bus-table row order and preserves the original `net.bus.index` labels as `PPBusID`. It translates bus references in lines, transformers, loads, generators, external grids, shunts, static generators, and switches. For bus-bus switches, both endpoints are translated; for line/transformer switches, `element` remains the device label. Duplicate bus labels and active elements referencing missing buses raise `ValueError` before replacing the existing Python grid. Disabled elements may retain missing connections; these get separate negative internal labels during conversion.

Python handles, editor arguments, branch endpoint columns, and `bus_id` use internal IDs. `BusHandle.pp_id` and the `pp_bus_id` bus-table column expose the original pandapower label (absent for native buses). These labels are preserved in case and result snapshots. `BusID` remains stable when buses are removed; gaps are valid. `BusHandle.matrix_index` and the `matrix_index` bus-table column expose the current dense physical matrix offset, which can change on a topology rebuild. To align imported results with pandapower:

```python
results = grid.res_bus.set_index("pp_bus_id").reindex(net.bus.index)
```

This normalization applies to network ingestion, including CSV cases and `load_network()`. Native grid creation and editing continue to use RustPower IDs.

### Core connectivity and numerical caches

Native callers can obtain a validated connection with `NodeLookup::connection(bus_entity)` and construct reference components directly; preparation generates the label metadata for archives. Runtime connections use world-local Entity references (`FromBusRef`, `ToBusRef`, `TargetBusRef`, and resolved switch bus references). Label components remain the API/archive representation; topology preparation resolves new or deliberately changed labels and preserves established Entity identity. Deleting a bus and reusing its label cannot reconnect an existing Entity reference. Disabled elements may retain unresolved references, but get no matrix indices until their connections are valid.

The core assigns `BusMatrixIndex` and compact `FromBusMatrixIndex`, `ToBusMatrixIndex`, and `TargetBusMatrixIndex` components during preparation. Matrix assembly, injections, AC/DC state updates, Q limits, and SIMD result extraction use these caches directly. Newton iterations continue to operate on numerical arrays. `NodeLookup` provides label-to-Entity, Entity-to-matrix-index, and matrix-index-to-Entity access at boundaries. Solver permutation is separate from the physical matrix offset.

Topology preparation validates references and refreshes caches, including bus scales on same-size replacement. Native callers can use `elements::topology::prepare_topology(world)` to receive validation errors before rebuilding, and should request `FullRebuildEvent` after topology changes. Case archives store labels and reconstruct Entity references and indices in the receiving world; these runtime caches are not serialized.

## Installation

```bash
pip install rustpower
```

---

## Core Workflows

1. **Scenario A: Batch Analysis** — Convert supported basic elements from a pandapower network or CSV-ZIP file, solve with RustPower's ECS model, and analyze DataFrame results.
2. **Scenario B: Parameter Loops & Sweeps** — Modify element properties (`load.p_mw = ...`) in place and re-solve. Runs an incremental warm-start solve with cached matrices.
3. **Scenario C: Transactional Topology Research** — Add or remove buses, lines, loads, etc., using the `grid.edit()` context manager with atomicity and fused batch insertions.
4. **Scenario D: Direct SciPy / Pandapower Solver Backend** — Use `rustpower.solver.NewtonSolver` with pandapower's `ppc`/`ppci` data and $O(\text{nnz})$ CSR-to-CSC permutation to replace the numerical solve in external power flow pipelines.

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
grid.enable_cache(True)       # Already enabled by default; reuses symbolic caches
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

For seamless numerical solver replacement using pandapower's `ppc`/`ppci` data or custom SciPy code, pass the externally prepared matrices, vectors, bus partition, and permutation to `NewtonSolver`. This path retains pandapower's model preparation and is independent of `PowerGrid.from_pandapower()`:

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

Rust array indexing and the solver's sparse matrix representation use `usize`, a pointer-sized unsigned integer (64 bits on 64-bit platforms). NumPy/SciPy index arrays commonly use signed `int32` or `int64`; the binding converts these into owned `usize` buffers during setup. This type conversion is required for those signed inputs by the current solver representation. Integer index arrays are therefore outside the zero-copy claim for compatible numerical buffers.

For repeated solves, allocate NumPy output buffers once and reuse them:

```python
v_out = np.empty(solver.n_buses, dtype=np.complex128)
vm_out = np.empty(solver.n_buses, dtype=np.float64)
scalc_out = np.empty(solver.n_buses, dtype=np.complex128)

if solver.solve(max_iter=15, tol=1e-8):
    solver.extract_results(v=v_out, vm=vm_out, scalc=scalc_out)
```

`extract_results()` borrows writable, contiguous 1D NumPy buffers and writes results directly in original bus order, without allocating temporary result arrays. After a solve, power injections use the populated solver cache, so their extraction also avoids heap allocations. Compatible contiguous `complex128` inputs are borrowed directly at the Python binding boundary; the solver separately allocates internal storage and workspaces, and context setup copies and reorders data into that storage. Zero-copy interface access does not mean an allocation-free solver. Convenience getters such as `get_voltage()` return newly allocated arrays.

---

## API Reference

### `PowerGrid`

#### Constructors & Ingestion
- `PowerGrid(case_path=None, qlim=False, *, switch_model=None, **kwargs)`: Initialize a grid. Can load from a CSV-ZIP case file or start empty.
- `PowerGrid.from_pandapower(net, qlim=False, *, switch_model=None) -> PowerGrid`: Classmethod to construct a `PowerGrid` directly from a pandapower `net` object. Set `qlim=True` to enforce generator reactive-power limits.
- `grid.from_pp_net(net, *, switch_model=None)`: Ingest a pandapower `net` into an existing `PowerGrid` instance.
- `grid.load_network(net, *, switch_model=None)`: Ingest an internal `rustpower.Network` struct.

#### Solvers & Algorithms
- `solve(v_init=None, max_iter=None, tol=None) -> SolveReport`: Run power flow analysis.
  - `v_init`: Optional complex array in **dense physical matrix order** (`matrix_index`) for custom warm-start.
- `enable_cache(enable=True)`: Enable or disable Newton workspace/Jacobian caching. Enabled by default; the setting survives model rebuilds. Structural changes invalidate affected solver caches automatically.
- `clear_cache()`: Clear solver caches while preserving the cache policy, model and current Q-limit clamps. Caches are rebuilt at the next solve when enabled.
- `init_pf()`: Rebuild the model and restore bus classifications from the current device data. With Q limits enabled, this clears previous PV-to-PQ clamps so the next solve determines them again. Ordinary parameter updates preserve existing clamps.
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
- `grid.res_trafo -> pd.DataFrame`: Transformer results. `trafo_id` is the original transformer table label (missing for native transformers without a label).
- `grid.v -> np.ndarray`: Complex voltage vector in **bus-id order**. Assignable for warm-starts (`grid.v = custom_v`).
- `grid.y_bus() -> scipy.sparse.csc_matrix`: Admittance matrix.
- `grid.converged -> bool`: Whether the last solve converged.
- `grid.iterations -> int`: Number of iterations taken.
- `grid.post_process()`: Manually trigger result extraction (normally executed lazily).

#### Optional connection check

`grid.validate() -> list[str]` lists missing or invalid bus/line references, including out-of-service elements. It checks the current grid and changes nothing. An empty list means no reference issues were found; this does not test power-flow convergence or every electrical parameter. Saving does not run this check automatically.

```python
for problem in grid.validate():
    print(problem)
```

Archives store connection labels. After loading, a connection may attach to a replacement bus with the same ID. This can be intentional: reconnecting retained elements to a replacement bus is a valid use case. `validate()` reports missing or invalid references; it cannot decide whether a valid connection is the one you intended. Existing runtime Entity references still keep their original identity.

#### Persistence (Apache Arrow / Parquet)
- `grid.get_parquet_case() -> bytes`: Export grid topology and parameters to an in-memory ZIP of Parquet files.
- `grid.get_parquet_results() -> bytes`: Export simulation results to an in-memory ZIP of Parquet files.
- `grid.load_parquet_case(zip_bytes: bytes, *, switch_model=None)`: Load and reconstruct ECS state from a ZIP of Parquet files.

#### Transactional Editing (`grid.edit()`)

Zones are string labels. Pandapower numeric zones are converted to text; missing zones become `""`. Use `bus_handle.zone` to read or update the label.

Pandapower line switches (`et="l"`) are honored when a switch model is explicitly selected. An open terminal disconnects that bus while retaining the line's charging at the connected terminal; opening both terminals disconnects the entire line. Switch targets use the original line table indices, including sparse or reordered indices. Both explicit switch models include this line-terminal behavior. Two-winding transformer switches also support open terminals. The connected winding retains the transformer core and magnetizing losses; the disconnected winding has zero terminal current. Opening both windings disconnects the transformer. The line-switch plugin prepares terminal state and the effective patch before core Ybus assembly. Line results reconstruct the disconnected line-side voltage; fully disconnected and out-of-service lines publish zero results. Runtime Entity references and terminal caches are rebuilt after archive loading. Transformer results likewise reconstruct the voltage on the disconnected winding. Fully disconnected and out-of-service transformers publish zero device results. Opening a branch can leave buses without a supply; automatic removal of such islands is not part of this terminal model.

- `add_bus(vn_kv, name=None, vm_min=0.9, vm_max=1.1, zone="") -> (int, BusHandle)`
- `add_line(from_bus, to_bus, length_km, std_type=None, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, g_us_per_km=0.0, parallel=1, max_i_ka=0.0, name=None) -> LineHandle`
- `add_load(bus, p_mw=0.0, q_mvar=0.0, const_z_percent=0.0, const_i_percent=0.0, sn_mva=None, scaling=1.0, in_service=True, name=None) -> LoadHandle`
- `add_gen(bus, p_mw=0.0, vm_pu=1.0, sn_mva=None, min_q_mvar=-1e9, max_q_mvar=1e9, min_p_mw=-1e9, max_p_mw=1e9, scaling=1.0, in_service=True, name=None) -> GenHandle`
- `add_ext_grid(bus, vm_pu=1.0, va_degree=0.0, name=None) -> ExtGridHandle`
- `add_trafo(hv_bus, lv_bus, sn_mva=100.0, vn_hv_kv=110.0, vn_lv_kv=10.0, vk_percent=10.0, vkr_percent=1.0, pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0, tap_pos=0.0, tap_neutral=0.0, tap_step_percent=1.0, tap_min=-10.0, tap_max=10.0, in_service=True, name=None) -> TrafoHandle`
- `add_shunt(bus, q_mvar=0.0, p_mw=0.0, vn_kv=110.0, step=1, max_step=1, in_service=True, name=None) -> ShuntHandle`
- `remove(handle)`: Defer deletion until commit. Deleting a bus keeps its connected elements and puts them out of service. Reconnect them before enabling them again. Foreign-grid and stale handles are rejected; abort preserves queued removals.

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
