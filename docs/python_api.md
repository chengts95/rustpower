# Python API Guide

RustPower provides a high-performance, transactional Python API designed for both batch analysis and complex iterative workflows (e.g., time-series or optimization).

## Installation

```bash
pip install rustpower
```

## Core Concepts

The Python API is built around three main scenarios:

1.  **Scenario A: Batch Analysis** - Loading a case and solving it once.
2.  **Scenario B: Parameter Loops** - Modifying parameters (like load or generation) and re-solving. This uses a high-performance **incremental path** with warm starts.
3.  **Scenario C: Topology Research** - Adding or removing buses, lines, etc. using a **transactional editor**.

---

## Usage Examples

### 1. Basic Solve (Scenario A)

```python
import rustpower as rp

# Load a case (PandaPower .zip format)
grid = rp.PowerGrid("cases/IEEE118/data.zip")

# Run power flow
report = grid.solve()

if report.converged:
    print(f"Converged in {report.iterations} iterations")
    print(f"Runtime: {report.runtime_ms:.2f} ms")
    
    # Access results as Pandas DataFrames
    print(grid.res_bus.head())
    print(grid.res_line.head())
else:
    print("Power flow diverged!")
```

### 2. Fast Parameter Sweeps (Scenario B)

Modifying element properties directly takes an **incremental path**, bypassing expensive matrix rebuilds if only power values or voltage setpoints change.

```python
load = grid.load(bus=5)  # Get a handle to the load at bus 5

for p in [50.0, 60.0, 70.0]:
    load.p_mw = p        # Immediate-mode property write
    report = grid.solve() # Automatically runs incremental solve with warm start
    print(f"P={p} MW, Bus 5 Voltage: {grid.bus(5).vm_pu:.4f}")
```

### 3. Transactional Topology Changes (Scenario C)

All topology mutations (adding/removing elements) must go through the `grid.edit()` context manager. This ensures atomicity and allows for fused insertions.

```python
with grid.edit() as e:
    # Add a new bus and a line connecting it
    new_bus_id, b_handle = e.add_bus(110.0, name="Expansion")
    e.add_line(from_bus=0, to_bus=new_bus_id, length_km=10.0, r_ohm_per_km=0.1, x_ohm_per_km=0.4)
    
    # Add a load to the new bus
    e.add_load(new_bus_id, p_mw=10.0, q_mvar=2.0)

# Once the 'with' block exits, the transaction is committed.
# The next solve() will automatically trigger a full matrix rebuild.
grid.solve()
```

---

## API Reference

### `PowerGrid`
The main entry point.
- `__init__(case_path=None, qlim=False)`: Initialize a grid. `qlim=True` enables reactive power limit enforcement.
- `solve(v_init=None)`: Execute power flow. Returns a `SolveReport`.
  `v_init` is an optional complex warm-start vector in **bus-id order** (same
  layout as `v`). PV/slack setpoints are re-pinned after the override, so
  `v_init` only changes the Newton starting point, never the physics.
- `edit()`: Returns a `GridEditor` for topology changes.
- `bus(id)`, `load(bus_id)`, `gen(bus_id)`, `line(from_bus, to_bus)`: Find elements. Returns `None` if not found.
- `res_bus`, `res_line`: Result DataFrames.
- `v`: Complex voltage array (p.u.), **indexed by bus id** (element `i` is
  bus `i`). Readable after `solve()`; assignable as a warm-start vector.
  The internal PQ/PV/Slack solver permutation is applied automatically in
  both directions — users never see permuted data.

### `GridEditor`
- `add_bus(...)`, `add_line(...)`, `add_load(...)`, `add_gen(...)`, `add_trafo(...)`: Add elements.
- `remove(handle)`: Remove an element.

### Element Handles (`BusHandle`, `LoadHandle`, etc.)
- Property access (e.g., `load.p_mw`) maps to the underlying ECS components.
- Assigning to a property (e.g., `load.p_mw = 10`) schedules an update.
- `in_service`: Boolean property. Setting this to `False` logically removes the element from the network.

---

## Performance Notes

- **Zero-Allocation Hot Path**: The core Newton-Raphson loop avoids all heap allocations during iterations.
- **KLU Solver**: RustPower uses the KLU sparse solver, which is highly efficient for power system matrices.
- **Warm Starts**: Subsequent solves after parameter changes use the previous solution as the initial guess, often converging in 2-3 iterations.
