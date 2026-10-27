# Python API v2 design: a command-gateway grid model

Design decisions recorded on **2026-06-09**. Later revisions and the final decision
table below supersede the initial API sketches. The [Python API guide](../python_api.md)
describes the current public interface.

The design proceeds from user workflows to a Python contract, then implementation,
then backporting the mutation boundary to native Rust callers.

## 0. Two write modes

Both write paths pass through the mutation gateway:

- **Immediate mode:** assigning an element property, such as `load.p_mw = 30`, executes one command as an automatically committed operation. Ordinary `Mut<T>` writes maintain Bevy change ticks, while explicit dirty classification carries the correctness contract.
- **Deferred mode:** `grid.edit()` queues instructions and applies them at commit. The Harvard command buffer uses fused component insertion to reduce intermediate archetype migrations and fragmentation.

Reloading is a first-class operation: `load_network()` and pandapower ingestion
clear the existing grid entities, ingest the replacement model, and rebuild.
Contract checks require repeated loading to agree with fresh construction, including
entity counts and injection totals. The imported pandapower model is discarded once
ingestion completes; ECS components own the native model.

## 1. User workflows shape the interface

### A. Batch solving

```python
import rustpower as rp

grid = rp.PowerGrid("cases/IEEE118/data.zip")
grid.solve()
print(grid.res_bus)  # A DataFrame projection, familiar to pandapower users.
```

This is the original workflow sketch; see the current guide for supported inputs.

### B. Parameter loops: time series, Monte Carlo, and optimization

```python
load = grid.load(bus=5)
for p in profile:
    load.p_mw = p       # Immediate parameter edit.
    grid.solve()        # Select the appropriate update path internally.
    record(grid.v)
```

A normal parameter loop uses assignment and solve. The mutation pipeline chooses
the projection refresh and initialization behavior from the actual edit.

### C. Topology studies and expansion planning

```python
with grid.edit() as e:
    b, _ = e.add_bus(110.0)
    e.add_line(b5, b, 12.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
grid.solve()
```

Commit records the topology change; solving runs the required rebuild. The initial
sketch also proposed post-commit undo/redo. The final revision removed those methods;
abort/pop covers cancellation before commit.

The governing contract is that `solve()` observes model edits through the dirty/event
pipeline. Callers express the edit and receive the corresponding solve behavior.

## 2. Initial API contract sketch

The following records the initial facade proposal. The final decisions in section 5
and the current usage guide govern implementation details and retained methods.

### 2.1 PowerGrid facade

```python
class PowerGrid:
    def __init__(self, case_path: str | None = None, *,
                 f_hz: float = 50.0, sn_mva: float = 100.0): ...
    @classmethod
    def from_pandapower(cls, net) -> "PowerGrid": ...

    def bus(self, id: int) -> Bus: ...
    def load(self, bus: int | None = None, name: str | None = None) -> Load: ...
    def gen(self, bus: int | None = None, name: str | None = None) -> Gen: ...
    def line(self, from_bus: int, to_bus: int) -> Line: ...
    def loads(self, bus: int | None = None) -> list[Load]: ...

    def edit(self) -> GridEditor: ...
    def solve(self, *, tol: float = 1e-8, max_it: int = 10) -> SolveReport: ...

    @property
    def res_bus(self) -> pd.DataFrame: ...
    @property
    def res_line(self) -> pd.DataFrame: ...
    @property
    def v(self) -> np.ndarray: ...  # Complex voltage in physical bus order.
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def n_bus(self) -> int: ...
    @property
    def n_line(self) -> int: ...
    def describe(self) -> pd.DataFrame: ...
```

The original sketch proposed removing public initialization/loading/reset helpers,
folding them into construction and event handling, with temporary deprecated aliases.
Decision D6 later selected a clean tutorial/API transition. Current retained methods
are documented in the public guide.

### 2.2 GridEditor: unit of work

```python
class GridEditor:
    def add_bus(self, vn_kv, *, name=None, vm_min=0.9, vm_max=1.1, zone="") -> tuple[int, Bus]: ...
    def add_line(self, from_bus, to_bus, length_km, *, std_type=None, **kwargs) -> Line: ...
    def add_trafo(self, hv_bus, lv_bus, *, sn_mva, **kwargs) -> Trafo: ...
    def add_load(self, bus, p_mw, q_mvar, *, name=None) -> Load: ...
    def add_gen(self, bus, p_mw, vm_pu=1.0, **kwargs) -> Gen: ...
    def add_ext_grid(self, bus, vm_pu=1.0, va_degree=0.0) -> ExtGrid: ...
    def add_shunt(self, bus, q_mvar, *, p_mw=0.0, **kwargs) -> Shunt: ...
    def remove(self, element) -> None: ...
```

A successful context exit commits the queued operations and publishes the merged
dirty information. An exceptional exit aborts the transaction. This turns the deferred
buffer into a semantic boundary as well as a component-insertion optimization.

### 2.3 Element proxies: live views

```python
class Load:
    bus: int                  # Read-only association.
    p_mw: float               # Injection edit.
    q_mvar: float
    in_service: bool          # Topology edit.
    name: str

class Gen:
    bus: int
    p_mw: float               # Injection edit.
    vm_pu: float              # Voltage-setpoint edit.
    in_service: bool          # Topology edit.

class Bus:
    id: int
    vn_kv: float              # Admittance-related edit.
    vm_pu: float              # Result projection after solving.
    va_degree: float
    loads: list[Load]
    gens: list[Gen]

class Line:
    from_bus: int
    to_bus: int
    in_service: bool
    # Electrical parameter writes use the admittance update category.
    # Result properties include powers, current, and loading.
```

A proxy holds `(Entity, Py<PowerGrid>)`; property reads query ECS data and writes use
the mutation boundary. This keeps model facts in the World and gives every edit an
explicit route to its numerical dependencies.

### 2.4 SolveReport

```python
class SolveReport:
    converged: bool
    iterations: int
    runtime_ms: float
    rebuild: str
    def __bool__(self): return self.converged
```

The initial proposal distinguished none/injection/voltage/admittance/full rebuilds.
The Phase 1 implementation initially reported full/incremental. Divergence returns a
solve report; pre-solve validation errors carry readable exception messages.

## 3. Revised Rust architecture

The second June 9 revision removed concrete `GridCmd` objects. Immediate edits already
execute under the GIL and deferred edits already live in the Harvard instruction queue.
The gateway therefore remains a module boundary with concrete functions and messages.

```text
Python facade, element proxies, and GridEditor
    |
powerflow::mutation setters
    - update Target* model components synchronously for read-your-writes
    - own the sign convention and filter no-op edits
    - publish ParamDiff instructions
    |
ParamDiff::Injection { bus, dp, dq } / VoltageMag { bus, vm }
    |
consume_param_diffs: ordinary scheduled MessageReader + Query system
    - update SBusInjPu / VBusPu
    - publish SBusChangeEvent / VoltageChangeEvent
    |
structure_update: event-driven refresh of materialized numerical data
    |
PowerFlowMat -> ecs_run_pf -> converged VBusPu writeback for warm starts

Topology edits:
HarvardCommandBuffer commit -> FullRebuildEvent -> PFInit schedule
```

The third revision removed a separate DirtyBuses resource and an exclusive apply
function. Messages carry pending changes; an ordinary scheduled consumer applies
them and emits change events from one place. Full rebuilds clear pending ParamDiff
messages because rebuilding has already consumed their target values.

Incremental signed differences were retained with no-op filtering and periodic full
reconstruction. The design record reports roughly 3e-10 drift over 2000 edits in that
experiment. Earlier Phase 2 notes below describe the preceding recomputation approach.

### Change ticks

Correctness follows explicit messages and dirty categories. Setters know their target
Entity directly; projection refreshes follow their coarse event category. Bevy still
maintains normal ticks, which can provide secondary debugging or validation information.

### Model edits and runtime-state changes

- **Model edit**, such as `load.p_mw = 30`: update Target* components and rederive the numerical state. Target* values are serialized model facts.
- **Runtime-state delta**, such as a time-series or optimization inner-loop injection: update `SBusInjPu` and publish the corresponding coarse event while retaining model Target* values. A possible Python `bus.inject_diff(dp, dq)` would expose this path.

### Migration map

| Earlier component | Intended direction |
| --- | --- |
| Injection arithmetic inside handles | Centralize sign and aggregation rules in the injection pipeline |
| Handwritten `init_pf()` pipeline | One rerunnable PFInit schedule, also used by startup |
| GridBuilder | GridEditor with transactional semantics |
| Python-side bus_to_elements/id_map copies | ECS queries or a system-maintained resource index |
| `python/solver.rs` NewtonSolver | Retain as the separate low-level solver interface |
| Manual events in time-series state | Adopt the shared mutation/coarse-event contract |
| FullRebuildEvent production | Publish from editor commits and topology setters |

## 4. Implementation phases recorded in the design

**Phase 0: contract design.** Freeze the initial section 2 signatures before implementation.

**Phase 1: Python facade.** Implement properties, edit contexts, and result projections
on the existing core. Rewrite the API contract checks, including transaction rollback,
and retain numerical comparisons. The final decisions later refined missing-element
and compatibility behavior.

**Phase 2: unified mutation pipeline, recorded complete on 2026-06-09.** Introduce
`powerflow::mutation`, remove scattered injection arithmetic, and replace tick-driven
correctness with explicit updates. This phase initially used DirtyBuses and bus
reaggregation; the later same-day revisions above selected message consumption and
signed deltas. The phase record reports 31 contract checks, repeated-edit coverage,
and a pandapower numerical comparison around 3e-9.

**Phase 3: unified PFInit schedule, recorded complete on 2026-06-09.** Use one rerunnable
sequence for ingestion, projection cleanup, zeroing, device setup, bus classification,
injection reconstruction, initialization, permutation, and solver reset. Editor commits
and in_service setters emit FullRebuildEvent. Full-model dirty state is distinct from
bus-type/QLim structural changes so that a Q-limit PV-to-PQ transition is retained by
its appropriate update path. `solve()` updates the App and reports LastStructureAction.
The Python initialization helper runs the same schedule synchronously.

**Phase 4: native backport.** Move time-series and other native writers onto the same
mutation/coarse-event contract and remove duplicate Python-side indexes. The intended
acceptance condition is a shared pipeline with passing native and Python checks.

The final design removed post-commit undo/redo, concrete GridCmd objects, and command-log
replay. Transaction cancellation before commit uses the command buffer's abort/pop.

## 5. Final decisions recorded on 2026-06-09

| ID | Decision | Outcome |
| --- | --- | --- |
| D1 | Parameter edits | Property assignment executes the matching immediate mutation |
| D2 | add_* location | GridEditor owns topology construction and fused insertion |
| D3 | Missing element | Return None; callers select their required strictness |
| D4 | in_service=False | Topology dirty category and full rebuild |
| D5 | Results | Read-only res_bus and other DataFrame projections |
| D6 | Legacy transition | Clean API/tutorial transition |

### Naming follow-up

Pandapower IO DTOs already occupy Bus/Line/Load/Gen class names. Phase 1 therefore
used BusHandle/LoadHandle and similar proxy names. A proposed later move of IO DTOs
into a `rustpower.io` submodule would make room for shorter proxy names.

### Phase 1 implementation notes

- Setters initially reused existing incremental/event operations pending migration into the gateway.
- GridEditor.remove defers deletion until commit; abort preserves the object. Removing a bus retains connected devices and marks them out of service; reactivation requires valid bus endpoints. Foreign-grid handles are rejected.
- SolveReport initially distinguished full and incremental rebuilds; finer categories were a subsequent design goal.

For the separate shared-baseline contingency pipeline, see
[Parallel contingency execution](n1_parallel_execution.md) and the
[N−1 usage guide](../N1_USAGE.md).
