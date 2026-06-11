"""Type stubs for the rustpower Python API v2.

Contract: docs/design/python_api_v2.md.
Core promise: solve() is always correct — topology changes (editor commits,
in_service toggles) trigger an automatic full rebuild; parameter changes via
element properties take the incremental path with warm starts.
"""
import numpy as np
import pandas as pd
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Element proxies (live views over ECS entities; hold no data themselves)
# ---------------------------------------------------------------------------

class BusHandle:
    """Bus proxy. Result properties are valid after solve()."""
    @property
    def id(self) -> int: ...
    @property
    def vn_kv(self) -> float: ...
    @property
    def vm_pu(self) -> float:
        """Voltage magnitude (p.u.) from the last solve."""
        ...
    @property
    def va_degree(self) -> float: ...
    @property
    def p_mw(self) -> float:
        """Net injected active power (MW) from the last solve."""
        ...
    @property
    def q_mvar(self) -> float: ...
    def set_load(self, p_mw: float, q_mvar: float) -> None:
        """Update all loads attached to this bus."""
        ...
    def set_gen(self, p_mw: float, vm_pu: float) -> None:
        """Update all PV generators attached to this bus (slack excluded)."""
        ...

class LoadHandle:
    """Load proxy. Property assignment = immediate-mode command."""
    @property
    def bus(self) -> int: ...
    p_mw: float
    """Active power consumption (MW). Assignment takes effect at next solve()."""
    q_mvar: float
    in_service: bool
    """Topology-class: assignment triggers a full rebuild at the next solve."""

class GenHandle:
    """PV generator proxy. Property assignment = immediate-mode command."""
    @property
    def bus(self) -> int: ...
    p_mw: float
    """Active power production (MW)."""
    vm_pu: float
    """Voltage magnitude setpoint (p.u.)."""
    in_service: bool

class LineHandle:
    """Line proxy. Flow properties are valid after solve()."""
    @property
    def from_bus(self) -> int: ...
    @property
    def to_bus(self) -> int: ...
    in_service: bool
    """Topology-class: assignment triggers a full rebuild at the next solve."""
    @property
    def p_from_mw(self) -> float: ...
    @property
    def q_from_mvar(self) -> float: ...
    @property
    def i_ka(self) -> float: ...

class TrafoHandle: ...
class ExtGridHandle: ...
class ShuntHandle: ...
class SGenHandle: ...
class SwitchHandle: ...

# ---------------------------------------------------------------------------
# Solve report
# ---------------------------------------------------------------------------

class SolveReport:
    """Returned by PowerGrid.solve(). Truthy iff converged."""
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def runtime_ms(self) -> float: ...
    @property
    def rebuild(self) -> str:
        """Rebuild level this solve triggered: 'full' | 'incremental'."""
        ...
    def __bool__(self) -> bool: ...

# ---------------------------------------------------------------------------
# Transactional editor (Unit of Work)
# ---------------------------------------------------------------------------

class GridEditor:
    """
    All topology mutations go through the editor. Commands are buffered
    (fused insert) and applied once on commit; an exception inside the
    `with` block aborts the transaction and leaves the grid unchanged.

        with grid.edit() as e:
            b, _ = e.add_bus(110.0)
            e.add_line(b0, b, 12.0)
        grid.solve()   # automatic full rebuild
    """
    def __enter__(self) -> "GridEditor": ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...

    def add_bus(self, vn_kv: float, name: Optional[str] = None,
                vm_min: float = 0.9, vm_max: float = 1.1,
                zone: int = 0) -> Tuple[int, BusHandle]: ...
    def add_line(self, from_bus: int, to_bus: int, length_km: float,
                 std_type: Optional[str] = None, r_ohm_per_km: float = 0.1,
                 x_ohm_per_km: float = 0.1, c_nf_per_km: float = 0.0,
                 g_us_per_km: float = 0.0, parallel: int = 1,
                 max_i_ka: float = 0.0, name: Optional[str] = None) -> LineHandle: ...
    def add_load(self, bus: int, p_mw: float, q_mvar: float,
                 name: Optional[str] = None) -> LoadHandle:
        """p_mw > 0 indicates consumption."""
        ...
    def add_gen(self, bus: int, p_mw: float, vm_pu: float = 1.0,
                p_min: float = -1000.0, p_max: float = 1000.0,
                q_min: float = -1000.0, q_max: float = 1000.0,
                name: Optional[str] = None) -> GenHandle: ...
    def add_ext_grid(self, bus: int, vm_pu: float = 1.0, va_degree: float = 0.0,
                     name: Optional[str] = None) -> ExtGridHandle: ...
    def add_trafo(self, hv_bus: int, lv_bus: int, sn_mva: float = 1.0,
                  vn_hv_kv: float = 110.0, vn_lv_kv: float = 10.0,
                  vk_percent: float = 10.0, vkr_percent: float = 0.1,
                  pfe_kw: float = 0.0, i0_percent: float = 0.0,
                  shift_degree: float = 0.0, tap_pos: float = 0.0,
                  tap_neutral: float = 0.0, tap_step_percent: float = 1.25,
                  name: Optional[str] = None) -> TrafoHandle: ...
    def add_shunt(self, bus: int, q_mvar: float, p_mw: float = 0.0,
                  vn_kv: float = 110.0, step: int = 1,
                  name: Optional[str] = None) -> ShuntHandle: ...
    def remove(self, element: Any) -> None:
        """Remove an element by handle. Removing a bus removes attached
        elements. Phase 1: not rolled back by abort."""
        ...
    def commit(self) -> None: ...
    def abort(self) -> None: ...

# ---------------------------------------------------------------------------
# PowerGrid facade
# ---------------------------------------------------------------------------

class PowerGrid:
    """
    Core grid object. Three workflows:

    A. Batch:      PowerGrid("case.zip").solve(); grid.res_bus
    B. Parameter:  load.p_mw = p; grid.solve()        # incremental, warm start
    C. Topology:   with grid.edit() as e: ...; grid.solve()  # auto rebuild
    """
    def __init__(self, case_path: Optional[str] = None, qlim: bool = False,
                 **kwargs: Any):
        """qlim=True enforces generator reactive limits (PV buses are demoted
        to PQ when their Q output saturates, with an outer iteration loop)."""
        ...
    @classmethod
    def from_pandapower(cls, net: Any) -> "PowerGrid":
        """Build from a live pandapower net. The pandapower data model is
        discarded after ingestion."""
        ...
    def load_network(self, net: "Network") -> None:
        """Replace the grid contents: clears all entities, re-ingests, and
        rebuilds. Existing handles become invalid."""
        ...

    # -- element access (query-backed; returns None when not found) --------
    def bus(self, id: int) -> Optional[BusHandle]: ...
    def load(self, bus: Optional[int] = None,
             name: Optional[str] = None) -> Optional[LoadHandle]: ...
    def loads(self, bus: Optional[int] = None) -> List[LoadHandle]: ...
    def gen(self, bus: Optional[int] = None,
            name: Optional[str] = None) -> Optional[GenHandle]: ...
    def line(self, from_bus: int, to_bus: int) -> Optional[LineHandle]: ...

    # -- mutation gateway ---------------------------------------------------
    def edit(self) -> GridEditor: ...
    def set_base(self, f_hz: float = 50.0, sn_mva: float = 100.0) -> None: ...

    # -- solve ---------------------------------------------------------------
    def solve(self, v_init: Optional[np.ndarray] = None) -> SolveReport:
        """Run the power flow. Raises RuntimeError only on validation errors
        (empty grid, no slack); divergence is reported via a falsy report.
        v_init: Optional initial voltage guess (p.u. complex) for all buses."""
        ...

    # -- results -------------------------------------------------------------
    @property
    def res_bus(self) -> pd.DataFrame: ...
    @property
    def res_line(self) -> pd.DataFrame: ...
    @property
    def v(self) -> np.ndarray:
        """Complex bus voltages (p.u.). Original bus order."""
        ...
    @v.setter
    def v(self, value: np.ndarray) -> None:
        """Override the initial voltage guess (vinit) for all buses."""
        ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...

    # -- overview -------------------------------------------------------------
    @property
    def n_bus(self) -> int: ...
    @property
    def n_line(self) -> int: ...
    def describe(self) -> pd.DataFrame: ...
    def display_case_buses(self) -> pd.DataFrame: ...
    def display_case_loads(self) -> pd.DataFrame: ...

# ---------------------------------------------------------------------------
# IO (DTOs at the ingestion boundary; will move to rustpower.io in Phase 2)
# ---------------------------------------------------------------------------

class Network: ...

def load_csv_zip(path: str) -> Network: ...
def version() -> str: ...
def features() -> List[str]: ...
