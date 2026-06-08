import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

class BusHandle:
    """Internal entity handle representing a Bus in the grid."""
    def set_load(self, p_mw: float, q_mvar: float) -> None:
        """Update all loads attached to this bus."""
        ...
    def set_gen(self, p_mw: float, vm_pu: float) -> None:
        """Update all generators attached to this bus."""
        ...

class LineHandle:
    """Internal entity handle representing a Transmission Line."""
    ...

class TrafoHandle:
    """Internal entity handle representing a Transformer."""
    ...

class LoadHandle:
    """Internal entity handle representing a Load device."""
    def set_p(self, value: float) -> None:
        """Set active power consumption (MW)."""
        ...
    def set_q(self, value: float) -> None:
        """Set reactive power consumption (MVar)."""
        ...

class GenHandle:
    """Internal entity handle representing a Generator device."""
    def set_p(self, value: float) -> None:
        """Set active power production (MW)."""
        ...
    def set_vm(self, value: float) -> None:
        """Set voltage magnitude setpoint (p.u.)."""
        ...

class ExtGridHandle:
    """Internal entity handle representing an External Grid (Slack) device."""
    ...

class GridBuilder:
    """
    Deferred grid builder for high-performance batch construction.
    Commands are buffered until commit() is called.
    Supports context management: `with grid.defer() as b:`.
    """
    def __enter__(self) -> 'GridBuilder': ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...
    
    def add_bus(self, vn_kv: float, name: Optional[str] = None, 
                vm_min: float = 0.9, vm_max: float = 1.1, zone: int = 0) -> Tuple[int, BusHandle]:
        """Add a bus to the command buffer."""
        ...

    def add_line(self, from_bus: int, to_bus: int, length_km: float, 
                 std_type: Optional[str] = None, r_ohm_per_km: float = 0.1, 
                 x_ohm_per_km: float = 0.1, c_nf_per_km: float = 0.0, 
                 g_us_per_km: float = 0.0, parallel: int = 1, 
                 max_i_ka: float = 0.0, name: Optional[str] = None) -> LineHandle:
        """Add a transmission line to the command buffer."""
        ...

    def commit(self) -> None:
        """Apply all buffered changes to the PowerGrid."""
        ...

class PowerGrid:
    """
    Core PowerGrid object managing topology, parameters, and simulation orchestration.
    """
    def __init__(self, case_path: Optional[str] = None, _qlim: bool = False, **kwargs: Any):
        """
        Initialize the power grid.
        """
        ...

    def builder(self) -> GridBuilder:
        """Return a GridBuilder for deferred operations."""
        ...

    def defer(self) -> GridBuilder:
        """Alias for builder(), recommended for use with 'with' statements."""
        ...

    def bus(self, id: int) -> Optional[BusHandle]:
        """Retrieve a handle for a specific Bus ID. Returns None if not found."""
        ...

    def load(self, id: int) -> Optional[LoadHandle]:
        """Retrieve a handle for the first load at a specific Bus ID. Returns None if not found."""
        ...

    def set_base(self, f_hz: float = 50.0, sn_mva: float = 100.0) -> None:
        """Set system base frequency (Hz) and base power (MVA)."""
        ...

    def add_bus(self, vn_kv: float, name: Optional[str] = None, 
                vm_min: float = 0.9, vm_max: float = 1.1, zone: int = 0) -> Tuple[int, BusHandle]:
        """Synchronously add a bus to the grid."""
        ...

    def add_line(self, from_bus: int, to_bus: int, length_km: float, 
                 std_type: Optional[str] = None, r_ohm_per_km: float = 0.1, 
                 x_ohm_per_km: float = 0.1, c_nf_per_km: float = 0.0, 
                 g_us_per_km: float = 0.0, parallel: int = 1, 
                 max_i_ka: float = 0.0, name: Optional[str] = None) -> LineHandle:
        """Synchronously add a transmission line."""
        ...

    def add_load(self, bus: int, p_mw: float, q_mvar: float, name: Optional[str] = None) -> LoadHandle:
        """Add a static load. p_mw > 0 indicates consumption."""
        ...

    def add_gen(self, bus: int, p_mw: float, vm_pu: float = 1.0, 
                p_min: float = -1000.0, p_max: float = 1000.0, 
                q_min: float = -1000.0, q_max: float = 1000.0, 
                name: Optional[str] = None) -> GenHandle:
        """Add a generator."""
        ...

    def add_ext_grid(self, bus: int, vm_pu: float = 1.0, va_degree: float = 0.0, 
                    name: Optional[str] = None) -> ExtGridHandle:
        """Add an external grid (Slack)."""
        ...

    def from_pp_net(self, net: Any) -> None:
        """Load grid from a pandapower net object."""
        ...

    def init_pf(self) -> None:
        """
        Initialize/Rebuild the power flow matrices. 
        MUST be called after adding elements or modifying source components (Load, Gen, etc.).
        """
        ...

    def solve(self) -> None:
        """Run power flow calculation."""
        ...

    @property
    def n_bus(self) -> int: ...
    @property
    def n_line(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...

    def display_case_buses(self) -> 'pd.DataFrame': ...
    def display_case_lines(self) -> 'pd.DataFrame': ...
    def display_case_loads(self) -> 'pd.DataFrame': ...
    def display_buses(self) -> 'pd.DataFrame': ...
    def display_lines(self) -> 'pd.DataFrame': ...

    def get_bus_results(self) -> Dict[str, np.ndarray]: ...
    def get_line_results(self) -> Dict[str, np.ndarray]: ...

def version() -> str:
    """Return the RustPower version string."""
    ...
