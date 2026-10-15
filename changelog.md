# Changelog

## [0.5.3]  

### Changed

- Python networks with switches need `switch_model="merge"` (Type A) or `"admittance"` (Type B). The grid remembers your choice. Unsupported switch types raise an error.
- Add optional `LineSwitchPlugin` for open line switches. Both Type A and Type B include it.
- Use Entity references for connections and cached `usize` indices for matrix access. Bus IDs stay separate from matrix indices.
- Store bus zones as strings in Rust and Python.
- Add `grid.validate()` to list invalid connections, including disabled elements. Run it when you want to check; saving stays unchanged. Reconnecting to a replacement bus after loading may be intentional.
- **Removal API change:** `remove()` queues a deletion. When the edit is committed, the bus is deleted and its connected elements are kept but marked out of service. Until then, nothing is deleted or disabled. Canceling the edit cancels both changes. Reconnect those elements before enabling them again.
- Use V4 Jacobian assembly with fewer lookup tables and searches.
- Enable Newton caching by default. Use `PowerGrid.clear_cache()` to clear it.
- **Rust API change:** `NewtonCache.j_pattern` now uses `JacobianCache`. Use `j_row_indices.len()` for the nonzero count.

### Fixed

- Fix pandapower bus IDs that have gaps or do not start at zero (#15). Keep the original IDs as `PPBusID`.
- Handle open line switches at either end. Keep charging at the connected end (#14).
- Accept text, numbers, and missing values in pandapower bus zones (#13).
- Fix bus-bus switch merging, finite switch impedance, and switch updates. Return results for every original bus.
- Reject invalid connections only for elements that are in service.
- Check network connections before replacing the current grid. A failed check keeps the old grid and its handles.
- Fix bus and branch state after grid replacement or archive loading. Fix DC solves with one unknown.
- Honor `qlim=True`. Combine generator Q limits at each bus, avoid counting clamped Q twice, and keep limited buses as PQ until `init_pf()`.
- Fix AC/DC cache updates and voltage starts. Q-limit retries keep the previous voltage; failed solves return failure without panicking.
- Fix stale results and voltage ordering in time-series runs.

### Experimental

- Add `experimental_fused_scalc` to calculate power and the Jacobian in one Ybus pass. Off by default.
- Add sparse matrix helpers and regression tests for bus IDs, switches, caches, Q limits, and AC/DC solves.

## [0.5.2] - Fix and Improvement 
- Fix many pandapower model processing bugs, now is accurate in most network cases.
- Add dcpf to solve some types of networks.
- Use standard 2-port network patch for branches.
- Refactor the postprocessing and remove obselete codes and structures.
- Add native newtonpf solver without ECS networks to further save memories.

## [0.5.1] - Fix and Improvement 
- Fix transformer model error caused by oversight in complex number division, still not same to pandapower's transformer handling but explainable.
- Fix python solver api docstring and add interface for iterations, tols, max_iter. 
- Avoid useless memory consumptions in python solver api, use numpy array in setup context.
- Further optimize the newton_pf and use norm Inf for tolerance check.
- Add an experimental jacobian cache api to store jacobian and Sbus workspace. 
- Suppress klu wrapper bindgen warnings and add search path to linux `/usr/lib`.
- New test scripts to benchmark other opensource pf libs.

## [0.5.0] - Pre-release
- Fix KLU wrapper small memory leak.
- Add intial python wrapper.
- Upgrade to Bevy 0.19.0.
- Upgrade bevy_archive to 0.4.0.
- **Newton-Raphson performance optimization**: Three variants benchmarked on PEGASE9241 (9241-bus):
  - **`fill_jacobian_ultimate`**: Directly fills the real-valued Jacobian matrix from `Ybus` + `V` + `Vnorm` + `Ibus` using a pre-computed sparsity pattern (`JacobianPattern`), bypassing the complex dS/dVm, dS/dVa CSC construction and slice/stack assembly entirely.
  - **Element-wise `dSbus_dV`**: Replaces the original 5× SpGEMM path with a single-pass O(nnz) traversal, avoiding expensive sparse matrix multiplications.
  - Combined, the optimized Newton-Raphson (`newton_pf`, using `fill_jacobian_ultimate`) achieves  **1.82×** on PEGASE9241.
  - **Benchmark results** (10 loops, PEGASE9241, release mode,intel 10700K):
    | Variant | rsparse | KLU |
    |---------|---------|-----|
    | no opt (original CSC path) | 216.92 ms | 184.72 ms |
    | half opt (element-wise dSbus_dV) | 175.50 ms (1.24×) | 139.11 ms (1.33×) |
    | opt (fill_jacobian_ultimate) | 152.34 ms (1.42×) | 119.19 ms (1.55×) |

## [0.4.1] - 2026-5-17
- Backport 0.5.0 new jacobian matrix formation, around 20-40% speed-up for a round of netwon iteration.
- Use bevy_archive 0.3.0 for archive case file.
  
## [0.4.0] - 2025-11-20
- Update to `bevy` 0.17.x. 
- Enable Arrow-based Parquet snapshot and binary file format (`bevy_archive` 0.2.x).

## [0.3.3] - 2025-9-19
- Fix solvers to reset symbolic lu factorizations after structure changes.

## [0.3.2] - 2025-7-31
- Remove the debug print in transformer module.

## [0.3.1] - 2025-6-27
- Allow asymmetrical transfomer `Ybus` injections.
- Temporal fix for `ecs_net` example.


## [0.3.0] - 2025-6-4
- Refactor power grid elements as ECS components and removed the legacy OOP `PFNetwork`.
- Refactor solvers and add `faer` solver as a new option.
- Modularized solver interfaces to allow flexible backend switching.
- Initial support for time-series simulations.
- Initial support for native ECS archive files.
  
## [0.2.0] - 2024-11-21
### Added
- **World’s First ECS-Based Power Flow Solver**:  
  Introduced the first-ever steady-state power system analysis program using the **Bevy ECS** architecture. This groundbreaking update shifts from the legacy `PFNetwork` to a modular, extensible design, paving the future way for advanced applications such as:
  - **Time-Series Simulations**
  - **Stochastic Power Flow**
  - **Optimal Power Flow**
  - **Custom Plugins for Domain-Specific Needs**  

  **Deprecation Notice**: The old `PFNetwork` is now deprecated. While it remains available as a demo for the Newton-Raphson power flow solver, it is no longer suitable for practical problem-solving. Users are encouraged to migrate to the ECS-based version for better scalability and flexibility.

- **Post-Processing Trait**:  
  Added a post-processing trait to demonstrate Rust's compositional design philosophy and how simulation results can be handled within the ECS framework, similar to working with dataframes. Users can implement their own post-processing methods, with provided examples serving as a starting point.

- **Switch Element Handling (Experimental)**:  
  Introduced experimental support for handling switch elements between buses, offering two optional methods:
  1. **Admittance-Based Method**: Models switches via admittance adjustments.
  2. **Node-Merging Method**: Simplifies switches by merging connected nodes.  
  These methods are implemented as optional plugins and are disabled by default.

### Fixed
- **Improved JSON Parsing for `pandapower`**:  
  Enhanced compatibility with `pandapower` networks, thanks to contributions from [@mancioshell](https://github.com/mancioshell).
- **Corrected Shunt Element Behavior**:  
  Shunt elements are now treated as admittances rather than PQ injections, ensuring consistency with `pandapower`’s implementation.

---

## [0.1.0] - 2024-05-10
### Added
- **Initial Project Release**:  
  - Established the foundational framework for Ybus and Sbus calculations.
  - Implemented the Newton-Raphson method for power flow analysis.
