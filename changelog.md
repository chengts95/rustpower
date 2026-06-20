# Changelog
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
