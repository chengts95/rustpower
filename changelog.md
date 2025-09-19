# Changelog

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
