# Changelog

## [0.2.0] - 2024-11-21
### Added
- **Major Architectural Overhaul**:  
  Introduced a new Bevy ECS-based application for power flow computation and future practical applications, marking a significant shift from the legacy `PFNetwork`.  
  - **Modular and Extensible Design**: Users can now develop custom plugins to extend functionality. This modular approach enables integration of domain-specific features, such as time-series simulations or real-time power flow monitoring, in future releases. See plugins and Post-Processing Trait for details.

  - **Deprecated Legacy Framework**: The old `PFNetwork` is officially deprecated. Users should usethe ECS-based version for enhanced flexibility and scalability. The old PFNetwork was too simple to meet the demand of solving pratical problems, but it will serve as a demo for the basic Netwon-Raphson power flow solver.

- **Post-Processing Trait**:  
  Added a post-processing trait to demonstrate Rust's compositional design philosophy and ECS's data manipulation capabilities. Users can treat simulation results like a dataframe and implement custom post-processing methods. An example implementation is provided to help users get started.

- **Switch Element Handling (Experimental)**:  
  Added experimental support for modeling switch elements between buses:
  1. **Admittance-based Method**: Represents switches via admittance adjustments.
  2. **Node-merging Method**: Simplifies switch behavior by merging connected nodes.  
  These methods are implemented as optional plugins and are disabled by default.

### Fixed
- Enhanced JSON parsing support for `pandapower` networks, contributed by [@mancioshell](https://github.com/mancioshell).
- Corrected the conversion of shunt elements, which are now treated as admittances rather than PQ injections, ensuring compatibility with `pandapower`'s behavior.

## [0.1.0] - 2024-5-10
### Added
- Established the initial project framework with core functionality for Ybus and Sbus calculations.
- Implemented the Newton-Raphson method for power flow analysis.