# RustPower

RustPower is a power flow calculation library written in Rust, designed to perform steady-state analysis on electrical power systems.

## Current Features

- Calculation of nodal admittance matrix (Ybus)
- Nodal power injection vector calculation (Sbus)
- Power flow analysis using Newton-Raphson method
- Supports for pandapower network Json files via Serde (todo:CSV parsing)
- Handles external grid nodes and transformer elements
- RSparse and KLU solver (preferred but optional )

## Acknowledgements

This project draws inspiration and knowledge from the following libraries:

- [Pandapower](https://github.com/e2nIEE/pandapower): An easy-to-use Python package for power system analysis.
- [PyPower](/https://github.com/rwl/PYPOWER): A Python implementation of the power flow analysis tool MatPower.
- [MatPower](https://matpower.org/): A package of MATLAB-based power system simulation, analysis, and optimization tools.
  
## Comparison with Existing Tools 

PyPower and PandaPower: These tools use traditional ways to build admittance matrices. They perform dynamic slicing in each iteration to extract vectors and matrices for PQ and PV nodes. While this approach is general, it can lead to performance bottlenecks when dealing with large-scale power systems.

This Program: By using some simple linear algebra methods, this program constructs the required vectors and matrices during initialization, avoiding the need for dynamic slicing in each iteration. I believe it will never become necessary to check individual column in the jacobian matrix during its construction. This not only simplifies the code logic but also significantly improves computational efficiency beyond the benefits brought by Rust and KLU.

## Installation

Add this library to your `Cargo.toml` file:

```toml
[dependencies]
RustPower = "0.1.0"
```

## Usage

```Rust
use pf_module::{io::pandapower::Network, prelude::*};
fn main() {
    // Define your power flow network or load pandapower files
    let json = "{...}";
    let net: Network = serde_json::from_str(file_path).unwrap();
    let pf = PFNetwork::from(net);
    let v_init = pf.create_v_init();
    let tol = Some(1e-8);
    let max_it = Some(10);

    let _v = (&pf).run_pf(v_init.clone(), max_it, tol);
}
// Define your power flow network
let mut network = PFNetwork::new();
// Add buses, loads, generators, transformers, etc.

// Run power flow analysis
let v_init = DVector::from_element(network.buses.len(), Complex::new(1.0, 0.0));
let converged_v = network.run_pf(v_init, None, None);

// Access results, perform further analysis, etc.
```

## License

This project is licensed under the MPLv2 License - see the LICENSE file for details.

## Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request.

## Authors

- Tianshi Cheng
## Contact

Feel free to open a issue or PR.
