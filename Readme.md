# RustPower

RustPower is a cutting-edge power flow calculation library written in Rust, specifically designed for steady-state analysis of electrical power systems. With the introduction of **ECS-based architecture** in version 0.2.0, RustPower offers unparalleled modularity and extensibility.

---
## **What's New in 0.3.0**
- **New Solvers**:  
  **faer**: A highly efficient and scalable solver for large-scale power systems.
- **Inital support for native ECS archive files**
- **Initali support for time-series simulations**

## **What's New in 0.2.0**
- **World's First ECS-Based Power Flow Solver**:  
  RustPower now adopts the **Entity-Component-System (ECS)** architecture using Bevy, enabling modular design and extensibility for domain-specific applications such as:
  - Time-series simulations.
  - Real-time monitoring.
  - Custom plugin development.  
  The legacy `PFNetwork` is now deprecated but remains available as a demo for the basic Newton-Raphson solver.

- **Post-Processing Trait**:  
  Added a flexible post-processing trait to manage simulation results, allowing users to handle data as if working with a dataframe. This demonstrates Rust's compositional design philosophy and makes ECS highly effective for handling large datasets.

- **Experimental Switch Handling**:  
  Introduced two optional methods for modeling switch elements:
  1. **Admittance-Based Method**: Adjusts admittance matrices.
  2. **Node-Merging Method**: Merges connected nodes for simplified modeling.  
  These are implemented as plugins and can be enabled as needed.

---

## **Key Features**
- High-performance power flow computation with Newton-Raphson.
- Modular and extensible design using ECS for future-proof applications.
- Support for `pandapower` JSON network files (with experimental CSV support).
- Handles external grid nodes, transformers, and switch elements.
- Includes both RSparse and KLU solvers (KLU requires `SUITESPARSE_DIR` on Windows).

---

## **Performance Comparison**

RustPower achieves industry-leading performance in power flow calculations:
- **IEEE 39-Bus System**:  
  - ~300 microseconds with KLU (3 iterations).  
  - ~500 microseconds with RSparse solver.  
  10x faster than Python/Numba implementations.

- **PEGASE 9241 System**:  
  Demonstrates significant performance advantages over Python-based solutions, even without multi-threading. RustPower is highly optimized for speed and avoids the complexity of C/C++ memory management.  
![Performance Chart 1](imgs/performance_1.png)  
![Performance Chart 2](imgs/performance_2.png)

---

## **Installation**

As `rustpower` is not yet published on [Crates.io](https://crates.io/), you can add it to your project directly from GitHub:

1. Add the following line to your `Cargo.toml`:

   ```toml
   [dependencies]
   rustpower = { git = "https://github.com/chengts95/rustpower", branch = "main" }

---

## **Usage Example**

### **Basic ECS Example**

```rust
use ecs::post_processing::PostProcessing; // for print bus results
use rustpower::{io::pandapower::*, prelude::*};

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let zipfile = format!("{}/cases/pegase9241/data.zip", dir);
    let net = load_csv_zip(&zipfile).unwrap();

    // Initialize the ECS application with plugins
    let mut pf_net = default_app();

    // Register the power network as a resource
    pf_net.world_mut().insert_resource(PPNetwork(net));
    pf_net.update(); // Initializes the data for the first run

    // Retrieve results
    let results = pf_net.world().get_resource::<PowerFlowResult>().unwrap();
    assert!(results.converged);
    println!("Converged in {} iterations", results.iterations);

    // Post-process and print results
    pf_net.post_process();
    pf_net.print_res_bus();
}
```

For more examples, check the `examples` and `cases` folder.

---

## **License**

This project is licensed under the MPLv2 License. See the [LICENSE](LICENSE) file for more details.

---

## **Contributions**

Contributions are welcome! Feel free to open an issue or submit a pull request to help improve the library.

---

## **Authors**
- Tianshi Cheng

---

## **Acknowledgements**

This project draws inspiration from:
- [Pandapower](https://github.com/e2nIEE/pandapower)
- [PyPower](https://github.com/rwl/PYPOWER)
- [MatPower](https://matpower.org)

Special thanks to:  
[T. Cheng, T. Duan, and V. Dinavahi, "ECS-Grid: Data-Oriented Real-Time Simulation Platform for Cyber-Physical Power Systems," IEEE Transactions on Industrial Informatics, vol. 19, no. 11, pp. 11128-11138, 2023.](https://era.library.ualberta.ca/items/5e45c2ff-9b92-41c7-b685-020110b77239)

Although ECS-Grid is a more complex electromagnetic transient (EMT) simulation system, its design philosophy and methodologies greatly influenced the development of this steady-state power flow solver.

