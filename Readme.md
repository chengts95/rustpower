# RustPower
[![Crates.io](https://img.shields.io/crates/v/rustpower.svg)](https://crates.io/crates/rustpower)
[![Docs.rs](https://docs.rs/rustpower/badge.svg)](https://docs.rs/rustpower)
[![CI](https://github.com/chengts95/rustpower/actions/workflows/rust.yml/badge.svg)](https://github.com/chengts95/rustpower/actions)
RustPower is a ECS-based power flow calculation library written in Rust, specifically designed for steady-state analysis of electrical power systems. It also provides experimental python binding and has a solver interface for python libs such as pandapower.

## **Key Features**
- High-performance Newton-Raphson power flow computation.
- Modular and extensible design using ECS for future-proof applications.
- Support for `pandapower` format power system data.
- Handles external grid nodes, transformers, and switch elements.
- Includes both  RSparse, Faer and KLU solvers (KLU requires `SUITESPARSE_DIR` on Windows and proper configs on Linux).

---

## **Performance Comparison**

RustPower is designed for extreme performance and memory efficiency. Below is a comparison between established standards and RustPower on the hot loop (caches Ybus and solver data for a invariant topology).

### **Core Solve Time (Newton-Raphson, Hot Loop)**
 
* Tested on Intel i7-10700K@4.7GHz with 32GB DDR4-3000 under Windows 11 with identical iteration counts at flat start inital condition.
  
| Case | Pandapower 3.5.4 (PyPI, Numba) | LightSim2Grid 1.0.0 (PyPI, KLU) | RustPower 0.5.1 (Python, KLU) | **RustPower (Rust Native, KLU)** |
| --- | --- | --- | --- | --- |
| **IEEE 39** | 15.1 ms | 0.035 ms | 0.044 ms | **0.023 ms** |
| **IEEE 118** | 17.1 ms | 0.095 ms | 0.080 ms | **0.059 ms** |
| **PEGASE 9241** | 244.1 ms | 22.9 ms | 21.0 ms | **20.0 ms** |

  

  *  On Intel Core Ultra 7 288V@5.1GHz, 32GB LPDDR5X-8533 under CachyOS / Linux 7.x.
  
| Case | Pandapower 3.5.4 (PyPI, Numba) | LightSim2Grid 1.0.0 (PyPI, KLU) | RustPower 0.5.1 (Python, KLU) | **RustPower (Rust Native, KLU)** |
| --- | --- | --- | --- | --- |
| **IEEE 39** | 4.36 ms | 0.017 ms | 0.021 ms | **0.014 ms** |
| **IEEE 118** | 5.18 ms | 0.051 ms | 0.040 ms | **0.031 ms** | 
| **PEGASE 9241** | 147.19 ms | 13.67 ms | 11.53 ms | **11.36 ms** |


*Note: Python columns reflect end-to-end execution within the Python runtime using official package builds.*

RustPower achieves native C++-grade performance with sub-millisecond execution on IEEE benchmark grids, scaling to solve the 9241-bus PEGASE system in just 11.3 ms on a modern laptop computer.

### **Key Advantages**
 

* **Low Memory Footprint**: For the 9,241-bus PEGASE system, RustPower peaks at only ~34 MB of RAM (a **15× reduction** compared to the 500+ MB footprint of pandapower frameworks). This allows thousands of parallel grid simulations (e.g., N-1 screening, Monte Carlo, RL environments) to run concurrently on standard hardware or resource-constrained cloud containers without thrashing memory bandwidth.
**Memory Safety**: Leveraging Rust's compile-time ownership and memory safety guarantees, RustPower eliminates segmentation faults, data races, and memory leaks by design. Combined with our ECS-based architecture, the core power flow loop operates with zero heap allocations during iterations.
* **Seamless Ecosystem Interoperability**: Delivers pure-native performance while remaining drop-in compatible with established Python workflows. It provides zero-cost data ingestion directly from standard formats like `pandapower`, eliminating migration friction for existing pipelines.

 

### **Advanced Features**

### **Plugin-Based Architecture**
RustPower leverages the **Bevy Plugin System**, allowing users to extend the solver with custom logic without modifying the core. Current official plugins include:
- **Archive Plugin**: A high-performance state persistence system.
- **QLim Plugin**: Automatically enforces generator reactive power limits by dynamically switching PV buses to PQ during the iteration process.
- **Switch Plugins**: Optional modeling for switch elements:
  - **Type A**: Node-merging method (aggregates nodes for simplified modeling).
  - **Type B**: Admittance-based method (directly processes switch admittance).
- **Time-Series Plugin**: A complex, high-level plugin for handling quasi-static time-series simulations with scheduled events.

### **High-Performance Data Archiving**

RustPower features a unique **Archive System** (based on `bevy_archive`) that enables flexible runtime handling of any ECS structure:
- **Custom Arrow Integration**: To handle complex power system structures that are difficult for standard `serde`, we implemented **custom schema overrides**. This ensures type-safe and efficient data transition.
- **Multi-Format Persistence**: Seamlessly save the entire network state and results into:
  - **Apache Parquet**: For compressed, high-performance binary storage (ideal for large-scale time-series).
  - **CSV**: For easy inspection and interoperability with Excel/Pandas.

### **Time-Series Simulations**
By combining the ECS architecture with the Archive system, RustPower can execute large-scale time-series simulations with minimal overhead. Check the `examples/time_series.rs` for a complete workflow.

---

RustPower is available on [Crates.io](https://crates.io/crates/rustpower). You can add it to your project using:

```bash
cargo add rustpower
```

Or by adding the following to your `Cargo.toml`:

```toml
[dependencies]
rustpower = "0.5.1"
```

---

## **Usage Example**

### **Python API (Recommended for Data Science)**

```python
import rustpower as rp

# Load a network and solve
grid = rp.PowerGrid("cases/pegase9241/data.zip")
report = grid.solve()

if report:
    print(f"Converged in {report.iterations} iterations")
    print(grid.res_bus.head())

# Fast parameter updates
load = grid.load(bus=10)
load.p_mw = 100.0
grid.solve() # Runs an incremental solve (fast!)
```

### **Basic Rust ECS Example**

```rust
use rustpower::{io::pandapower::*, prelude::*};
use ecs::post_processing::PostProcessing; // for print bus results

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

