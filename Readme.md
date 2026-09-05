# RustPower
[![Crates.io](https://img.shields.io/crates/v/rustpower.svg)](https://crates.io/crates/rustpower)
[![Docs.rs](https://docs.rs/rustpower/badge.svg)](https://docs.rs/rustpower)
[![CI](https://github.com/chengts95/rustpower/actions/workflows/rust.yml/badge.svg)](https://github.com/chengts95/rustpower/actions)
RustPower is an ECS-based power flow calculation library written in Rust, specifically designed for high-performance steady-state analysis of electrical power systems. It provides a transactional Python binding and direct solver interfaces for Python libraries such as pandapower.

## **Key Features**
- **High-Performance Newton-Raphson Engine**: Optimized single-pass $O(\text{nnz})$ `dSbus_dV` evaluation and branch-free symbolic Jacobian filling (`fill_jacobian_ultimate`).
- **DCPF Solver & Warm-Start Initialization**: Built-in DC power flow calculation and DCPF-initialized Newton-Raphson for accelerated convergence on stressed networks.
- **Iwamoto Optimal Multiplier**: Robust nonlinear solver using optimal deceleration step size $\mu$ to ensure convergence on ill-conditioned or heavily-loaded grids.
- **Standard 2-Port Branch Modeling**: Rigorous $\Pi$-equivalent model for lines and transformers with complex tap ratios and phase shifts.
- **Stateful Jacobian & Solver Caching (`NewtonCache`)**: Reuses symbolic LU factorization and sparsity patterns across consecutive iterations.
- **Modular ECS Architecture**: Built on Bevy ECS for data-oriented composability, custom plugins, and zero heap allocation during hot iteration loops.
- **Seamless Pandapower Interoperability**: Direct zero-copy data ingestion from `pandapower` networks in Python, plus CSV-ZIP and JSON formats.
- **Multiple High-Performance Solvers**: Supports RSparse, Faer, and SuiteSparse KLU backends.
- **Apache Arrow & Parquet Archiving**: High-performance state persistence powered by `bevy_archive 0.5.0`.

---

## **Performance Comparison**

RustPower is designed for extreme performance and memory efficiency. Below is a comparison between established standards and RustPower on the hot loop (caches Ybus and solver data for an invariant topology).

### **Core Solve Time (Newton-Raphson, Hot Loop)**
 
* Tested on Intel i7-10700K@4.7GHz with 32GB DDR4-3000 under Windows 11 with identical iteration counts at flat start initial condition:
  
| Case | Pandapower 3.5.4 (PyPI, Numba) | LightSim2Grid 1.0.0 (PyPI, KLU) | RustPower 0.5.2 (Python, KLU) | **RustPower (Rust Native, KLU)** |
| --- | --- | --- | --- | --- |
| **IEEE 39** | 15.1 ms | 0.035 ms | 0.044 ms | **0.023 ms** |
| **IEEE 118** | 17.1 ms | 0.095 ms | 0.080 ms | **0.059 ms** |
| **PEGASE 9241** | 244.1 ms | 22.9 ms | 21.0 ms | **20.0 ms** |

* On Intel Core Ultra 7 288V@5.1GHz, 32GB LPDDR5X-8533 under CachyOS / Linux 7.x:
  
| Case | Pandapower 3.5.4 (PyPI, Numba) | LightSim2Grid 1.0.0 (PyPI, KLU) | RustPower 0.5.2 (Python, KLU) | **RustPower (Rust Native, KLU)** |
| --- | --- | --- | --- | --- |
| **IEEE 39** | 4.36 ms | 0.017 ms | 0.021 ms | **0.014 ms** |
| **IEEE 118** | 5.18 ms | 0.051 ms | 0.040 ms | **0.031 ms** | 
| **PEGASE 9241** | 147.19 ms | 13.67 ms | 11.53 ms | **11.36 ms** |

*Note: Python columns reflect end-to-end execution within the Python runtime using official package builds.*

RustPower achieves native C++-grade performance with sub-millisecond execution on IEEE benchmark grids, scaling to solve the 9241-bus PEGASE system in just 11.3 ms on a modern laptop.

### **Key Advantages**

* **Low Memory Footprint**: For the 9,241-bus PEGASE system, RustPower peaks at only ~34 MB of RAM (a **15× reduction** compared to the 500+ MB footprint of pandapower frameworks). This allows thousands of parallel grid simulations (e.g., N-1 screening, Monte Carlo, RL environments) to run concurrently on standard hardware or resource-constrained cloud containers without thrashing memory bandwidth.
* **Memory Safety & Zero Allocation**: Leveraging Rust's compile-time ownership guarantees, RustPower eliminates segmentation faults, data races, and memory leaks by design. Combined with our ECS architecture, the core power flow iteration loop operates with zero heap allocations.
* **Seamless Ecosystem Interoperability**: Delivers pure-native performance while remaining drop-in compatible with established Python workflows. It provides zero-cost data ingestion directly from standard formats like `pandapower`, eliminating migration friction for existing pipelines.

---

### **Plugin-Based Architecture**
RustPower leverages the **Bevy Plugin System**, allowing users to extend the solver with custom logic without modifying the core:
- **`BasePFPlugin`**: Core power flow pipeline (structure init, matrix builder, Newton-Raphson).
- **`DcpfNewtonPfPlugin`**: Solves DC power flow to initialize bus voltage angles prior to AC power flow.
- **`IwamotoPlugin`**: Implements Iwamoto's optimal step-size multiplier method for robust convergence under extreme load conditions.
- **`QLimPlugin`**: Automatically enforces generator reactive power limits by dynamically switching PV buses to PQ during the iteration process.
- **`SwitchPluginTypeA` / `SwitchPluginTypeB`**: Optional modeling for switch elements (node-merging or admittance-based).
- **`TimeSeriesDefaultPlugins`**: Quasi-static time-series simulations with scheduled events.
- **`ArchivePlugin`**: High-performance ECS state persistence system.

### **High-Performance Data Archiving**
RustPower features a unique **Archive System** (based on `bevy_archive 0.5.0`):
- **Custom Arrow & Parquet Integration**: Columnar snapshot storage for high-performance time-series data and grid state persistence.
- **Multi-Format Persistence**: Save and restore full network state into Apache Parquet (`.parquet` / `.zip`), TOML, or CSV.

---

## **Installation**

### Rust Crate
Add RustPower to your `Cargo.toml`:

```toml
[dependencies]
rustpower = "0.5.2"
```

Available features:
- `rsparse` (default): Fast native pure-Rust sparse solver.
- `arrow` (default): Apache Arrow and Parquet snapshot support via `bevy_archive`.
- `faer`: High-performance portable linear algebra solver.
- `klu`: SuiteSparse KLU direct sparse solver (statically or dynamically linked).
- `python`: Python C-extension module bindings (PyO3).

### Python Package
```bash
pip install rustpower
```

---

## **Usage Examples**

### **Python API**

```python
import rustpower as rp
import pandapower.networks as nw

# 1. Load network from pandapower or a CSV-ZIP case file
net = nw.case118()
grid = rp.PowerGrid.from_pandapower(net)
# Or: grid = rp.PowerGrid("cases/IEEE118/data.zip")

# 2. Configure solver options (optional)
grid.enable_cache(True)       # Enable Jacobian / LU factorization caching
grid.enable_dcpf_init(True)   # Enable DCPF angle initialization for faster convergence
# grid.enable_iwamoto(True)   # Use Iwamoto optimal multiplier for ill-conditioned cases

# 3. Solve power flow
report = grid.solve(tol=1e-8, max_iter=20)

if report.converged:
    print(f"Converged in {report.iterations} iterations, time: {report.runtime_ms:.2f} ms")
    print(grid.res_bus.head())
    print(grid.res_line.head())
    print(grid.res_trafo.head())

# 4. Fast parameter updates (warm-start incremental path)
load = grid.load(bus=10)
if load:
    load.p_mw = 100.0
grid.solve()  # Runs incremental solve reusing cached matrix topology

# 5. Transactional topology editing
with grid.edit() as e:
    new_bus_id, _ = e.add_bus(vn_kv=110.0, name="Substation_C")
    e.add_line(from_bus=10, to_bus=new_bus_id, length_km=15.0)
    e.add_load(bus=new_bus_id, p_mw=25.0, q_mvar=5.0)
grid.solve()  # Automatically triggers topology rebuild
```

### **Rust ECS Examples**

#### 1. Basic Power Flow
Run with: `cargo run --example basic_powerflow`

```rust
use std::env;
use rustpower::io::pandapower::load_csv_zip;
use rustpower::prelude::*;
use ecs::post_processing::PostProcessing;

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let zipfile = format!("{}/cases/IEEE118/data.zip", dir);
    let net = load_csv_zip(&zipfile).expect("Failed to load case zip");

    // Initialize the ECS application with default plugins
    let mut pf_net = default_app();
    pf_net.world_mut().insert_resource(PPNetwork(net));
    pf_net.update(); // Initializes and solves power flow

    // Retrieve results
    let results = pf_net.world().get_resource::<PowerFlowResult>().unwrap();
    assert!(results.converged);
    println!("Converged in {} iterations", results.iterations);

    // Post-process and print results
    pf_net.post_process();
    pf_net.print_res_bus();
}
```

#### 2. DCPF-Initialized Power Flow
Run with: `cargo run --example dcpf_example`

```rust
use rustpower::prelude::{
    default_app,
    ecs::dcpf::{DcpfNewtonPfPlugin, DcpfSolverActive},
    PPNetwork, PowerFlowResult,
};

let mut pf_net = default_app();
pf_net.add_plugins(DcpfNewtonPfPlugin);
pf_net.world_mut().insert_resource(DcpfSolverActive);
pf_net.world_mut().insert_resource(PPNetwork(net));
pf_net.update();
```

#### 3. Iwamoto Optimal Multiplier Solver
Run with: `cargo run --example iwamoto_example`

```rust
use rustpower::prelude::{
    default_app,
    CustomSolverActive, IwamotoPlugin, PPNetwork, PowerFlowResult,
};

let mut pf_net = default_app();
pf_net.add_plugins(IwamotoPlugin);
pf_net.world_mut().insert_resource(CustomSolverActive);
pf_net.world_mut().insert_resource(PPNetwork(net));
pf_net.update();
```

Additional examples available in the `examples/` directory:
- `archive_example.rs`: TOML-based state restoration.
- `arrow_archive_example.rs`: Apache Arrow / Parquet snapshot serialization.

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

