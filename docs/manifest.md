# ⚡ RustPower Simulation Framework

**RustPower** is a modular power system simulation framework built using the [ECS (Entity-Component-System)](https://bevyengine.org/learn/book/getting-started/ecs/) paradigm. It provides a data-driven architecture for electric power network modeling, simulation, and result tracking.

## 📦 Key Characteristics

- **ECS-based design**  
  RustPower uses Bevy ECS to construct its core architecture. All physical entities (buses, generators, lines, etc.) are modeled as components, and behaviors are implemented as systems and plugins.

- **PandaPower as input source**  
  Supported basic network elements can be converted from pandapower `csv` and `json` files into RustPower's native ECS model. AC power-flow results have been validated on selected IEEE benchmark systems; this conversion does not provide full pandapower feature compatibility.

- **Native ECS data schema (from 0.3.0)**  
  Since version `0.3.0`, RustPower includes native ECS component definitions under `elements/`, allowing full ECS-native modeling without reliance on external Pandapower imports.

- **High-Performance Python Bindings (from 0.5.0, enhanced in 0.5.2)**  
  A transactional Python grid API supports Pandas/NumPy workflows, DCPF warm starts, and matrix caching. Separately, the Python `NewtonSolver` interface provides seamless numerical solver replacement using pandapower's `ppc`/`ppci` matrices and vectors, retaining pandapower's model preparation and surrounding workflow. The solver binding offers zero-copy access to compatible NumPy buffers and zero-allocation result extraction into pre-allocated arrays; the solver still allocates its internal storage and workspaces. See the [integration scope](python_api.md#pandapower-integration-scope).

- **Composable file format via `bevy_archive` (0.5.0)**  
  RustPower introduces a plugin-based snapshot and archive system built on [`bevy_archive`](https://github.com/chengts95/bevy_archive), enabling modular, component-level persistence and state restoration via Apache Arrow / Parquet, TOML, and CSV.

- **Strictly data-driven design philosophy**  
  The framework follows the principles of data-oriented programming. System behavior is not hardcoded but driven entirely by data presence and scheduling. All algorithms are implemented as composable plugins.

## 🔌 Built-in Plugin System

- **Base plugin set (`BasePFPlugin`)**
  - Structure initialization & topological classification
  - Node tagging and CSC matrix builder
  - Newton-Raphson power flow solver
- **DCPF initialization plugin (`DcpfNewtonPfPlugin`)**
  - Calculates DC power flow linear solution to initialize voltage angles, improving convergence on stressed networks.
- **Iwamoto optimal multiplier plugin (`IwamotoPlugin`)**
  - Deceleration step-size multiplier solver for ill-conditioned and heavily loaded power systems.
- **Q-limit adjustment plugin (`QLimPlugin`)**
  - Automatically checks and enforces generator reactive power limits by dynamically switching PV→PQ bus types during iteration.
- **Switch plugins (`SwitchPluginTypeA`, `SwitchPluginTypeB`)**
  - Native bus-bus switching; choose one plugin. Type A merges closed zero-impedance switches exactly. Type B approximates them with high conductance, which may worsen conditioning.
  - Both models use conductance `1 / z_ohm` for closed finite switches. The current finite model is resistive; pandapower's `switch_rx_ratio` option is not implemented.
  - Type A prepares deterministic dense merge mappings when topology changes. Ybus is reduced by mapping its nonzeros directly; injections are summed, and voltage results expand to every original bus. The Newton loop operates only on the reduced arrays.
  - Voltage control priority is Slack, then PV, then PQ; ties choose the first bus in physical matrix order. Reactive limits combine across generators in a merged node. Bus results keep fixed injections on their original buses and assign the remaining power balance to the representative controller; distributed controller sharing is not modeled.
  - Switch state, connection, impedance, and removal edits trigger the shared rebuild pipeline at AC/DC preparation. Bus-type changes reuse the physical Ybus and merge mapping. These mappings and matrix indices are derived state, rebuilt after archive loading.
  - `LineSwitchPlugin` independently supports line-terminal switches; Type A/B include it. Normal line setup builds a closed patch, then switch preparation adjusts affected lines immediately before core Ybus assembly. Any open switch at a terminal disconnects it.
  - `OpenLinePolicy::PreserveCharging` (default) eliminates the floating terminal and retains charging; `DisableBranch` removes the entire branch contribution. Insert the policy resource before adding the plugin, or update it later; policy changes trigger preparation. Repeated rebuilds restore the current normal model before applying terminal state. An independent `OutOfService` marker always wins.
  - Line switches resolve to a line Entity and `LineTerminal`. `SwitchLineRef`, `OpenLineTerminals`, and voltage reconstruction data are transient caches; archives retain original labels and rebuild these caches. Results report zero current at the open switch and reconstruct the disconnected line-side voltage. Fully disconnected or disabled lines use zero results.
  - `TransformerSwitchPlugin` independently handles two-winding transformer terminals; Type A/B include it. An open winding carries zero terminal current while the connected winding retains magnetizing current and core losses. Both-open and out-of-service transformers contribute zero. Three-winding transformer switches remain unsupported.
- **Time series plugin (`TimeSeriesDefaultPlugins`)**
  - Provides scheduled actions, time-stepping, and state recording using ECS resources and systems.
- **Archive plugin (`ArchivePlugin`)**
  - Apache Arrow and Parquet snapshot persistence.

## 💡 Philosophy

RustPower is not just a solver, but a simulation runtime.  
It is designed to be inspectable, schedulable, and state-persistent by default.


---

## 🧪 Examples

RustPower applications are built around an ECS runtime using plugin registration and resource insertion. The following examples demonstrate how to load a power network, execute a power flow analysis, and interact with system results.

---

### 🔰 Basic Power Flow (with default plugins)

This example demonstrates the minimal setup using `default_app()`, which includes the core plugins required for structure tagging, matrix building, and power flow solving.

```rust,ignore
use ecs::post_processing::PostProcessing; // for result printing
use rustpower::{io::pandapower::*, prelude::*};
use std::env;

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let zipfile = format!("{}/cases/pegase9241/data.zip", dir);
    let net = load_csv_zip(&zipfile).unwrap();

    // Initialize the ECS application with plugins
    let mut pf_net = default_app();

    // Register the power network as a resource
    pf_net.world_mut().insert_resource(PPNetwork(net));
    pf_net.update(); // Executes all registered systems

    // Retrieve and validate results
    let results = pf_net.world().get_resource::<PowerFlowResult>().unwrap();
    assert!(results.converged);
    println!("Converged in {} iterations", results.iterations);

    // Post-process and print results
    pf_net.post_process();
    pf_net.print_res_bus();
}
```

**What this does:**

* Loads a PandaPower `.zip` case into the ECS world
* Runs initializing, matrix construction, and Newton-Raphson solver
* Accesses and prints the voltage results

### 🧩 Snapshot-based Workflow with Plugin Extensions

This example demonstrates how to load a simulation snapshot (`case_file`), extend the behavior with optional plugins (e.g., `QLimPlugin`), and run the power flow as a reusable ECS application.

```rust,ignore
use std::env;
use rustpower::{io::archive::aurora_format::ArchiveSnapshotRes, prelude::*};
use bevy_archive::prelude::*;
use ecs::post_processing::PostProcessing;
use ecs::powerflow::qlim::QLimPlugin;
fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap(); // find the case file
    let file = format!("{}/cases/pegase9241/pegase9241.toml", dir); //this is snapshot of a bevy ECS

    // 1. Initialize the ECS app
    let mut app = default_app();

    // 2. (Optional) Extend the app with new functionality
    app.add_plugins(QLimPlugin); // Enables PV→PQ switch based on Q-limit violation

    // 3. Load case file snapshot into ECS world
    let manifest = read_manifest_from_file(&file, None).unwrap();
    app.world_mut().resource_scope::<ArchiveSnapshotRes, _>(|world, archive| {
        load_world_manifest(world, &manifest, &archive.0.case_file_reg).unwrap();
    }); // this is how to load a snapshot, be aware the archive snapshot has seperated archive registry.

    // 4. Run the simulation step
    app.update();

    // 5. Retrieve and check results
    let result = app.world().get_resource::<PowerFlowResult>().unwrap();
    assert!(result.converged);
    println!("Solved in {} iterations", result.iterations);

    // 6. Post-process results
    app.post_process(); // populate result data
    app.print_res_bus();
}
```

### ⚡ DCPF-Initialized Power Flow

For stressed or large networks where flat-start ($V_i = 1.0 \angle 0^\circ$) struggles or takes many iterations, `DcpfNewtonPfPlugin` solves a DC power flow linear system first to obtain realistic initial phase angles:

```rust,ignore
use rustpower::prelude::{
    default_app,
    ecs::dcpf::{DcpfNewtonPfPlugin, DcpfSolverActive},
    PPNetwork, PowerFlowResult,
};

let mut app = default_app();
app.add_plugins(DcpfNewtonPfPlugin);
app.world_mut().insert_resource(DcpfSolverActive);

app.world_mut().insert_resource(PPNetwork(net));
app.update();
```



---

#### 💡 Extending the Simulation with Custom Plugins

RustPower supports injecting additional numerical logic through plugins:

* **`DcpfNewtonPfPlugin`** performs DC power flow angle pre-initialization
* **`IwamotoPlugin`** provides optimal multiplier step-size deceleration
* **`QLimPlugin`** automatically applies generator Q-limit clamping during iteration
* Additional plugins can be written to:

  * Apply constraint-based topology changes
  * Add voltage-dependent load models
  * Insert FACTS device behaviors
  * Perform fault injection or switching events

**Plugin logic is declarative and modular**, and interacts with the ECS world via:

* Custom events (e.g., `NodeTypeChangeEvent`)
* Queries and resources
* Schedule stages (e.g., `AfterSolve`, `PostUpdate`)

You can register your logic using:

```rust,ignore
app.add_systems(Update, your_system.in_set(AfterSolve));
```

Or group it as:

```rust,ignore
#[derive(Default)]
pub struct YourPlugin;
impl Plugin for YourPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, your_system);
    }
}
```

## 🕒 Enabling Time Series Simulation in RustPower

To switch RustPower from **single-shot power flow** to **continuous time-stepped simulation**, follow these three essential steps:

### Add Time Series Plugins

```rust,ignore
pf_net.add_plugins(TimeSeriesDefaultPlugins);
```

This activates:

* Time tracking (`Time`, `DeltaTime`)
* Action scheduler (`ScheduledStaticActions`)
* State recording (only records if you insert `TimeSeriesData` resource) 

All logic is modular — added only if needed.

---

### Set Simulation Time Step

```rust,ignore
pf_net.insert_resource(DeltaTime(15.0 * 60.0)); // 15 minutes
```

Defines how much simulated time advances with each `app.update()` call.
(You control the rhythm of the simulation.)

---

### Spawn Scheduled Events

```rust,ignore
pf_net.world_mut().spawn(ScheduledStaticActions {
    queue: vec![
        ScheduledStaticAction {
            execute_at: 30.0 * 60.0,
            action: ScheduledActionKind::SetTargetPMW { bus: 0, value: 1000.0 },
        },
    ].into(),
});
```

These actions will **automatically execute at specific simulation times**, modifying grid behavior (e.g. generator outputs) as the system evolves.

---

### ✅ Summary: 3-Step Pattern

| Step | What You Do                    | What It Enables                              |
| ---- | ------------------------------ | -------------------------------------------- |
| 1    | Add `TimeSeriesDefaultPlugins` | Enable all time-aware ECS systems            |
| 2    | Set `DeltaTime`                | Control the timestep size                    |
| 3    | Spawn `ScheduledStaticActions` | Inject behavior changes at scheduled moments |

This is the fully data-driven way to simulate time-series power flows in `RustPower`.



## 🗃️ ECS Snapshot & Archive System (BevyArchive)

RustPower uses `bevy_archive` to support **structured snapshots** of the ECS world. This enables:

* ⚡ **State Persistence**
  Save the full simulation world at any time into `.toml`, `.json`, or `.msgpack`.

* 🔁 **World Reconstruction**
  Instantly reload any archived world snapshot for replay, diffing, or continued simulation.

* 🧊 **Component-Precise Export**
  Each **Archetype** (unique component combination) is stored independently — columnar or binary.

* 🎁 **Multi-Format Output**
  Export to **CSV**, **JSON**, or **MsgPack**:

  * **Embed**: all data stored inline in a single file
  * **File**: each archetype stored in an external file, ideal for large systems

---

### Example: Load from Manifest

```rust,ignore
use bevy_archive::prelude::*;
let net = read_manifest_from_file("case.toml", None).unwrap();

pf_net.world_mut().resource_scope::<ArchiveSnapshotRes, _>(|world, registry| {
    load_world_manifest(world, &net, &registry.0.case_file_reg).unwrap();
});
```

---

### Example: Save with Strategy

```rust,ignore
let manifest = save_world_manifest(&world, &registry).unwrap();
manifest.to_file("output.toml", None).unwrap();
```

or with file-separated CSV:

```rust,ignore
let guide = ExportGuidance::file_all(ExportFormat::Csv, "outdir/");
let manifest = save_world_manifest_with_guidance(&world, &registry, &guide).unwrap();
manifest.to_file("output.toml", None).unwrap();
```



## Experimental contingency studies

- [N−1 usage and measurements](N1_USAGE.md): Rust/Python APIs, examples, screening modes, and benchmark records.
- [Parallel execution design](design/n1_parallel_execution.md): bounded task flow, shared data, sparse results, collector systems, and future recovery/heterogeneous execution.
