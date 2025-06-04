# ⚡ RustPower Simulation Framework

**RustPower** is a modular power system simulation framework built using the [ECS (Entity-Component-System)](https://bevyengine.org/learn/book/getting-started/ecs/) paradigm. It provides a data-driven architecture for electric power network modeling, simulation, and result tracking.

## 📦 Key Characteristics

- **ECS-based design**  
  RustPower uses Bevy ECS to construct its core architecture. All physical entities (buses, generators, lines, etc.) are modeled as components, and behaviors are implemented as systems and plugins.

- **PandaPower as input source**  
  Power network data can be imported from `csv` and `json` files, providing compatibility with a wide range of real-world test cases.

- **Native ECS data schema (from 0.3.0)**  
  Since version `0.3.0`, RustPower includes native ECS component definitions under `elements/`, allowing full ECS-native modeling without reliance on external Pandapower imports.

- **Composable file format via `bevy_archive`**  
  RustPower introduces a plugin-based snapshot and archive system built on [`bevy_archive`](https://github.com/chengts95/bevy_archive), enabling modular, component-level persistence and state restoration.

- **Strictly data-driven design philosophy**  
  The framework follows the principles of data-oriented programming. System behavior is not hardcoded but driven entirely by data presence and scheduling. All algorithms are implemented as composable plugins.

## 🔌 Built-in Plugin System

- **Base plugin set**
  - Power flow solver
  - Structure initialization
  - Node tagging and matrix builder
- **Optional plugin**
  - Q-limit adjustment (`QLimPlugin`) that dynamically changes PV→PQ bus types during iteration
- **Time series plugin**
  - Provides scheduled actions, time-stepping, and state recording using ECS resources and systems

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

---

#### 💡 Extending the Simulation with Custom Plugins

RustPower supports injecting additional numerical logic through plugins:

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




