use std::{env, io::Write};

use bevy_app::App;
use bevy_archive::{
    binary_archive::{WorldArrowSnapshot, WorldBinArchSnapshot},
    prelude::*,
};
use ecs::post_processing::PostProcessing;
use rustpower::{
    io::archive::aurora_format::ArchiveSnapshotRes,
    prelude::
        *
    ,
};

#[macro_export]
macro_rules! timeit {
    ($name:ident, $times:expr, $block:expr) => {{
        use std::time::{Duration, Instant};
        let mut total_duration = Duration::new(0, 0);
        let mut max_duration = Duration::new(0, 0);
        let mut min_duration = Duration::new(u64::MAX, 999_999_999);

        for _ in 0..$times {
            let start_time = Instant::now();
            let _result = $block();
            let end_time = Instant::now();
            let duration = end_time - start_time;

            total_duration += duration;
            if duration > max_duration {
                max_duration = duration;
            }
            if duration < min_duration {
                min_duration = duration;
            }
        }

        let avg_duration = total_duration / $times;
        println!(
            " {} loops, {} - Average: {:?}, Max: {:?}, Min: {:?}",
            $times,
            stringify!($name),
            avg_duration,
            max_duration,
            min_duration
        );
    }};
}
#[allow(dead_code)]
trait ZipRustPowerSnapshotTrait {
    fn to_case_file_zip(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>>;
    fn to_case_file(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>>;
    fn to_sim_states(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>>;

    fn from_case_file(manifest: Vec<u8>) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;
}
/// Provides snapshot interface on [`App`] for saving/loading full simulation state.
impl ZipRustPowerSnapshotTrait for App {
    fn to_case_file_zip(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let reg = self
            .world()
            .get_resource::<ArchiveSnapshotRes>()
            .ok_or("Missing ArchiveSnapshotRes")?;
        let case_reg = &reg.0.case_file_reg;
        let world = self.world();
        let arr = WorldArrowSnapshot::from_world_reg(world, &case_reg)?;
        arr.to_zip(None)
    }
    fn to_case_file(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let reg = self
            .world()
            .get_resource::<ArchiveSnapshotRes>()
            .ok_or("Missing ArchiveSnapshotRes")?;
        let case_reg = &reg.0.case_file_reg;
        let world = self.world();
        let arr = WorldArrowSnapshot::from_world_reg(world, &case_reg)?;
        let bin = WorldBinArchSnapshot::from(arr);
        Ok(bin.to_msgpack()?)
    }

    fn to_sim_states(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let reg = self
            .world()
            .get_resource::<ArchiveSnapshotRes>()
            .ok_or("Missing ArchiveSnapshotRes")?;
        let case_reg = &reg.0.pf_state_reg;
        let world = self.world();
        let arr = WorldArrowSnapshot::from_world_reg(world, &case_reg)?;
        arr.to_zip(None)
    }

    fn from_case_file(manifest: Vec<u8>) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        let mut app = default_app(); 

        let archive = app
            .world()
            .get_resource::<ArchiveSnapshotRes>()
            .ok_or("Missing ArchiveSnapshotRes")?;
        let registry = archive.0.case_file_reg.clone();
        let w = app.world_mut();
        let arrow = WorldArrowSnapshot::from_zip(&manifest)?;
        arrow.to_world_reg(w, &registry)?;
        Ok(app)
    }
}
fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let file = format!("{}/cases/pegase9241/pegase9241_parquet.zip", dir);

    // Initialize the default ECS application with predefined plugins
    let zip = std::fs::read(&file).unwrap();
    let mut pf_net = App::from_case_file(zip).unwrap();
    // Initialize the default ECS application with predefined plugins

    // pf_net.add_plugins(QLimPlugin);
    pf_net
        .world_mut()
        .resource_scope::<ArchiveSnapshotRes, _>(|world, registry| {
            save_world_manifest(world, &registry.0.case_file_reg)
                .unwrap()
                .to_file("test", None)
                .unwrap();
        });

    pf_net.update(); //this will initalize the data for pf in the first run

    // Extract and validate the results
    let results = pf_net
        .world()
        .get_resource::<PowerFlowResult>()
        .unwrap()
        .clone();

    let data = pf_net.to_case_file_zip().unwrap();
    let mut f = std::fs::File::create("test.zip").unwrap();
    f.write_all(&data).unwrap();

    // Post-process and print the results
    pf_net.post_process();
    pf_net.print_res_bus();
    assert_eq!(results.converged, true);
    println!("ECS APP converged within {} iterations", results.iterations);
    //timeit!(pegase9241, 10, || pf_net.update());
}
