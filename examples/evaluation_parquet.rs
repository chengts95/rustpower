use std::env;
use std::io::Write;
use bevy_archive::{binary_archive::WorldArrowSnapshot, prelude::*};
use rustpower::{io::archive::aurora_format::{ArchivePlugin, ArchiveSnapshotRes}, prelude::*};
use rustpower::prelude::{PowerGrid, PowerFlow, DataOps};
use rustpower::io::pandapower::load_csv_zip;

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let zip_118 = format!("{}/cases/IEEE118/data.zip", dir);
    
    println!("Loading IEEE 118 data...");
    let net = load_csv_zip(&zip_118).expect("Failed to load 118 data");

    let mut pf_net = PowerGrid::default();
    // pf_net.app_mut().add_plugins(ArchivePlugin); // Already included in DefaultPlugins via PowerGrid::default() -> default_app()
    
    pf_net.world_mut().insert_resource(PPNetwork(net));
    
    println!("Running Power Flow...");
    pf_net.init_pf_net();
    pf_net.run_pf();
    pf_net.post_process();

    println!("Archiving to Parquet...");
    let world = pf_net.world();
    let archive_res = world.get_resource::<ArchiveSnapshotRes>().expect("Missing ArchiveSnapshotRes");
    
    // We want to archive the output results (Vm, Va, P, Q)
    let output_reg = &archive_res.0.output_reg;
    let arrow_snap = WorldArrowSnapshot::from_world_reg(world, output_reg).expect("Failed to create Arrow snapshot");
    
    // Save to a zip file which will contain parquet files
    let zip_data = arrow_snap.to_zip(None).expect("Failed to convert to zip");
    let mut f = std::fs::File::create("ieee118_results_parquet.zip").unwrap();
    f.write_all(&zip_data).unwrap();
    
    println!("Archive saved to ieee118_results_parquet.zip");
    println!("Evaluation: This format allows columnar storage of ECS components. Each archetype/component group becomes a Parquet file, which is ideal for large-scale analysis in Pandas/Polars.");
}
