use std::env;
use bevy_archive::prelude::*;
use rustpower::prelude::ecs::network::{DataOps, PowerFlow, PowerGrid};
use rustpower::prelude::ecs::post_processing::PostProcessing;
use rustpower::io::archive::aurora_format::ArchiveSnapshotRes;
use rustpower::io::pandapower::load_csv_zip;
use rustpower::prelude::PPNetwork; 

fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let folder = format!("{}/cases/IEEE118", dir);
    let zip_file = folder.to_owned() + "/data.zip";
    
    // 1. Load CSV
    let net = load_csv_zip(&zip_file).unwrap();

    // 2. Initialize App
    let mut pf_net = PowerGrid::default(); 

    pf_net.world_mut().insert_resource(PPNetwork(net));
     
    pf_net.init_pf_net();
    pf_net.run_pf();
    // 4. Dump in Aurora format
    let output_dir = format!("{}/docs/ieee118_dump", dir);
    if !std::path::Path::new(&output_dir).exists() {
        std::fs::create_dir_all(&output_dir).unwrap();
    }

    pf_net.world_mut().resource_scope::<ArchiveSnapshotRes, _>(|world, registry| {
        let manifest = save_world_manifest(world, &registry.0.case_file_reg).unwrap();
        let manifest_path = format!("{}/manifest.toml", output_dir);
        let file = std::fs::File::create(manifest_path).unwrap();
        serde_json::to_writer_pretty(file, &manifest).unwrap();
    });

    println!("IEEE 118 Aurora dump saved to docs/ieee118_dump/manifest.toml");
    pf_net.post_process();
    pf_net.print_res_line();
}
