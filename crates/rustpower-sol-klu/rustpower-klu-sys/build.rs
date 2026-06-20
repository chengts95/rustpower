extern crate bindgen;

use std::env;
use std::path::PathBuf;
#[cfg(all(target_os = "windows", target_env = "msvc"))]
fn main() {
    let suitesparse_dir = env::var("SUITESPARSE_DIR").unwrap_or_default();
    if suitesparse_dir.is_empty() {
        panic!("SUITESPARSE_DIR is not found");
    }
    println!("cargo:rustc-link-search={}/lib", suitesparse_dir);

    let is_static = env::var("CARGO_FEATURE_STATIC").is_ok();

    if is_static {
        println!("cargo:rustc-link-lib=suitesparseconfig_static");
        println!("cargo:rustc-link-lib=camd_static");
        println!("cargo:rustc-link-lib=amd_static");
        println!("cargo:rustc-link-lib=btf_static");
        println!("cargo:rustc-link-lib=ccolamd_static");
        println!("cargo:rustc-link-lib=colamd_static");
        println!("cargo:rustc-link-lib=klu_static");
    } else {
        println!("cargo:rustc-link-lib=suitesparseconfig");
        println!("cargo:rustc-link-lib=camd");
        println!("cargo:rustc-link-lib=amd");
        println!("cargo:rustc-link-lib=btf");
        println!("cargo:rustc-link-lib=ccolamd");
        println!("cargo:rustc-link-lib=colamd");
        println!("cargo:rustc-link-lib=klu");
    }
    println!("cargo:rustc-link-lib=vcomp");
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", suitesparse_dir))
        .clang_arg(format!("-I{}/include/suitesparse", suitesparse_dir))
        .blocklist_item("FP_NORMAL")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_NAN")
        .derive_default(true)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

#[cfg(all(target_os = "windows", target_env = "gnu"))]
fn main() {
    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    println!("cargo:rustc-link-search=C:/Program Files (x86)/SuiteSparse/lib");

    let is_static = env::var("CARGO_FEATURE_STATIC").is_ok();
    let link_type = if is_static { "static=" } else { "" };

    // Tell cargo to tell rustc to link the klu
    // library.
    println!("cargo:rustc-link-lib={}klu", link_type);
    println!("cargo:rustc-link-lib={}camd", link_type);
    println!("cargo:rustc-link-lib={}amd", link_type);
    println!("cargo:rustc-link-lib={}btf", link_type);
    println!("cargo:rustc-link-lib={}ccolamd", link_type);
    println!("cargo:rustc-link-lib={}colamd", link_type);
    println!("cargo:rustc-link-lib={}suitesparseconfig", link_type);

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        .derive_default(true)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .blocklist_item("FP_NORMAL")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_NAN")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn main() {
    let is_static = env::var("CARGO_FEATURE_STATIC").is_ok();
    let link_type = if is_static { "static=" } else { "" };

    // Tell cargo to tell rustc to link the klu
    // library.
    println!("cargo:rustc-link-lib={}klu", link_type);
    println!("cargo:rustc-link-lib={}camd", link_type);
    println!("cargo:rustc-link-lib={}amd", link_type);
    println!("cargo:rustc-link-lib={}btf", link_type);
    println!("cargo:rustc-link-lib={}ccolamd", link_type);
    println!("cargo:rustc-link-lib={}colamd", link_type);
    println!("cargo:rustc-link-lib={}suitesparseconfig", link_type);

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=omp");
        println!("cargo:rustc-link-search=/usr/local/lib");
    } else if cfg!(target_os = "macos") {
        // Search Homebrew paths
        println!("cargo:rustc-link-search=/usr/local/lib");
        println!("cargo:rustc-link-search=/opt/homebrew/lib");
    }

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .derive_default(true)
        .blocklist_item("FP_NORMAL")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_NAN")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    if cfg!(target_os = "linux") {
        builder = builder.clang_arg("-I/usr/include/suitesparse")
                         .clang_arg("-I/usr/local/include/suitesparse");
    } else if cfg!(target_os = "macos") {
        builder = builder.clang_arg("-I/usr/local/include")
                         .clang_arg("-I/opt/homebrew/include");
    }

    let bindings = builder.generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
