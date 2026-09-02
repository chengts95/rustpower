extern crate bindgen;

use std::env;
use std::path::PathBuf;

// LDL needs only: ldl + amd (ordering) + suitesparseconfig.
// Our LM augmented matrix is quasi-definite, so plain no-pivot LDL' applies.

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
        println!("cargo:rustc-link-lib=amd_static");
        println!("cargo:rustc-link-lib=ldl_static");
    } else {
        println!("cargo:rustc-link-lib=suitesparseconfig");
        println!("cargo:rustc-link-lib=amd");
        println!("cargo:rustc-link-lib=ldl");
    }
    println!("cargo:rustc-link-lib=vcomp");
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", suitesparse_dir))
        .clang_arg(format!("-I{}/include/suitesparse", suitesparse_dir))
        .blocklist_item("FP_NORMAL")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_NAN")
        .derive_default(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

#[cfg(all(target_os = "windows", target_env = "gnu"))]
fn main() {
    println!("cargo:rustc-link-search=C:/Program Files (x86)/SuiteSparse/lib");

    let is_static = env::var("CARGO_FEATURE_STATIC").is_ok();
    let link_type = if is_static { "static=" } else { "" };

    println!("cargo:rustc-link-lib={}ldl", link_type);
    println!("cargo:rustc-link-lib={}amd", link_type);
    println!("cargo:rustc-link-lib={}suitesparseconfig", link_type);

    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .derive_default(true)
        .blocklist_item("FP_NORMAL")
        .blocklist_item("FP_SUBNORMAL")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_NAN")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn main() {
    let is_static = env::var("CARGO_FEATURE_STATIC").is_ok();
    let link_type = if is_static { "static=" } else { "" };

    println!("cargo:rustc-link-lib={}ldl", link_type);
    println!("cargo:rustc-link-lib={}amd", link_type);
    println!("cargo:rustc-link-lib={}suitesparseconfig", link_type);

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-search=/usr/local/lib");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-search=/usr/local/lib");
        println!("cargo:rustc-link-search=/opt/homebrew/lib");
    }

    println!("cargo:rerun-if-changed=wrapper.h");

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
        builder = builder
            .clang_arg("-I/usr/include/suitesparse")
            .clang_arg("-I/usr/local/include/suitesparse");
    } else if cfg!(target_os = "macos") {
        builder = builder
            .clang_arg("-I/usr/local/include")
            .clang_arg("-I/opt/homebrew/include");
    }

    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
