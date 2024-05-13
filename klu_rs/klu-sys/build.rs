extern crate bindgen;

use std::env;
use std::path::PathBuf;
#[cfg(all(target_os = "windows", target_env = "msvc"))]
fn main() {
   
    let suitesparse_dir = env::var("SUITESPARSE_DIR")
        .unwrap_or(String::from(""));
    if suitesparse_dir == ""{
        panic!("SUITESPARSE_DIR is not found");
    }
    println!("cargo:rustc-link-search={}/lib", suitesparse_dir);

    // Tell cargo to tell rustc to link the klu
    // library.
    println!("cargo:rustc-link-lib=suitesparseconfig_static");
    println!("cargo:rustc-link-lib=camd_static");
    println!("cargo:rustc-link-lib=amd_static");
    println!("cargo:rustc-link-lib=btf_static");
    println!("cargo:rustc-link-lib=ccolamd_static");
    println!("cargo:rustc-link-lib=colamd_static");
    println!("cargo:rustc-link-lib=klu_static");
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
        .clang_arg(format!("-I{}/suitesparse/include", suitesparse_dir))
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

    // Tell cargo to tell rustc to link the klu
    // library.
    println!("cargo:rustc-link-lib=static=klu");
    println!("cargo:rustc-link-lib=static=camd");
    println!("cargo:rustc-link-lib=static=amd");
    println!("cargo:rustc-link-lib=static=btf");
    println!("cargo:rustc-link-lib=static=ccolamd");
    println!("cargo:rustc-link-lib=static=colamd");
    println!("cargo:rustc-link-lib=static=suitesparseconfig");

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

#[cfg(target_os = "linux")]
fn main() {
    // Tell cargo to tell rustc to link the klu
    // library.
    println!("cargo:rustc-link-lib=static=klu");
    println!("cargo:rustc-link-lib=static=camd");
    println!("cargo:rustc-link-lib=static=amd");
    println!("cargo:rustc-link-lib=static=btf");
    println!("cargo:rustc-link-lib=static=ccolamd");
    println!("cargo:rustc-link-lib=static=colamd");
    println!("cargo:rustc-link-lib=static=suitesparseconfig");
    println!("cargo:rustc-link-lib=iomp5");

    println!("cargo:rustc-link-search=/usr/local/lib");

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
