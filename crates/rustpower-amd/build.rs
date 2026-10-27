#[cfg(feature = "c_amd")]
use std::env;
#[cfg(feature = "c_amd")]
fn main() {
    if env::var_os("CARGO_FEATURE_C_AMD").is_none() {
        return;
    }
    println!("cargo:rerun-if-changed=vendor/amd");
    println!("cargo:rerun-if-env-changed=SUITESPARSE_DIR");
    let mut c = cc::Build::new();
    c.include("vendor/amd")
        .define("DLONG", None)
        .file("vendor/amd/amd_super2.c")
        .file("vendor/amd/ybus_order.c");
    if let Ok(path) = env::var("SUITESPARSE_DIR") {
        c.include(format!("{path}/include"))
            .include(format!("{path}/include/suitesparse"));
    }
    for path in [
        "/usr/include/suitesparse",
        "/usr/local/include/suitesparse",
        "/opt/homebrew/include",
        "/opt/homebrew/include/suitesparse",
    ] {
        c.include(path);
    }
    c.compile("rustpower_ybus_amd");
}

#[cfg(not(feature = "c_amd"))]
fn main() {}
