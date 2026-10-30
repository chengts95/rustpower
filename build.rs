use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    for file in [
        "native/acpf_cuda/configure.cmd",
        "native/acpf_cuda/build_dll.cmd",
        "native/acpf_cuda/include/acpf_cuda.h",
        "native/acpf_cuda/src/executor.cu",
        "native/acpf_cuda/src/newton_kernels.cu",
        "native/acpf_cuda/src/branch_kernels.cu",
    ] {
        println!("cargo:rerun-if-changed={file}");
    }
    for variable in ["CUDA_PATH", "CUDSS_DIR", "RUSTPOWER_CUDA_ARCH"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    // The ordinary CPU build must remain independent of every native CUDA tool.
    if env::var_os("CARGO_FEATURE_EXPERIMENTAL_N1_CUDA").is_none() {
        return;
    }
    if !cfg!(windows) {
        panic!("experimental_n1_cuda currently requires Windows, CUDA and cuDSS");
    }

    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let output = out_dir.join("acpf-cuda");
    let script = root.join("native/acpf_cuda/build_dll.cmd");
    let status = Command::new("cmd")
        .current_dir(&root)
        .arg("/d")
        .arg("/c")
        .arg(&script)
        .arg(&output)
        .status()
        .expect("failed to launch the acpf_cuda DLL build");
    assert!(status.success(), "acpf_cuda DLL build failed: {status}");

    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=dylib=acpf_cuda");
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("unexpected Cargo OUT_DIR layout");
    // acpf_cuda.dll is linked through its import library. Its CUDA/cuDSS runtime
    // dependencies were collected by build_dll.cmd and travel with our binary.
    for entry in fs::read_dir(&output).expect("failed to inspect acpf_cuda build output") {
        let path = entry
            .expect("failed to inspect acpf_cuda runtime file")
            .path();
        if path.extension().is_some_and(|ext| ext == "dll") {
            let name = path.file_name().expect("runtime DLL has no file name");
            fs::copy(&path, profile_dir.join(name))
                .expect("failed to copy acpf_cuda runtime beside Cargo binaries");
        }
    }
}
