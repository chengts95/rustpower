# Experimental CUDA N-1 Guide

RustPower 0.6.0 adds an experimental FP64 CUDA/cuDSS executor to the existing N-1 study API. The CPU and accelerated paths share baseline preparation, branch-outage patch generation, island screening, watch rules, and result categories.

## Installation

The standard installation remains lightweight and works without NVIDIA software:

```bash
pip install rustpower
```

Windows wheels contain the small `acpf_cuda.dll` bridge but not the large NVIDIA runtime libraries. To enable GPU execution, install the optional runtime dependencies:

```bash
pip install "rustpower[cuda]"
```

The `cuda` extra installs `nvidia-cudss-cu12==1.0.0.35` and its matching pip dependencies. A CUDA Toolkit installation is only needed to compile the bridge from source, not to use a wheel. A compatible NVIDIA driver and device are still required.

Linux and macOS wheels currently provide the CPU N-1 backend only.

## Backend selection

```python
import pandapower.networks as networks
import rustpower as rp

net = networks.case118()
grid = rp.PowerGrid.from_pandapower(net, qlim=False)
study = grid.prepare_n1(voltage_limits=True, line_loading_limit=100.0)

available, diagnostic = rp.n1_cuda_status()
print(available, diagnostic)

# Use CUDA when the runtime and device are available; otherwise use the CPU.
report = study.run(workers=4, backend="auto")
print(report.summary["backend"])  # the backend that actually executed

# Explicit selections never silently change backends.
cpu = study.run(workers=4, backend="cpu")
cuda = study.run(backend="cuda", cuda_batch_size=64, cuda_streams=1)
hybrid = study.run(
    workers=4,
    backend="hybrid",
    batch_size=8,
    cuda_batch_size=64,
    cuda_streams=1,
    cuda_queue_batches=2,
)
```

`backend="auto"` is the default. It selects CUDA only for a `base-ac` study when the native library, NVIDIA runtime, driver, and device 0 all pass the preflight probe. Otherwise it selects the CPU before execution starts. An explicit `cuda` or `hybrid` request raises `RuntimeError` when CUDA is unavailable.

The hybrid executor fills its bounded CUDA queue first and overflows tasks to CPU workers. It preserves N-1 result semantics, but it is not guaranteed to outperform either pure backend for every network size or fault count. Small systems often favor the CPU; large batches on large systems are the intended GPU use case.

Set `RUSTPOWER_N1_DISABLE_CUDA=1` to force CPU auto-selection, including on a GPU-equipped machine. This is useful for deployment control and fallback testing.

## Current limitations

- Experimental API and execution path; Windows x64 only for CUDA.
- CUDA and hybrid execution currently require `initialization="base-ac"`.
- Device 0 is used; multi-device scheduling is not implemented.
- Newton iteration, Ybus updates, and Jacobian values are FP64.
- The N-1 model uses a fixed bus partition. Prepare grids with `qlim=False`.
- Multiple CUDA streams use independent executors on one device, not multiple GPUs.
- CPU remains the default for ordinary Rust Cargo builds and for non-CUDA Python wheels.

## Native Rust builds

The default Cargo features do not compile or load CUDA. Enable it explicitly:

```bash
cargo build --features experimental_n1_cuda
```

On Windows, the build script locates MSVC and NVCC. Set these variables when they are not auto-detected:

```text
CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
CUDSS_DIR=C:\path\to\nvidia\cu12
RUSTPOWER_CUDA_ARCH=sm_120
```

`CUDSS_DIR` must contain `include`, `lib`, and `bin`. Use the architecture appropriate for the local GPU, such as `sm_75` for Turing or `sm_120` for RTX 5070. The build also embeds matching PTX for forward-compatible JIT. The official portable Windows wheel uses `sm_75` plus PTX.

`acpf_cuda.dll` is loaded on demand rather than linked as a startup dependency of the RustPower Python extension. Consequently, a missing CUDA runtime does not prevent `import rustpower` or CPU N-1 execution. Native CUDA/cuDSS calls remain ahead-of-time compiled C++; PTX is a compatibility artifact, not an NVRTC implementation of the Newton kernels.

## Validation

Run the Python API regression suite with CPU-only behavior forced:

```bash
RUSTPOWER_N1_DISABLE_CUDA=1 python scripts/test_n1_api.py
```

Run it without that variable to include CUDA and hybrid equivalence tests when a device is available. The tests compare processed, passed, violated, numerical-failure, island, and iteration counts and check detailed CUDA event output.

The RustPower GPU benchmark reports contain the larger PEGASE 9241 measurements and the RTX 2070 SUPER versus RTX 5070 comparison. Performance results are platform- and workload-specific; correctness and actual backend selection should be checked independently of throughput.
