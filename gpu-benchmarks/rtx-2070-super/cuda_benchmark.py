"""可重复的 CUDA 性能档案：只计 GPU 执行时间，不把初始化算进成绩。"""

from __future__ import annotations

import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def timed_gpu_runs(operation, warmup: int, repeats: int) -> list[float]:
    """用 CUDA Event 测量；爱莉希雅的小提醒：预热轮不能混进正式成绩呀♪"""
    for _ in range(warmup):
        operation()
    cp.cuda.Device().synchronize()

    elapsed_ms: list[float] = []
    for _ in range(repeats):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        operation()
        end.record()
        end.synchronize()
        elapsed_ms.append(float(cp.cuda.get_elapsed_time(start, end)))
    return elapsed_ms


def summarize_times(times_ms: list[float]) -> dict:
    ordered = sorted(times_ms)
    return {
        "repeats": len(times_ms),
        "best_ms": min(times_ms),
        "median_ms": statistics.median(times_ms),
        "worst_ms": max(times_ms),
        "all_ms": times_ms,
        "median_minus_best_pct": (
            (statistics.median(times_ms) / min(times_ms) - 1.0) * 100.0
        ),
        "ordered_ms": ordered,
    }


def throughput_summary(times_ms: list[float], work_per_run: float, unit_scale: float) -> dict:
    result = summarize_times(times_ms)
    values = [work_per_run / (ms / 1000.0) / unit_scale for ms in times_ms]
    result["best"] = max(values)
    result["median"] = statistics.median(values)
    result["all"] = values
    return result


def memory_bandwidth() -> dict:
    # 1 GiB 源区和目标区；读+写合计按 2 GiB 计算有效带宽。
    nbytes = 1 << 30
    source = cp.empty(nbytes, dtype=cp.uint8)
    destination = cp.empty(nbytes, dtype=cp.uint8)
    source.fill(37)

    def copy_device_to_device():
        # 直接调用 CUDA 复制接口，避免数组 API 的参数歧义和额外拷贝。
        cp.cuda.runtime.memcpyAsync(
            destination.data.ptr,
            source.data.ptr,
            nbytes,
            cp.cuda.runtime.memcpyDeviceToDevice,
            cp.cuda.Stream.null.ptr,
        )

    times = timed_gpu_runs(copy_device_to_device, warmup=3, repeats=10)
    cp.cuda.Device().synchronize()
    if not bool(cp.all(destination[: 1 << 20] == 37)):
        raise RuntimeError("显存复制校验失败")
    result = throughput_summary(times, work_per_run=2 * nbytes, unit_scale=1e9)
    result.update({"bytes_per_buffer": nbytes, "unit": "GB/s", "direction": "device-to-device"})
    del source, destination
    cp.get_default_memory_pool().free_all_blocks()
    return result


FMA_SOURCE = r"""
extern "C" __global__ void fma_fp32(float *output, int iterations) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const float a = 1.00000011920928955f;
    const float b = 0.99999988079071045f;
    float c0 = 0.000001f * (index + 1);
    float c1 = c0 + 0.1f, c2 = c0 + 0.2f, c3 = c0 + 0.3f;
    float c4 = c0 + 0.4f, c5 = c0 + 0.5f, c6 = c0 + 0.6f, c7 = c0 + 0.7f;
    for (int i = 0; i < iterations; ++i) {
        c0 = fmaf(a, b, c0); c1 = fmaf(a, b, c1);
        c2 = fmaf(a, b, c2); c3 = fmaf(a, b, c3);
        c4 = fmaf(a, b, c4); c5 = fmaf(a, b, c5);
        c6 = fmaf(a, b, c6); c7 = fmaf(a, b, c7);
    }
    output[index] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
}

extern "C" __global__ void fma_fp64(double *output, int iterations) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const double a = 1.0000000000000002;
    const double b = 0.9999999999999998;
    double c0 = 0.000001 * (index + 1);
    double c1 = c0 + 0.1, c2 = c0 + 0.2, c3 = c0 + 0.3;
    double c4 = c0 + 0.4, c5 = c0 + 0.5, c6 = c0 + 0.6, c7 = c0 + 0.7;
    for (int i = 0; i < iterations; ++i) {
        c0 = fma(a, b, c0); c1 = fma(a, b, c1);
        c2 = fma(a, b, c2); c3 = fma(a, b, c3);
        c4 = fma(a, b, c4); c5 = fma(a, b, c5);
        c6 = fma(a, b, c6); c7 = fma(a, b, c7);
    }
    output[index] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
}
"""


def fma_throughput(precision: str, iterations: int, repeats: int) -> dict:
    # 按设备报告值填满常驻线程；8 条独立依赖链提供充足 ILP。
    properties = cp.cuda.runtime.getDeviceProperties(0)
    threads = properties["multiProcessorCount"] * properties["maxThreadsPerMultiProcessor"]
    block = 256
    grid = threads // block
    if precision == "fp32":
        output = cp.empty(threads, dtype=cp.float32)
    elif precision == "fp64":
        output = cp.empty(threads, dtype=cp.float64)
    else:
        raise ValueError(precision)

    kernel = cp.RawKernel(FMA_SOURCE, f"fma_{precision}", options=("--std=c++11",))

    def run():
        kernel((grid,), (block,), (output, np.int32(iterations)))

    times = timed_gpu_runs(run, warmup=2, repeats=repeats)
    cp.cuda.Device().synchronize()
    if not bool(cp.all(cp.isfinite(output))) or float(output[0]) == 0.0:
        raise RuntimeError(f"{precision.upper()} FMA 校验失败")

    flops = threads * iterations * 8 * 2
    result = throughput_summary(times, work_per_run=flops, unit_scale=1e12)
    result.update(
        {
            "unit": "TFLOP/s",
            "method": "custom dependent FMA kernel, 8 independent chains per thread",
            "threads": threads,
            "iterations": iterations,
            "flops_per_run": flops,
        }
    )
    del output
    cp.get_default_memory_pool().free_all_blocks()
    return result


def gemm_throughput(precision: str, size: int, repeats: int) -> dict:
    dtype = {"fp16": cp.float16, "fp32": cp.float32, "fp64": cp.float64}[precision]
    # 全一矩阵让结果校验简单明确；cuBLAS 不会根据矩阵取值跳过运算。
    a = cp.ones((size, size), dtype=dtype)
    b = cp.ones((size, size), dtype=dtype)
    output = cp.empty((size, size), dtype=dtype)
    handle = cp.cuda.device.get_cublas_handle()
    if precision == "fp16":
        math_mode = cp.cuda.cublas.CUBLAS_TENSOR_OP_MATH
        method = "cuBLAS HGEMM with tensor-op math enabled"
    else:
        math_mode = cp.cuda.cublas.CUBLAS_DEFAULT_MATH
        method = "cuBLAS GEMM with default (non-TF32) math"
    cp.cuda.cublas.setMathMode(handle, math_mode)

    def run():
        cp.matmul(a, b, out=output)

    times = timed_gpu_runs(run, warmup=3, repeats=repeats)
    cp.cuda.Device().synchronize()
    sample = float(output[0, 0])
    if abs(sample - size) > max(0.01, size * 1e-5):
        raise RuntimeError(f"{precision.upper()} GEMM 校验失败：得到 {sample}，预期 {size}")

    flops = 2 * (size**3)
    result = throughput_summary(times, work_per_run=flops, unit_scale=1e12)
    result.update(
        {
            "unit": "TFLOP/s",
            "method": method,
            "matrix_shape": [size, size, size],
            "flops_per_run": flops,
            "output_sample": sample,
        }
    )
    del a, b, output
    cp.get_default_memory_pool().free_all_blocks()
    return result


def main() -> None:
    cp.cuda.Device(0).use()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"].decode()
    print(f"Device: {device_name}", flush=True)
    print(f"CuPy: {cp.__version__}; CUDA runtime: {cp.cuda.runtime.runtimeGetVersion()}", flush=True)

    report: dict = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": device_name,
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "sm_count": properties["multiProcessorCount"],
            "global_memory_gib": properties["totalGlobalMem"] / (1 << 30),
            "memory_bus_width_bits": properties["memoryBusWidth"],
            "memory_clock_khz": properties["memoryClockRate"],
            "gpu_clock_khz": properties["clockRate"],
            "max_threads_per_sm": properties["maxThreadsPerMultiProcessor"],
            "pci_bus_id": properties["pciBusID"],
            "pci_device_id": properties["pciDeviceID"],
        },
        "software": {
            "python": sys.version.split()[0],
            "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "platform": platform.platform(),
        },
        "method": {
            "timing": "CUDA events around GPU operations only; warmup excluded",
            "comparison_rule": "Reuse this script and dependency versions when benchmarking another GPU",
        },
        "results": {},
    }

    stages = [
        ("memory_d2d", "Device memory bandwidth", memory_bandwidth),
        ("fp32_fma", "FP32 FMA throughput", lambda: fma_throughput("fp32", 262144, 7)),
        ("fp64_fma", "FP64 FMA throughput", lambda: fma_throughput("fp64", 32768, 5)),
        ("fp32_gemm", "FP32 4096³ GEMM", lambda: gemm_throughput("fp32", 4096, 10)),
        ("fp16_gemm_tensor", "FP16 4096³ tensor GEMM", lambda: gemm_throughput("fp16", 4096, 10)),
        ("fp64_gemm", "FP64 4096³ GEMM", lambda: gemm_throughput("fp64", 4096, 5)),
    ]
    for key, label, operation in stages:
        print(f"Running: {label} ...", flush=True)
        report["results"][key] = operation()
        value = report["results"][key]
        print(
            f"  median {value['median']:.3f} {value['unit']}; "
            f"best {value['best']:.3f} {value['unit']}",
            flush=True,
        )
        # 每个阶段都落盘，就算后续被打断，也能保住已完成的成绩。
        (RESULTS / "cuda-results.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    report["status"] = "complete"
    (RESULTS / "cuda-results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {RESULTS / 'cuda-results.json'}", flush=True)


if __name__ == "__main__":
    main()
