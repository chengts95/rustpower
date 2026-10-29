"""用 cuDSS 求解 PEGASE 9241 的真实 Newton Jacobian，并保存分阶段成绩。"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PACKAGES = ROOT / "python-packages"
NVIDIA = PACKAGES / "nvidia"

# cudaDataType_t / cuDSS 1.0 C API 常量。
CUDA_R_64F = 1
CUDA_R_32I = 10
CUDSS_MTYPE_GENERAL = 0
CUDSS_MVIEW_FULL = 0
CUDSS_BASE_ZERO = 0
CUDSS_LAYOUT_COL_MAJOR = 0
CUDSS_PHASE_ANALYSIS = 1 | 2
CUDSS_PHASE_FACTORIZATION = 1 << 2
CUDSS_PHASE_REFACTORIZATION = 1 << 3
CUDSS_PHASE_SOLVE = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9)
CUDSS_DATA_INFO = 0
CUDSS_DATA_LU_NNZ = 1
CUDSS_DATA_NPIVOTS = 2

STATUS_NAMES = {
    0: "CUDSS_STATUS_SUCCESS",
    1: "CUDSS_STATUS_NOT_INITIALIZED",
    2: "CUDSS_STATUS_ALLOC_FAILED",
    3: "CUDSS_STATUS_INVALID_VALUE",
    4: "CUDSS_STATUS_NOT_SUPPORTED",
    5: "CUDSS_STATUS_EXECUTION_FAILED",
    6: "CUDSS_STATUS_INTERNAL_ERROR",
    7: "CUDSS_STATUS_IR_FAILED",
}


def check(status: int, operation: str) -> None:
    if status != 0:
        name = STATUS_NAMES.get(status, f"unknown status {status}")
        raise RuntimeError(f"{operation}: {name}")


def stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "median_ms": statistics.median(values),
        "best_ms": min(values),
        "worst_ms": max(values),
        "all_ms": values,
    }


class CuDssApi:
    """精简的 cuDSS C API 包装；只暴露这次测试需要的函数。"""

    def __init__(self) -> None:
        # 句柄必须一直活着，Windows 才知道去哪里找依赖 DLL。
        self._dll_directories = [
            os.add_dll_directory(str((NVIDIA / relative).resolve()))
            for relative in ("cu12/bin", "cuda_runtime/bin", "cublas/bin", "cuda_nvrtc/bin")
        ]
        self.dll = ctypes.WinDLL(str((NVIDIA / "cu12/bin/cudss64_1.dll").resolve()))
        p = ctypes.c_void_p
        i64 = ctypes.c_int64
        i = ctypes.c_int
        size = ctypes.c_size_t
        pp = ctypes.POINTER(p)
        ps = ctypes.POINTER(size)
        pi = ctypes.POINTER(i)

        self.dll.cudssCreate.argtypes = [pp]
        self.dll.cudssCreate.restype = i
        self.dll.cudssDestroy.argtypes = [p]
        self.dll.cudssDestroy.restype = i
        self.dll.cudssConfigCreate.argtypes = [pp]
        self.dll.cudssConfigCreate.restype = i
        self.dll.cudssConfigDestroy.argtypes = [p]
        self.dll.cudssConfigDestroy.restype = i
        self.dll.cudssDataCreate.argtypes = [p, pp]
        self.dll.cudssDataCreate.restype = i
        self.dll.cudssDataDestroy.argtypes = [p, p]
        self.dll.cudssDataDestroy.restype = i
        self.dll.cudssSetStream.argtypes = [p, p]
        self.dll.cudssSetStream.restype = i
        self.dll.cudssSetJitCacheDir.argtypes = [p, ctypes.c_char_p]
        self.dll.cudssSetJitCacheDir.restype = i
        self.dll.cudssMatrixCreateCsr.argtypes = [pp, i64, i64, i64, p, p, p, p, i, i, i, i, i, i]
        self.dll.cudssMatrixCreateCsr.restype = i
        self.dll.cudssMatrixCreateDn.argtypes = [pp, i64, i64, i64, p, i, i]
        self.dll.cudssMatrixCreateDn.restype = i
        self.dll.cudssMatrixDestroy.argtypes = [p]
        self.dll.cudssMatrixDestroy.restype = i
        self.dll.cudssExecute.argtypes = [p, i, p, p, p, p, p]
        self.dll.cudssExecute.restype = i
        self.dll.cudssDataGet.argtypes = [p, p, i, p, size, ps]
        self.dll.cudssDataGet.restype = i
        self.dll.cudssGetProperty.argtypes = [i, pi]
        self.dll.cudssGetProperty.restype = i

    def version(self) -> tuple[int, int, int]:
        values = []
        for prop in range(3):
            value = ctypes.c_int()
            check(self.dll.cudssGetProperty(prop, ctypes.byref(value)), f"cudssGetProperty({prop})")
            values.append(value.value)
        return tuple(values)


class CuDssSolver:
    def __init__(self, api: CuDssApi) -> None:
        self.api = api
        self.dll = api.dll
        self.handle = ctypes.c_void_p()
        self.config = ctypes.c_void_p()
        self.data = ctypes.c_void_p()
        check(self.dll.cudssCreate(ctypes.byref(self.handle)), "cudssCreate")
        check(self.dll.cudssConfigCreate(ctypes.byref(self.config)), "cudssConfigCreate")
        check(self.dll.cudssDataCreate(self.handle, ctypes.byref(self.data)), "cudssDataCreate")
        check(self.dll.cudssSetStream(self.handle, ctypes.c_void_p(cp.cuda.Stream.null.ptr)), "cudssSetStream")
        jit_dir = ROOT / "cudss-jit-cache"
        jit_dir.mkdir(parents=True, exist_ok=True)
        check(
            self.dll.cudssSetJitCacheDir(self.handle, str(jit_dir).encode("utf-8")),
            "cudssSetJitCacheDir",
        )

    def execute(self, phase: int, matrix, solution, rhs) -> None:
        check(
            self.dll.cudssExecute(self.handle, phase, self.config, self.data, matrix, solution, rhs),
            f"cudssExecute(phase={phase})",
        )

    def data_i64(self, parameter: int) -> int:
        value = ctypes.c_int64()
        written = ctypes.c_size_t()
        check(
            self.dll.cudssDataGet(
                self.handle,
                self.data,
                parameter,
                ctypes.byref(value),
                ctypes.sizeof(value),
                ctypes.byref(written),
            ),
            f"cudssDataGet({parameter})",
        )
        return int(value.value)

    def data_i32(self, parameter: int) -> int:
        # NPIVOTS 跟随稀疏索引类型；本次 CSR 使用 int32。
        value = ctypes.c_int32()
        written = ctypes.c_size_t()
        check(
            self.dll.cudssDataGet(
                self.handle,
                self.data,
                parameter,
                ctypes.byref(value),
                ctypes.sizeof(value),
                ctypes.byref(written),
            ),
            f"cudssDataGet({parameter})",
        )
        return int(value.value)

    def close(self) -> None:
        if self.data.value:
            self.dll.cudssDataDestroy(self.handle, self.data)
            self.data = ctypes.c_void_p()
        if self.config.value:
            self.dll.cudssConfigDestroy(self.config)
            self.config = ctypes.c_void_p()
        if self.handle.value:
            self.dll.cudssDestroy(self.handle)
            self.handle = ctypes.c_void_p()


def timed_phase(solver: CuDssSolver, phase: int, matrix, solution, rhs) -> float:
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    solver.execute(phase, matrix, solution, rhs)
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - start) * 1000.0


def solve_case(
    solver: CuDssSolver,
    matrix: sp.csr_matrix,
    rhs_host: np.ndarray,
    factorizations: int,
    solves: int,
) -> tuple[dict, np.ndarray]:
    nrows, ncols = matrix.shape
    nnz = matrix.nnz

    # H2D 单独计时；正式分解与回代不再为主机传输“背锅”。
    cp.cuda.Stream.null.synchronize()
    transfer_start = time.perf_counter()
    d_indptr = cp.asarray(matrix.indptr, dtype=cp.int32)
    d_indices = cp.asarray(matrix.indices, dtype=cp.int32)
    d_values = cp.asarray(matrix.data, dtype=cp.float64)
    d_rhs = cp.asarray(rhs_host, dtype=cp.float64)
    d_solution = cp.zeros(ncols, dtype=cp.float64)
    cp.cuda.Stream.null.synchronize()
    h2d_ms = (time.perf_counter() - transfer_start) * 1000.0

    a = ctypes.c_void_p()
    x = ctypes.c_void_p()
    b = ctypes.c_void_p()
    dll = solver.dll
    try:
        check(
            dll.cudssMatrixCreateCsr(
                ctypes.byref(a),
                nrows,
                ncols,
                nnz,
                ctypes.c_void_p(d_indptr.data.ptr),
                None,
                ctypes.c_void_p(d_indices.data.ptr),
                ctypes.c_void_p(d_values.data.ptr),
                CUDA_R_32I,
                CUDA_R_32I,
                CUDA_R_64F,
                CUDSS_MTYPE_GENERAL,
                CUDSS_MVIEW_FULL,
                CUDSS_BASE_ZERO,
            ),
            "cudssMatrixCreateCsr",
        )
        check(
            dll.cudssMatrixCreateDn(
                ctypes.byref(b), nrows, 1, nrows, ctypes.c_void_p(d_rhs.data.ptr), CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
            ),
            "cudssMatrixCreateDn(rhs)",
        )
        check(
            dll.cudssMatrixCreateDn(
                ctypes.byref(x), ncols, 1, ncols, ctypes.c_void_p(d_solution.data.ptr), CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR
            ),
            "cudssMatrixCreateDn(solution)",
        )

        analysis_ms = timed_phase(solver, CUDSS_PHASE_ANALYSIS, a, x, b)
        factorization_ms = []
        for index in range(factorizations):
            phase = CUDSS_PHASE_FACTORIZATION if index == 0 else CUDSS_PHASE_REFACTORIZATION
            factorization_ms.append(timed_phase(solver, phase, a, x, b))
        solve_ms = [timed_phase(solver, CUDSS_PHASE_SOLVE, a, x, b) for _ in range(solves)]
        solution_host = cp.asnumpy(d_solution)

        residual = matrix @ solution_host - rhs_host
        residual_inf = float(np.linalg.norm(residual, ord=np.inf))
        rhs_inf = float(np.linalg.norm(rhs_host, ord=np.inf))
        matrix_inf = float(sp.linalg.norm(matrix, ord=np.inf))
        solution_inf = float(np.linalg.norm(solution_host, ord=np.inf))
        denominator = matrix_inf * solution_inf + rhs_inf
        result = {
            "matrix_order": nrows,
            "nnz": nnz,
            "h2d_ms": h2d_ms,
            "analysis_ms": analysis_ms,
            "first_factorization_ms": factorization_ms[0],
            "factorization_all": stats(factorization_ms),
            "refactorization": stats(factorization_ms[1:]) if len(factorization_ms) > 1 else None,
            "solve": stats(solve_ms),
            "lu_nnz": solver.data_i64(CUDSS_DATA_LU_NNZ),
            "pivot_count": solver.data_i32(CUDSS_DATA_NPIVOTS),
            "solver_info": solver.data_i32(CUDSS_DATA_INFO),
            "residual_inf_norm": residual_inf,
            "relative_residual_inf_norm": residual_inf / rhs_inf,
            "normwise_backward_error": residual_inf / denominator,
            "rhs_inf_norm": rhs_inf,
            "solution_inf_norm": solution_inf,
        }
        result["first_solve_total_ms"] = h2d_ms + analysis_ms + factorization_ms[0] + solve_ms[0]
        result["steady_refactor_solve_total_ms"] = (
            result["refactorization"]["median_ms"] + result["solve"]["median_ms"]
            if result["refactorization"] is not None
            else None
        )
        return result, solution_host
    finally:
        # 爱莉希雅的收尾礼仪：包装器和显存都要好好送回家♪
        for wrapper in (a, x, b):
            if wrapper.value:
                dll.cudssMatrixDestroy(wrapper)
        del d_indptr, d_indices, d_values, d_rhs, d_solution
        cp.get_default_memory_pool().free_all_blocks()


def self_test(api: CuDssApi) -> dict:
    matrix = sp.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64))
    rhs = np.array([1.0, 2.0], dtype=np.float64)
    solver = CuDssSolver(api)
    try:
        result, solution = solve_case(solver, matrix, rhs, factorizations=1, solves=1)
        expected = np.linalg.solve(np.asarray(matrix.todense()), rhs)
        error = float(np.max(np.abs(solution - expected)))
        assert error < 1e-12, error
        return {"relative_residual_inf_norm": result["relative_residual_inf_norm"], "max_solution_error": error}
    finally:
        solver.close()


def main() -> None:
    cp.cuda.Device(0).use()
    api = CuDssApi()
    version = api.version()
    print(f"cuDSS version: {version[0]}.{version[1]}.{version[2]}", flush=True)
    smoke = self_test(api)
    print(f"2x2 API self-test passed: max error {smoke['max_solution_error']:.3e}", flush=True)

    data = np.load(RESULTS / "pegase9241-newton1-jacobian.npz")
    matrix = sp.csr_matrix(
        (data["values"], data["indices"], data["indptr"]),
        shape=tuple(int(v) for v in data["shape"]),
    )
    # Newton 方程为 J*dx=-F；使用真实失配向量，不换成随机右端项。
    rhs = -data["rhs"].astype(np.float64)
    metadata = json.loads((RESULTS / "pegase9241-jacobian-metadata.json").read_text(encoding="utf-8"))

    solver = CuDssSolver(api)
    try:
        print(f"Running PEGASE Jacobian: n={matrix.shape[0]}, nnz={matrix.nnz}", flush=True)
        result, solution = solve_case(solver, matrix, rhs, factorizations=7, solves=20)
    finally:
        solver.close()

    result["validation_passed"] = bool(
        np.isfinite(solution).all()
        and result["solver_info"] == 0
        and result["relative_residual_inf_norm"] < 1e-8
    )
    report = {
        "schema_version": 1,
        "status": "complete" if result["validation_passed"] else "validation_failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "cudss_version": list(version),
        "cupy": cp.__version__,
        "platform": platform.platform(),
        "matrix_metadata": metadata,
        "configuration": {
            "precision": "float64",
            "matrix_type": "general nonsymmetric",
            "layout": "CSR, zero-based, int32 offsets and indices",
            "reordering": "cuDSS default",
            "factorization": "cuDSS default LU",
            "pivoting": "cuDSS automatic default",
            "timing": "wall time around cudssExecute plus CUDA stream synchronization",
            "h2d_transfer": "reported separately from solver phases",
            "factorizations": 7,
            "solves_after_final_factorization": 20,
        },
        "api_self_test": smoke,
        "result": result,
    }
    output = RESULTS / "cudss-pegase9241-results.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(RESULTS / "cudss-pegase9241-dx.npy", solution)

    print(f"Analysis: {result['analysis_ms']:.3f} ms", flush=True)
    print(f"First factorization: {result['first_factorization_ms']:.3f} ms", flush=True)
    print(
        f"Refactorization median/best: {result['refactorization']['median_ms']:.3f} / "
        f"{result['refactorization']['best_ms']:.3f} ms",
        flush=True,
    )
    print(
        f"Solve median/best: {result['solve']['median_ms']:.3f} / {result['solve']['best_ms']:.3f} ms",
        flush=True,
    )
    print(f"LU nnz: {result['lu_nnz']}; pivots: {result['pivot_count']}", flush=True)
    print(f"Relative residual: {result['relative_residual_inf_norm']:.3e}", flush=True)
    print(f"Backward error: {result['normwise_backward_error']:.3e}", flush=True)
    print(f"Validation: {result['validation_passed']}", flush=True)
    print(f"Saved: {output}", flush=True)
    if not result["validation_passed"]:
        raise RuntimeError("cuDSS solution failed numerical validation")


if __name__ == "__main__":
    main()
