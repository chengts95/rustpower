"""把 PEGASE 9241 的真实 AC 状态与 RustPower CSR 符号布局导出给 C++。"""
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "gpu-benchmarks/rtx-2070-super/results/pegase9241-newton1-jacobian.npz"
OUTPUT = ROOT / "target/acpf-cuda/pegase9241-jacobian.bin"


def parts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.ascontiguousarray(values.real, dtype="<f8"),
        np.ascontiguousarray(values.imag, dtype="<f8"),
    )


def main() -> None:
    data = np.load(SOURCE)
    required = {
        "rust_indptr", "rust_indices", "rust_values", "rust_rhs",
        "ybus_indptr", "ybus_indices", "ybus_values", "voltage_permuted",
        "sbus_permuted", "scalc_permuted", "converged_voltage_permuted",
        "converged_iterations", "converged_norm", "npv", "npq",
    }
    missing = required.difference(data.files)
    assert not missing, f"先运行 export_pegase9241_jacobian.py；缺少 {sorted(missing)}"

    indptr = np.asarray(data["rust_indptr"], dtype="<i4")
    indices = np.asarray(data["rust_indices"], dtype="<i4")
    values = np.asarray(data["rust_values"], dtype="<f8")
    rhs = np.asarray(data["rust_rhs"], dtype="<f8")
    ycp = np.asarray(data["ybus_indptr"], dtype="<i4")
    yri = np.asarray(data["ybus_indices"], dtype="<i4")
    y_re, y_im = parts(np.asarray(data["ybus_values"], dtype=np.complex128))
    v_re, v_im = parts(np.asarray(data["voltage_permuted"], dtype=np.complex128))
    sbus_re, sbus_im = parts(np.asarray(data["sbus_permuted"], dtype=np.complex128))
    scalc_re, scalc_im = parts(np.asarray(data["scalc_permuted"], dtype=np.complex128))
    converged_re, converged_im = parts(np.asarray(data["converged_voltage_permuted"], dtype=np.complex128))
    converged_iterations = np.asarray([data["converged_iterations"]], dtype="<i4")
    converged_norm = np.asarray([data["converged_norm"]], dtype="<f8")
    npv, npq = int(data["npv"]), int(data["npq"])
    nbus = len(v_re)
    n_active = npv + npq
    n, nnz = len(rhs), len(values)
    ynnz = len(yri)
    assert n == n_active + npq and len(indptr) == n + 1
    assert len(ycp) == nbus + 1 and int(ycp[-1]) == ynnz
    assert len(indices) == len(values) == int(indptr[-1]) == nnz

    # 与 JacobianCache::build_from_permuted 相同的分段和镜像位置。
    pq_ends = np.empty(n_active, dtype="<i4")
    active_ends = np.empty(n_active, dtype="<i4")
    diag_ptrs = np.empty(n_active, dtype="<i4")
    mirror = np.empty(ynnz, dtype="<i4")
    for k in range(nbus):
        rows = yri[ycp[k]:ycp[k + 1]]
        for p in range(ycp[k], ycp[k + 1]):
            i = int(yri[p])
            mirror_rows = yri[ycp[i]:ycp[i + 1]]
            mirror[p] = ycp[i] + int(np.searchsorted(mirror_rows, k))
            assert int(yri[mirror[p]]) == k
        if k < n_active:
            pq_ends[k] = np.searchsorted(rows, npq)
            active_ends[k] = np.searchsorted(rows, n_active)
            diag_ptrs[k] = ycp[k] + int(np.flatnonzero(rows == k)[0])
    active_ynnz = int(ycp[n_active])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    arrays = (
        indptr, indices, values, rhs,
        ycp, yri, mirror, y_re, y_im,
        v_re, v_im, sbus_re, sbus_im, scalc_re, scalc_im,
        pq_ends, active_ends, diag_ptrs,
        converged_re, converged_im, converged_iterations, converged_norm,
    )
    with OUTPUT.open("wb") as stream:
        stream.write(struct.pack("<8s7I", b"RPACPF2\0", n, nnz, nbus, ynnz, npv, npq, active_ynnz))
        for array in arrays:
            stream.write(np.ascontiguousarray(array).tobytes(order="C"))
    print(
        f"exported n={n} nnz={nnz} nbus={nbus} ynnz={ynnz} "
        f"npv={npv} npq={npq} active_ynnz={active_ynnz} to {OUTPUT}"
    )


if __name__ == "__main__":
    main()
