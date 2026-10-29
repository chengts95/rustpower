"""导出 PEGASE 9241 第一轮 Newton  Jacobian 与同状态失配向量。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    net = pn.case9241pegase()
    # 只走一轮 Newton，并保留初始电压；爱莉希雅想要的是同一份 J 与 F，不能错位哦♪
    try:
        pp.runpp(
            net,
            calculate_voltage_angles=True,
            init="flat",
            max_iteration=1,
            numba=False,
            v_debug=True,
        )
    except pp.auxiliary.LoadflowNotConverged:
        pass

    ppci = net._ppc["internal"]
    jacobian = ppci["J"].tocsr().astype(np.float64)
    jacobian.sort_indices()
    vm0 = ppci["Vm_it"][:, 0]
    va0 = ppci["Va_it"][:, 0]
    voltage0 = vm0 * np.exp(1j * va0)
    ybus = ppci["Ybus"].tocsr()
    sbus = ppci["Sbus"]
    pv = ppci["pv"]
    pq = ppci["pq"]

    mismatch = voltage0 * np.conj(ybus @ voltage0) - sbus
    rhs = np.concatenate((mismatch[pv].real, mismatch[pq].real, mismatch[pq].imag))

    n = jacobian.shape[0]
    assert jacobian.shape == (n, n)
    assert rhs.shape == (n,)
    assert n == len(pv) + 2 * len(pq)
    assert np.isfinite(jacobian.data).all() and np.isfinite(rhs).all()
    assert jacobian.indptr.dtype == np.int32 or jacobian.indptr.dtype == np.int64

    # RustPower 的算子使用 [PQ | PV | slack] 总线顺序，变量顺序为
    # [PQ angle | PV angle | PQ magnitude]。导出这一视图，避免 C++ 再猜映射。
    ref = np.setdiff1d(np.arange(ppci["bus"].shape[0]), np.concatenate((pv, pq)))
    bus_order = np.concatenate((pq, pv, ref))
    ybus_permuted = ybus[bus_order][:, bus_order].tocsc()
    ybus_permuted.sort_indices()
    voltage_permuted = voltage0[bus_order]
    sbus_permuted = sbus[bus_order]
    scalc_permuted = voltage_permuted * np.conj(ybus_permuted @ voltage_permuted)
    mismatch_permuted = scalc_permuted - sbus_permuted
    npv, npq = len(pv), len(pq)
    variable_order = np.concatenate((np.arange(npv, npv + npq), np.arange(npv), np.arange(npv + npq, n)))
    jacobian_permuted = jacobian[variable_order][:, variable_order].tocsr()
    jacobian_permuted.sort_indices()
    rhs_permuted = np.concatenate((mismatch_permuted.real[: npv + npq], mismatch_permuted.imag[:npq]))

    # 同一初始条件跑到真正收敛，保留完整 Newton 的参考电压和实际迭代次数。
    pp.runpp(net, calculate_voltage_angles=True, init="flat", max_iteration=20, numba=False, v_debug=True)
    converged_ppci = net._ppc["internal"]
    converged_iterations = int(net._ppc["iterations"])
    converged_vm = converged_ppci["Vm_it"][:, -1]
    converged_va = converged_ppci["Va_it"][:, -1]
    converged_voltage = converged_vm * np.exp(1j * converged_va)
    converged_voltage_permuted = converged_voltage[bus_order]
    converged_mismatch = converged_voltage * np.conj(ybus @ converged_voltage) - sbus
    converged_norm = float(np.max(np.abs(np.concatenate((
        converged_mismatch[pv].real, converged_mismatch[pq].real, converged_mismatch[pq].imag
    )))))

    matrix_path = RESULTS / "pegase9241-newton1-jacobian.npz"
    np.savez_compressed(
        matrix_path,
        indptr=jacobian.indptr.astype(np.int32),
        indices=jacobian.indices.astype(np.int32),
        values=jacobian.data.astype(np.float64),
        rhs=rhs.astype(np.float64),
        shape=np.array(jacobian.shape, dtype=np.int32),
        voltage0=voltage0,
        pv=pv.astype(np.int32),
        pq=pq.astype(np.int32),
        rust_indptr=jacobian_permuted.indptr.astype(np.int32),
        rust_indices=jacobian_permuted.indices.astype(np.int32),
        rust_values=jacobian_permuted.data.astype(np.float64),
        rust_rhs=rhs_permuted.astype(np.float64),
        ybus_indptr=ybus_permuted.indptr.astype(np.int32),
        ybus_indices=ybus_permuted.indices.astype(np.int32),
        ybus_values=ybus_permuted.data.astype(np.complex128),
        voltage_permuted=voltage_permuted.astype(np.complex128),
        sbus_permuted=sbus_permuted.astype(np.complex128),
        scalc_permuted=scalc_permuted.astype(np.complex128),
        converged_voltage_permuted=converged_voltage_permuted.astype(np.complex128),
        converged_iterations=np.array(converged_iterations, dtype=np.int32),
        converged_norm=np.array(converged_norm, dtype=np.float64),
        bus_order=bus_order.astype(np.int32),
        npv=np.array(npv, dtype=np.int32),
        npq=np.array(npq, dtype=np.int32),
    )
    metadata = {
        "case": "PEGASE 9241",
        "source": "pandapower.networks.case9241pegase",
        "newton_iteration": 1,
        "initialization": "flat",
        "buses": len(net.bus),
        "npv": len(pv),
        "npq": len(pq),
        "matrix_order": n,
        "nnz": jacobian.nnz,
        "density": jacobian.nnz / (n * n),
        "rhs_inf_norm": float(np.linalg.norm(rhs, ord=np.inf)),
        "rhs_2_norm": float(np.linalg.norm(rhs)),
        "matrix_values_max_abs": float(np.max(np.abs(jacobian.data))),
        "explicit_zero_values": int(np.count_nonzero(jacobian.data == 0.0)),
        "matrix_file": matrix_path.name,
        "layout": "CSR, zero-based, float64 values, int32 indices",
        "rhs_meaning": "F = [Re(mis[PV]); Re(mis[PQ]); Im(mis[PQ])], mis = V*conj(Ybus*V)-Sbus",
        "solve_equation": "J * dx = -F; benchmark uses b = -F",
    }
    (RESULTS / "pegase9241-jacobian-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
