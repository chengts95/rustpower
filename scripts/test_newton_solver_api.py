"""
Comprehensive Functional & Unit Test Suite for rustpower.solver.NewtonSolver
Tests all Python-side NewtonSolver APIs and ensures mathematical correctness.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from rustpower.solver import NewtonSolver

def run_tests():
    print("=" * 60)
    print("🧪 Running Comprehensive NewtonSolver API Test Suite")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Network Extraction (IEEE 118)
    # ------------------------------------------------------------------
    net = pn.case118()
    pp.runpp(net, algorithm='nr', tolerance_mva=1e-8, init='flat')
    ppci = net["_ppc"]
    internal = ppci["internal"]

    v_pp = internal["V"]
    Ybus = internal["Ybus"]
    Sbus = internal["Sbus"]
    pq = internal["pq"]
    pv = internal["pv"]
    ref = internal["ref"]

    n_buses = len(v_pp)
    v_init_flat = np.ones(n_buses, dtype=np.complex128)
    v_init_flat[pv] = np.abs(v_pp[pv])
    v_init_flat[ref] = v_pp[ref]

    # ------------------------------------------------------------------
    # Test 2: setup_from_nodes (Suggestion 1: Zero-boilerplate Setup)
    # ------------------------------------------------------------------
    print("\n[Test 1] setup_from_nodes (Automatic partition concatenation)...")
    solver = NewtonSolver()
    solver.setup_from_nodes(
        Ybus.indptr,
        Ybus.indices,
        Ybus.data,
        Sbus,
        v_init_flat,
        pq,
        pv,
        ref,
    )
    assert solver.n_buses == n_buses, f"Expected {n_buses} buses, got {solver.n_buses}"
    
    converged = solver.solve(max_iter=20, tol=1e-6)
    assert converged, "Solver failed to converge!"
    assert solver.converged is True
    assert solver.get_iterations() > 0
    print(f"  ✓ Converged in {solver.get_iterations()} iterations")

    v_rp = solver.get_voltage()
    diff = np.linalg.norm(v_rp - v_pp)
    assert diff < 1e-6, f"Voltage mismatch vs Pandapower: {diff:.2e}"
    print(f"  ✓ L2 Voltage difference vs Pandapower: {diff:.2e} (< 1e-6)")

    # ------------------------------------------------------------------
    # Test 3: Voltage Magnitudes & Angles (Suggestion 2)
    # ------------------------------------------------------------------
    print("\n[Test 2] Voltage Magnitude and Angle APIs (.vm, .va, .va_deg)...")
    vm = solver.vm
    vm_meth = solver.get_voltage_magnitude()
    np.testing.assert_allclose(vm, vm_meth, rtol=1e-12)
    np.testing.assert_allclose(vm, np.abs(v_pp), atol=1e-6)
    print(f"  ✓ .vm matches np.abs(v) (range: [{vm.min():.4f}, {vm.max():.4f}] p.u.)")

    va_rad = solver.va
    va_rad_meth = solver.get_voltage_angle(deg=False)
    np.testing.assert_allclose(va_rad, va_rad_meth, rtol=1e-12)
    np.testing.assert_allclose(va_rad, np.angle(v_pp), atol=1e-6)
    print(f"  ✓ .va (radians) matches np.angle(v)")

    va_deg = solver.va_deg
    va_deg_meth = solver.get_voltage_angle(deg=True)
    np.testing.assert_allclose(va_deg, va_deg_meth, rtol=1e-12)
    np.testing.assert_allclose(va_deg, np.angle(v_pp) * 180.0 / np.pi, atol=1e-5)
    print(f"  ✓ .va_deg matches np.angle(v) * 180 / pi")

    # ------------------------------------------------------------------
    # Test 4: Residual Norm & Power Injections (Suggestion 4)
    # ------------------------------------------------------------------
    print("\n[Test 3] Residual Norm & Injections (.residual_norm, .p_calc, .q_calc)...")
    res_norm = solver.residual_norm
    res_norm_meth = solver.get_residual()
    assert res_norm == res_norm_meth
    assert res_norm < 1e-6, f"Residual norm too large: {res_norm:.2e}"
    print(f"  ✓ Final residual norm ||F||_inf: {res_norm:.2e} (< 1e-6)")

    scalc = solver.get_scalc()
    p_calc = solver.p_calc
    q_calc = solver.q_calc
    np.testing.assert_allclose(p_calc, scalc.real, rtol=1e-12)
    np.testing.assert_allclose(q_calc, scalc.imag, rtol=1e-12)
    print(f"  ✓ .p_calc and .q_calc match complex Scalc decomposition")

    # ------------------------------------------------------------------
    # Test 5: Single-Bus & Batch Updates (Suggestion 3)
    # ------------------------------------------------------------------
    print("\n[Test 4] Local Updates (set_s_bus_at, update_s_bus_batch, set_v_init_at)...")
    # Single bus update
    target_bus = int(pq[0])
    orig_s = solver.s_bus[target_bus]
    new_s = complex(1.23, 0.45)
    solver.set_s_bus_at(target_bus, new_s)
    assert np.isclose(solver.s_bus[target_bus], new_s), "set_s_bus_at failed to update target bus!"
    # Restore
    solver.set_s_bus_at(target_bus, orig_s)
    assert np.isclose(solver.s_bus[target_bus], orig_s)
    print("  ✓ set_s_bus_at correctly modified single bus injection")

    # Batch bus update
    batch_buses = np.array([int(pq[1]), int(pq[2])], dtype=np.int64)
    batch_s = np.array([complex(0.5, 0.1), complex(0.8, 0.2)], dtype=np.complex128)
    solver.update_s_bus_batch(batch_buses, batch_s)
    assert np.isclose(solver.s_bus[batch_buses[0]], batch_s[0])
    assert np.isclose(solver.s_bus[batch_buses[1]], batch_s[1])
    print("  ✓ update_s_bus_batch correctly modified multiple bus injections")

    # Single & Batch v_init update
    solver.set_v_init_at(target_bus, complex(1.05, 0.0))
    assert np.isclose(solver.v_init[target_bus], 1.05)
    solver.update_v_init_batch(batch_buses, np.array([1.02, 1.03], dtype=np.complex128))
    assert np.isclose(solver.v_init[batch_buses[0]], 1.02)
    assert np.isclose(solver.v_init[batch_buses[1]], 1.03)
    print("  ✓ set_v_init_at and update_v_init_batch correctly modified v_init")

    # ------------------------------------------------------------------
    # Test 6: setup_context Safeguard (Cache & Symbolic Invalidation)
    # ------------------------------------------------------------------
    print("\n[Test 5] setup_context & setup_from_nodes Cache Invalidation Safeguard...")
    # Enable cache on solver
    solver.enable_cache(True)
    solver.solve() # Populates NewtonCache and KLU factorizations
    
    # Now switch network to Case 39 on the same solver instance
    net39 = pn.case39()
    pp.runpp(net39, algorithm='nr', tolerance_mva=1e-8, init='flat')
    i39 = net39["_ppc"]["internal"]
    v_init39 = np.ones(len(i39["V"]), dtype=np.complex128)
    v_init39[i39["pv"]] = np.abs(i39["V"][i39["pv"]])
    v_init39[i39["ref"]] = i39["V"][i39["ref"]]

    # Re-setup: This MUST clean stale cache from 118 buses and cleanly solve 39 buses!
    solver.setup_from_nodes(
        i39["Ybus"].indptr,
        i39["Ybus"].indices,
        i39["Ybus"].data,
        i39["Sbus"],
        v_init39,
        i39["pq"],
        i39["pv"],
        i39["ref"],
    )
    assert solver.n_buses == 39, f"Expected 39 buses, got {solver.n_buses}"
    
    conv39 = solver.solve()
    assert conv39 is True, "Case 39 failed to converge after re-setup!"
    v_rp39 = solver.get_voltage()
    diff39 = np.linalg.norm(v_rp39 - i39["V"])
    assert diff39 < 1e-6, f"Case 39 mismatch vs Pandapower: {diff39:.2e}"
    print("  ✓ Re-setup cleanly invalidated stale cache and solved new grid flawlessly!")

    # ------------------------------------------------------------------
    # Test 7: reset_cache Method
    # ------------------------------------------------------------------
    print("\n[Test 6] reset_cache API...")
    solver.reset_cache()
    conv_after_reset = solver.solve()
    assert conv_after_reset is True
    print("  ✓ reset_cache successfully reset internal solver state")

    print("\n" + "=" * 60)
    print("🎉 ALL NEWTONSOLVER API TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
