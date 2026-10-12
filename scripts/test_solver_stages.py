"""Regression checks for Python entry points, independent of timing benchmarks.

Set RUSTPOWER_TEST_LIBRARY to test an isolated librustpower.so without installing it.
Run with a Python environment containing NumPy and pandapower.
"""

import importlib.util
import os
import unittest

import numpy as np
import pandapower.networks as pn

if library := os.environ.get("RUSTPOWER_TEST_LIBRARY"):
    spec = importlib.util.spec_from_file_location("rustpower", library)
    rustpower = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rustpower)
else:
    import rustpower


class SolverStages(unittest.TestCase):
    def test_standalone_dc_consumes_load_edits_once(self):
        net = pn.case9()
        grid = rustpower.PowerGrid.from_pandapower(net)
        old = grid.compute_dcpf_v()
        row = net.load.index[0]
        bus = int(net.load.at[row, "bus"])
        net.load.at[row, "p_mw"] += 20.0
        grid.loads(bus=bus)[0].p_mw = float(net.load.at[row, "p_mw"])
        updated = grid.compute_dcpf_v()
        fresh = rustpower.PowerGrid.from_pandapower(net)
        np.testing.assert_allclose(updated, fresh.compute_dcpf_v(), atol=1e-12)
        self.assertGreater(np.max(np.abs(updated - old)), 1e-4)
        np.testing.assert_allclose(grid.compute_dcpf_v(), updated, atol=1e-12)
        self.assertTrue(grid.solve(tol=1e-10).converged)
        self.assertTrue(fresh.solve(tol=1e-10).converged)
        np.testing.assert_allclose(grid.v, fresh.v, atol=1e-9)

    def test_applied_dc_voltage_survives_the_next_prepare(self):
        grid = rustpower.PowerGrid.from_pandapower(pn.case9())
        self.assertTrue(grid.solve(tol=1e-10).converged)
        dc_voltage = grid.compute_dcpf_v()
        grid.apply_dcpf_init()
        report = grid.solve(max_iter=0, tol=1e-10)
        self.assertFalse(report.converged)
        np.testing.assert_allclose(grid.v, dc_voltage, atol=1e-12)

    def test_dc_strategy_can_be_selected_before_initialization(self):
        grid = rustpower.PowerGrid()
        grid.enable_dcpf_init(True)
        grid.from_pp_net(pn.case9())
        self.assertTrue(grid.solve(tol=1e-10).converged)
        grid.enable_dcpf_init(False)
        self.assertTrue(grid.solve(tol=1e-10).converged)
        grid.enable_dcpf_init(True)
        self.assertTrue(grid.solve(tol=1e-10).converged)

    def test_clear_and_rebuild_do_not_change_the_solution(self):
        grid = rustpower.PowerGrid.from_pandapower(pn.case39())
        self.assertTrue(grid.solve(tol=1e-10).converged)
        expected = grid.v.copy()
        grid.clear_cache()
        self.assertTrue(grid.solve(tol=1e-10).converged)
        np.testing.assert_allclose(grid.v, expected, atol=1e-9)
        grid.init_pf()
        self.assertTrue(grid.solve(tol=1e-10).converged)
        np.testing.assert_allclose(grid.v, expected, atol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
