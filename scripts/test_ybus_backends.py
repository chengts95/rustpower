"""Native/ECS Python dispatch and numerical agreement for Ybus AMD backends."""
import unittest
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import rustpower
from rustpower.solver import NewtonSolver as LegacyNewtonSolver
if "klu_ybus_amd" in rustpower.features():
    from rustpower.solver import AmdNewtonSolver as NewtonSolver


@unittest.skipUnless("klu_ybus_amd_rust" in rustpower.features(), "requires Rust Ybus AMD feature")
class YbusBackends(unittest.TestCase):
    def test_legacy_class_and_grid_remain_opt_in(self):
        solver = LegacyNewtonSolver()
        self.assertFalse(hasattr(solver, "set_ybus_ordering_backend"))
        grid = rustpower.PowerGrid.from_pandapower(pn.case39())
        self.assertEqual(grid.ybus_ordering_stats()["backend"], "scalar")
        self.assertTrue(grid.solve(tol=1e-10).converged)
        self.assertEqual(grid.ybus_ordering_stats()["preparations"], 0)

    def test_native_switching_preserves_solution_and_reuses_symbolic(self):
        net = pn.case39()
        pp.runpp(net, lightsim2grid=False)
        internal = net._ppc["internal"]
        y = internal["Ybus"].tocsr()
        pq, pv, ref = (internal[key] for key in ("pq", "pv", "ref"))
        v = np.ones(len(net.bus), dtype=complex)
        v[pv] = np.abs(internal["V"][pv])
        v[ref] = internal["V"][ref]
        perm = np.concatenate([pq, pv, ref]).astype(np.int64)
        s = NewtonSolver()
        self.assertEqual(s.ybus_ordering_stats()["backend"], "rust")
        s.setup_context(y.indptr, y.indices, y.data, internal["Sbus"], v,
                        perm, np.argsort(perm), len(pv), len(pq))
        expected = None
        for backend in ("c", "rust", "scalar", "rust"):
            s.set_ybus_ordering_backend(backend)
            self.assertTrue(s.solve(30, 1e-10))
            self.assertEqual(s.ybus_ordering_stats()["backend"], backend)
            self.assertEqual(s.ybus_ordering_stats()["optimized"], backend != "scalar")
            actual = s.get_voltage()
            if expected is None:
                expected = actual.copy()
            np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
            generation = s.ybus_ordering_stats()["preparations"]
            self.assertTrue(s.solve(30, 1e-10))
            self.assertEqual(s.ybus_ordering_stats()["preparations"], generation)
            self.assertEqual(len(s.get_scalc()), len(net.bus))
        before = s.ybus_ordering_stats()
        with self.assertRaises(ValueError):
            s.set_ybus_ordering_backend("unknown")
        self.assertEqual(s.ybus_ordering_stats(), before)

    def test_grid_switching_uses_selected_backend(self):
        net = pn.case39()
        grid = rustpower.PowerGrid.from_pandapower(net)
        grid.enable_cache(True)
        v = np.ones(len(net.bus), dtype=complex)
        expected = None
        for backend in ("c", "rust", "scalar", "rust"):
            grid.set_ybus_ordering_backend(backend)
            self.assertTrue(grid.solve(v, max_iter=30, tol=1e-10).converged)
            grid.post_process()
            stats = grid.ybus_ordering_stats()
            self.assertEqual(stats["backend"], backend)
            self.assertEqual(stats["optimized"], backend != "scalar")
            if expected is None:
                expected = grid.v.copy()
            np.testing.assert_allclose(grid.v, expected, atol=1e-9, rtol=0)
            self.assertTrue(grid.solve(v, max_iter=30, tol=1e-10).converged)
            self.assertEqual(grid.ybus_ordering_stats()["preparations"], stats["preparations"])


if __name__ == "__main__":
    unittest.main()
