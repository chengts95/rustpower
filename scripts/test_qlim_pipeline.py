"""Q-limit regression checks using power balance in RustPower's own Ybus.

RUSTPOWER_TEST_LIBRARY can select a freshly built extension without replacing
the installed Python package. Run with the project's pandapower environment.
"""

import unittest
from copy import deepcopy

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from test_solver_stages import rustpower


def make_grid(net, dc=False):
    grid = rustpower.PowerGrid.from_pandapower(net, qlim=True)
    grid.enable_dcpf_init(dc)
    return grid


def generator_q(grid, net):
    """Generation = network injection minus the specified non-generator injection.

    Shunts are already part of Ybus, so their Q must not be added again.
    """
    v = grid.v
    q = (v * np.conj(grid.y_bus @ v)).imag * net.sn_mva
    for table, sign in ((net.load, 1), (net.sgen, -1)):
        active = table[table.in_service]
        np.add.at(q, active.bus.to_numpy(dtype=int),
                  sign * active.q_mvar.to_numpy() * active.scaling.to_numpy())
    return q


def limited_case():
    net = pn.case9()
    net.gen.loc[0, "min_q_mvar"] = -300.0
    net.gen.loc[0, "max_q_mvar"] = 5.0
    return net


class QlimPipeline(unittest.TestCase):
    def assert_constraints(self, grid, net):
        self.assertTrue(grid.converged)
        q = generator_q(grid, net)
        slack = set(net.ext_grid.loc[net.ext_grid.in_service, "bus"])
        active = net.gen[net.gen.in_service]
        slack.update(active.loc[active.slack, "bus"])
        for bus, gens in active.groupby("bus"):
            if bus in slack:
                continue
            lower, upper = gens.min_q_mvar.sum(), gens.max_q_mvar.sum()
            self.assertGreaterEqual(q[bus], lower - 1e-3, f"Qmin at bus {bus}")
            self.assertLessEqual(q[bus], upper + 1e-3, f"Qmax at bus {bus}")
            if abs(abs(grid.v[bus]) - gens.vm_pu.iloc[0]) > 1e-6:
                self.assertLess(min(abs(q[bus] - lower), abs(q[bus] - upper)),
                                1e-3, f"voltage relaxed away from Q bound at bus {bus}")

    def test_cold_and_repeated_ac_and_dc_initialization(self):
        for network in (pn.case39, pn.case1354pegase, pn.case9241pegase):
            net = network()
            for dc in (False, True):
                with self.subTest(network=network.__name__, dc=dc):
                    grid = make_grid(net, dc)
                    self.assertTrue(grid.solve(tol=1e-9).converged)
                    self.assert_constraints(grid, net)
                    previous = grid.v.copy()
                    self.assertTrue(grid.solve(tol=1e-9).converged)
                    self.assert_constraints(grid, net)
                    np.testing.assert_allclose(grid.v, previous, atol=1e-7, rtol=0)
                    updated = deepcopy(net)
                    row = net.load.index[0]
                    bus = int(net.load.at[row, "bus"])
                    handle = grid.loads(bus=bus)[0]
                    for scale in (0.99, 1.01):
                        value = float(net.load.at[row, "p_mw"]) * scale
                        handle.p_mw = value
                        updated.load.at[row, "p_mw"] = value
                        self.assertTrue(grid.solve(tol=1e-9).converged)
                        self.assert_constraints(grid, updated)

    def test_voltage_edit_preserves_bound_until_explicit_initialization(self):
        for dc in (False, True):
            with self.subTest(dc=dc):
                net = limited_case()
                grid = make_grid(net, dc)
                self.assertTrue(grid.solve(tol=1e-10).converged)
                bus = int(net.gen.at[0, "bus"])
                self.assertAlmostEqual(generator_q(grid, net)[bus], 5.0, places=5)
                self.assertGreater(abs(abs(grid.v[bus]) - net.gen.at[0, "vm_pu"]), 1e-4)
                previous = grid.v.copy()
                net.gen.at[0, "vm_pu"] = 0.95
                grid.gen(bus=bus).vm_pu = 0.95
                # This also covers an external preparation call made immediately
                # after a Q-limit outer loop, before another App.update().
                grid.compute_dcpf_v()
                self.assertTrue(grid.solve(tol=1e-10).converged)
                np.testing.assert_allclose(grid.v, previous, atol=1e-8, rtol=0)
                self.assertAlmostEqual(generator_q(grid, net)[bus], 5.0, places=5)
                self.assert_constraints(grid, net)
                # Only explicit initialization restores the original bus types.
                grid.init_pf()
                self.assertTrue(grid.solve(tol=1e-10).converged)
                fresh = make_grid(net, dc)
                self.assertTrue(fresh.solve(tol=1e-10).converged)
                np.testing.assert_allclose(grid.v, fresh.v, atol=1e-8, rtol=0)
                self.assertAlmostEqual(abs(grid.v[bus]), 0.95, places=8)
                self.assert_constraints(grid, net)

    def test_load_changes_preserve_the_previous_bound(self):
        net = limited_case()
        grid = make_grid(net)
        self.assertTrue(grid.solve(tol=1e-10).converged)
        bus = int(net.gen.at[0, "bus"])
        self.assertAlmostEqual(generator_q(grid, net)[bus], 5.0, places=5)
        for row in net.load.index:
            net.load.at[row, "p_mw"] *= 0.5
            net.load.at[row, "q_mvar"] = 0.0
            handle = grid.loads(bus=int(net.load.at[row, "bus"]))[0]
            handle.p_mw = float(net.load.at[row, "p_mw"])
            handle.q_mvar = 0.0
        self.assertTrue(grid.solve(tol=1e-10).converged)
        self.assertAlmostEqual(generator_q(grid, net)[bus], 5.0, places=5)
        self.assert_constraints(grid, net)
        # Independent fixed-PQ reference: replace the limited generator by a
        # fixed injection, so pandapower does not restore its voltage constraint.
        pp.create_sgen(net, bus=bus, p_mw=float(net.gen.at[0, "p_mw"]), q_mvar=5.0)
        net.gen.at[0, "in_service"] = False
        pp.runpp(net, init="flat", tolerance_mva=1e-8, lightsim2grid=False)
        reference = net.res_bus.vm_pu.to_numpy() * np.exp(
            1j * np.deg2rad(net.res_bus.va_degree.to_numpy()))
        np.testing.assert_allclose(grid.v, reference, atol=1e-8, rtol=0)

    def test_same_bus_generators_share_the_q_band(self):
        net = pn.case9()
        net.gen.at[0, "vm_pu"] = 1.03
        plain = rustpower.PowerGrid.from_pandapower(net)
        self.assertTrue(plain.solve(tol=1e-10).converged)
        bus = int(net.gen.at[0, "bus"])
        q = generator_q(plain, net)[bus]
        self.assertGreater(q, 5.0)
        net.gen.at[0, "p_mw"] /= 2
        net.gen.at[0, "max_q_mvar"] = (q + 5.0) / 2
        net.gen.loc[net.gen.index.max() + 1] = net.gen.loc[0].copy()
        grid = make_grid(net)
        self.assertTrue(grid.solve(tol=1e-10).converged)
        np.testing.assert_allclose(grid.v, plain.v, atol=1e-8, rtol=0)
        self.assert_constraints(grid, net)

    def test_offline_generator_cannot_clamp_online_generator(self):
        net = pn.case9()
        net.gen.at[0, "vm_pu"] = 1.03
        plain = rustpower.PowerGrid.from_pandapower(net)
        self.assertTrue(plain.solve(tol=1e-10).converged)
        row = net.gen.index.max() + 1
        net.gen.loc[row] = net.gen.loc[0].copy()
        net.gen.loc[row, ["min_q_mvar", "max_q_mvar", "in_service"]] = [0.0, 0.0, False]
        grid = make_grid(net)
        self.assertTrue(grid.solve(tol=1e-10).converged)
        np.testing.assert_allclose(grid.v, plain.v, atol=1e-8, rtol=0)

    def test_full_rebuild_and_disabled_cache_preserve_constraints(self):
        net = limited_case()
        grid = make_grid(net)
        self.assertTrue(grid.solve(tol=1e-10).converged)
        expected = grid.v.copy()
        grid.clear_cache()
        self.assertTrue(grid.solve(tol=1e-10).converged)
        np.testing.assert_allclose(grid.v, expected, atol=1e-8)
        grid.enable_cache(False)
        grid.init_pf()
        self.assertTrue(grid.solve(tol=1e-10).converged)
        np.testing.assert_allclose(grid.v, expected, atol=1e-8)
        self.assert_constraints(grid, net)

    def test_nonconvergence_returns_a_report(self):
        grid = make_grid(limited_case())
        self.assertFalse(grid.solve(max_iter=0).converged)


if __name__ == "__main__":
    unittest.main(verbosity=2)
