"""Explicit switch-model selection and physical Python voltage arrays."""
import unittest
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import rustpower as rp


class SwitchModelTests(unittest.TestCase):
    def bus_switch_net(self, z=0.0):
        net = pp.create_empty_network(sn_mva=100.0)
        a = pp.create_bus(net, 110.0, index=100)
        b = pp.create_bus(net, 110.0, index=300)
        c = pp.create_bus(net, 110.0, index=900)
        pp.create_ext_grid(net, a)
        pp.create_switch(net, a, b, et="b", closed=True, z_ohm=z)
        pp.create_line_from_parameters(net, b, c, 1.0, 0.1, 0.2, 0.0, 1.0)
        pp.create_load(net, c, 1.0, 0.2)
        return net

    def test_no_switches_need_no_selection(self):
        self.assertTrue(rp.PowerGrid.from_pandapower(pn.case14()).solve().converged)

    def test_switches_require_choice_including_open_or_line(self):
        for kind, closed in (("b", True), ("b", False), ("l", True), ("l", False)):
            with self.subTest(kind=kind, closed=closed):
                net = pn.case14()
                bus = int(net.line.at[0, "from_bus"])
                element = int(net.line.at[0, "to_bus"]) if kind == "b" else 0
                pp.create_switch(net, bus, element, et=kind, closed=closed)
                with self.assertRaisesRegex(ValueError, "explicitly select switch_model"):
                    rp.PowerGrid.from_pandapower(net)

    def test_cigre_lv_and_physical_voltage_arrays(self):
        net = pn.create_cigre_network_lv()
        pp.toolbox.reindex_buses(net, {i: 100 + 7 * int(i) for i in net.bus.index})
        net.bus = net.bus.iloc[::-1]
        pp.runpp(net, tolerance_mva=1e-10)
        grid = rp.PowerGrid.from_pandapower(net, switch_model="merge")
        self.assertTrue(grid.solve(tol=1e-10).converged)
        result = grid.res_bus.set_index("pp_bus_id").reindex(net.bus.index)
        np.testing.assert_allclose(result.vm_pu, net.res_bus.vm_pu, atol=1e-8, rtol=0)
        np.testing.assert_allclose(result.va_degree, net.res_bus.va_degree, atol=1e-6, rtol=0)
        self.assertEqual(len(grid.v), len(net.bus))
        np.testing.assert_allclose(abs(grid.v), result.vm_pu, atol=1e-8, rtol=0)
        dc = grid.compute_dcpf_v()
        self.assertEqual(len(dc), len(net.bus))
        self.assertTrue(grid.solve(v_init=dc, tol=1e-10).converged)
        grid.apply_dcpf_init()
        self.assertTrue(grid.solve(tol=1e-10).converged)
        load = grid.loads()[0]
        load.p_mw *= 1.01
        report = grid.solve(tol=1e-10)
        self.assertTrue(report.converged)
        self.assertEqual(report.rebuild, "incremental")
        self.assertEqual(len(grid.v), len(net.bus))

    def test_both_models_finite_resistance(self):
        net = self.bus_switch_net(z=0.1)
        pp.runpp(net, switch_rx_ratio=1e12, tolerance_mva=1e-10)
        for model in ("merge", "admittance"):
            with self.subTest(model=model):
                grid = rp.PowerGrid.from_pandapower(net, switch_model=model)
                self.assertTrue(grid.solve(tol=1e-10).converged)
                r = grid.res_bus.set_index("pp_bus_id").reindex(net.bus.index)
                np.testing.assert_allclose(r.vm_pu, net.res_bus.vm_pu, atol=1e-9, rtol=0)
                np.testing.assert_allclose(r.va_degree, net.res_bus.va_degree, atol=1e-7, rtol=0)

    def test_admittance_ideal_approximation(self):
        grid = rp.PowerGrid.from_pandapower(self.bus_switch_net(), switch_model="admittance")
        self.assertTrue(grid.solve(tol=1e-9).converged)
        self.assertEqual(len(grid.v), 3)
        self.assertLess(abs(grid.v[0] - grid.v[1]), 1e-6)

    def test_first_selection_reuse_and_conflict(self):
        grid = rp.PowerGrid()
        net = self.bus_switch_net()
        grid.from_pp_net(net, switch_model="merge")
        self.assertTrue(grid.solve().converged)
        grid.from_pp_net(net)
        original = grid.bus(0)
        with self.assertRaisesRegex(ValueError, "fixed once selected"):
            grid.from_pp_net(net, switch_model="admittance")
        self.assertEqual(original.id, 0)
        self.assertTrue(grid.solve().converged)
        with self.assertRaisesRegex(ValueError, "must be"):
            rp.PowerGrid(switch_model="auto")

    def test_failed_import_preserves_grid_and_selection(self):
        grid = rp.PowerGrid.from_pandapower(pn.case14())
        original = grid.bus(0)
        net = self.bus_switch_net()
        with self.assertRaisesRegex(ValueError, "explicitly select"):
            grid.from_pp_net(net)
        for impedance in (-1.0, float("nan"), float("inf")):
            net.switch.at[0, "z_ohm"] = impedance
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                grid.from_pp_net(net, switch_model="merge")
        self.assertEqual(original.id, 0)
        self.assertTrue(grid.solve().converged)
        net.switch.at[0, "z_ohm"] = 0.0
        # A failed import must not lock in its requested model.
        grid.from_pp_net(net, switch_model="admittance")
        self.assertTrue(grid.solve().converged)

    def test_unsupported_switch_and_voltage_mismatch(self):
        for kind in ("t", "t3", "unsupported"):
            net = pn.case14()
            pp.create_switch(net, int(net.trafo.at[0, "hv_bus"]), 0, et="t", closed=True)
            net.switch.at[0, "et"] = kind
            with self.assertRaisesRegex(ValueError, "unsupported switch"):
                rp.PowerGrid.from_pandapower(net, switch_model="merge")
        net = self.bus_switch_net()
        net.bus.at[300, "vn_kv"] = 20.0
        with self.assertRaisesRegex(ValueError, "different nominal"):
            rp.PowerGrid.from_pandapower(net, switch_model="merge")

    @unittest.skipUnless("arrow" in rp.features(), "requires Parquet support")
    def test_archive_requires_selection_and_reuses_it(self):
        source = rp.PowerGrid.from_pandapower(self.bus_switch_net(), switch_model="merge")
        source.solve(tol=1e-10)
        snapshot = source.get_parquet_case()
        restored = rp.PowerGrid.from_pandapower(pn.case14())
        old = restored.bus(0)
        with self.assertRaisesRegex(ValueError, "explicitly select"):
            restored.load_parquet_case(snapshot)
        self.assertEqual(old.id, 0)
        restored.load_parquet_case(snapshot, switch_model="merge")
        self.assertTrue(restored.solve(tol=1e-10).converged)
        np.testing.assert_allclose(restored.v, source.v, atol=1e-10)
        restored.load_parquet_case(snapshot)
        self.assertTrue(restored.solve(tol=1e-10).converged)

    def test_csv_case_constructor_requires_selection(self):
        net = self.bus_switch_net()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "switch-case.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for name in ("bus", "line", "load", "ext_grid", "switch"):
                    frame = net[name].copy()
                    # The CSV format uses the original constant-load fields.
                    if name == "load":
                        frame["const_i_percent"] = 0.0
                        frame["const_z_percent"] = 0.0
                    for col in frame.select_dtypes(include="bool"):
                        frame[col] = frame[col].map({True: "true", False: "false"})
                    archive.writestr(name + ".csv", frame.to_csv(index_label="index"))
            with self.assertRaisesRegex(ValueError, "explicitly select"):
                rp.PowerGrid(str(path))
            grid = rp.PowerGrid(str(path), switch_model="merge")
            self.assertTrue(grid.solve(tol=1e-10).converged)
            self.assertEqual(len(grid.v), 3)

    def test_native_network_loading_requires_selection(self):
        net = rp.Network()
        a, b = rp.Bus(index=100), rp.Bus(index=300)
        net.bus = [a, b]
        net.ext_grid = [rp.ExtGrid(bus=100)]
        net.switch = [rp.Switch(bus=100, element=300)]
        grid = rp.PowerGrid()
        with self.assertRaisesRegex(ValueError, "explicitly select"):
            grid.load_network(net)
        grid.load_network(net, switch_model="merge")
        self.assertTrue(grid.solve().converged)
        self.assertEqual(len(grid.v), 2)


if __name__ == "__main__":
    unittest.main()
