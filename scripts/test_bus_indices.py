"""Pandapower labels must never become Rust matrix offsets."""
import unittest

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import rustpower as rp


class BusIndexTests(unittest.TestCase):
    def check_case(self, net, expected=None):
        if expected is None:
            pp.runpp(net, init="flat", tolerance_mva=1e-10)
            expected = net.res_bus
        grid = rp.PowerGrid.from_pandapower(net, switch_model="merge")
        grid.solve(tol=1e-10)
        self.assertTrue(grid.converged)
        result = grid.res_bus
        self.assertEqual(sorted(result.bus_id), list(range(len(net.bus))))
        result = result.set_index("pp_bus_id").reindex(net.bus.index)
        np.testing.assert_allclose(result.vm_pu, expected.vm_pu, atol=1e-8, rtol=0)
        np.testing.assert_allclose(result.va_degree, expected.va_degree, atol=1e-6, rtol=0)
        for internal, original in enumerate(net.bus.index):
            self.assertEqual(grid.bus(internal).pp_id, int(original))
        return grid

    def test_issue_shift_and_gap(self):
        for mapping in ({i: i + 100 for i in range(14)}, {i: i * 3 for i in range(14)}):
            with self.subTest(mapping=mapping):
                net = pn.case14()
                pp.toolbox.reindex_buses(net, mapping)
                self.check_case(net)

    def test_negative_large_and_reordered_labels(self):
        net = pn.case14()
        pp.runpp(net, init="flat", tolerance_mva=1e-10)
        expected = net.res_bus.copy()
        # Above 2**53: preserving labels through floating-point storage is unsafe.
        labels = {i: -(2**53 + i * 3 + 1) if i % 2 else 2**53 + i * 3 + 1 for i in net.bus.index}
        pp.toolbox.reindex_buses(net, labels)
        net.bus = net.bus.iloc[::-1]
        expected.index = [labels[i] for i in expected.index]
        self.check_case(net, expected.reindex(net.bus.index))

    def test_open_line_with_distinct_bus_and_line_namespaces(self):
        net = pn.create_cigre_network_mv(with_der=False)
        # This fixture also contains closed transformer switches. Their model
        # is unsupported; omit them explicitly to exercise line switching.
        self.assertTrue(net.switch.loc[net.switch.et == "t", "closed"].all())
        net.switch = net.switch[net.switch.et != "t"].copy()
        pp.toolbox.reindex_buses(net, {i: i * 11 + 100 for i in net.bus.index})
        lines = {i: i * 3 + 900 for i in net.line.index}
        net.line.index = [lines[i] for i in net.line.index]
        mask = net.switch.et == "l"
        net.switch.loc[mask, "element"] = net.switch.loc[mask, "element"].map(lines)
        self.check_case(net)

    def test_missing_bus_rejected_without_replacing_grid(self):
        grid = self.check_case(pn.case14())
        original = grid.bus(0)
        cases = [("line", "from_bus"), ("trafo", "hv_bus"), ("load", "bus"),
                 ("gen", "bus"), ("ext_grid", "bus")]
        for table, column in cases:
            with self.subTest(table=table):
                net = pn.case14()
                df = net[table]
                df.at[df.index[0], column] = 999
                with self.assertRaisesRegex(ValueError, "missing pandapower bus 999"):
                    grid.from_pp_net(net)
                self.assertEqual(original.id, 0)
                grid.solve()
                self.assertTrue(grid.converged)
        net = pn.case14()
        index = pp.create_switch(net, 0, 1, et="b", closed=True)
        net.switch.at[index, "element"] = 999
        with self.assertRaisesRegex(ValueError, "switch.element.*999"):
            grid.from_pp_net(net)

    def test_duplicate_bus_labels_rejected(self):
        net = pn.case14()
        net.bus.index = [0] * len(net.bus)
        with self.assertRaisesRegex(ValueError, "duplicate pandapower bus index"):
            rp.PowerGrid.from_pandapower(net, switch_model="merge")

    def test_mixed_native_buses_preserve_integer_labels(self):
        net = pn.case14()
        pp.toolbox.reindex_buses(net, {i: 2**53 + i * 3 + 1 for i in net.bus.index})
        grid = rp.PowerGrid.from_pandapower(net, switch_model="merge")
        with grid.edit() as editor:
            internal, native = editor.add_bus(110.0)
        self.assertIsNone(native.pp_id)
        params = grid.get_bus_params().set_index("bus_id")
        self.assertEqual(int(params.loc[0, "pp_bus_id"]), 2**53 + 1)
        self.assertTrue(pd.isna(params.loc[internal, "pp_bus_id"]))

    def test_archive_preserves_original_labels(self):
        net = pn.case14()
        pp.toolbox.reindex_buses(net, {i: i + 100 for i in net.bus.index})
        grid = self.check_case(net)
        restored = rp.PowerGrid()
        restored.load_parquet_case(grid.get_parquet_case())
        restored.solve(tol=1e-10)
        result = restored.res_bus.set_index("pp_bus_id").reindex(net.bus.index)
        np.testing.assert_allclose(result.vm_pu, net.res_bus.vm_pu, atol=1e-8, rtol=0)
        self.assertEqual(restored.bus(0).pp_id, 100)


if __name__ == "__main__":
    unittest.main()
