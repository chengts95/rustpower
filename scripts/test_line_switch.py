"""Line-switch regressions against pandapower. Requires a local Python build."""
import unittest

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import rustpower as rp


class LineSwitchTests(unittest.TestCase):
    def assert_matches(self, net, switch_model="merge"):
        pp.runpp(net, init="flat", tolerance_mva=1e-10)
        grid = rp.PowerGrid.from_pandapower(net, switch_model=switch_model)
        grid.solve(tol=1e-10)
        self.assertTrue(grid.converged)
        buses = grid.res_bus.set_index("bus_id").reindex(net.bus.index)
        np.testing.assert_allclose(buses.vm_pu, net.res_bus.vm_pu, atol=1e-8, rtol=0)
        np.testing.assert_allclose(buses.va_degree, net.res_bus.va_degree, atol=1e-6, rtol=0)
        return grid

    def test_cigre_open_line_switches(self):
        net = pn.create_cigre_network_mv(with_der=False)
        for model in ("merge", "admittance"):
            with self.subTest(model=model):
                self.assert_matches(net, switch_model=model)

    def test_open_ends_and_charging(self):
        for ends in ((), ("from_bus",), ("to_bus",), ("from_bus", "to_bus")):
            with self.subTest(ends=ends):
                net = pn.case14()
                # Deliberately amplify charging: disabling the whole line is wrong.
                net.line.at[0, "c_nf_per_km"] = 4000.0
                for end in ends:
                    pp.create_switch(net, int(net.line.at[0, end]), 0, et="l", closed=False)
                grid = self.assert_matches(net)
                line = net.line.loc[0]
                result = grid.res_line
                row = result[(result.from_bus == line.from_bus) & (result.to_bus == line.to_bus)].iloc[0]
                expected = net.res_line.loc[0]
                for col in ("p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar", "i_from_ka", "i_to_ka", "vm_from_pu", "vm_to_pu", "va_from_degree", "va_to_degree"):
                    if np.isnan(expected[col]):
                        # Pandapower reports NaN current for a fully disconnected
                        # line; RustPower publishes zero terminal current.
                        self.assertEqual(row[col], 0.0, msg=col)
                    else:
                        self.assertAlmostEqual(row[col], expected[col], delta=1e-6, msg=col)

    def test_sparse_and_reordered_line_indices(self):
        net = pn.create_cigre_network_mv(with_der=False)
        mapping = {int(i): 100 + int(i) * 3 for i in net.line.index}
        net.line.index = [mapping[int(i)] for i in net.line.index]
        line_switches = net.switch.et == "l"
        net.switch.loc[line_switches, "element"] = net.switch.loc[line_switches, "element"].map(mapping)
        net.line = net.line.iloc[::-1]
        self.assert_matches(net)

    def test_multiple_switches_at_same_end(self):
        net = pn.case14()
        bus = int(net.line.at[0, "from_bus"])
        pp.create_switch(net, bus, 0, et="l", closed=False)
        pp.create_switch(net, bus, 0, et="l", closed=True)
        self.assert_matches(net)

    def test_service_state_and_reimport(self):
        net = pn.case14()
        pp.create_switch(net, int(net.line.at[0, "from_bus"]), 0, et="l", closed=False)
        grid = self.assert_matches(net)
        handle = grid.line(int(net.line.at[0, "from_bus"]), int(net.line.at[0, "to_bus"]))
        handle.in_service = False
        grid.solve(tol=1e-10)
        net.line.at[0, "in_service"] = False
        pp.runpp(net, init="flat", tolerance_mva=1e-10)
        np.testing.assert_allclose(grid.res_bus.set_index("bus_id").vm_pu, net.res_bus.vm_pu, atol=1e-8, rtol=0)
        # A closed switch must never revive an independently disabled line.
        net.switch.closed = True
        self.assert_matches(net)
        # Reimporting updated topology must also remove stale open-end patches.
        net.line.at[0, "in_service"] = True
        grid.from_pp_net(net)
        grid.solve(tol=1e-10)
        pp.runpp(net, init="flat", tolerance_mva=1e-10)
        np.testing.assert_allclose(grid.res_bus.set_index("bus_id").vm_pu, net.res_bus.vm_pu, atol=1e-8, rtol=0)

    @unittest.skipUnless("arrow" in rp.features(), "requires Parquet support")
    def test_archive_preserves_switch_target(self):
        net = pn.create_cigre_network_mv(with_der=False)
        grid = self.assert_matches(net)
        restored = rp.PowerGrid(switch_model="merge")
        restored.load_parquet_case(grid.get_parquet_case())
        restored.solve(tol=1e-10)
        np.testing.assert_allclose(restored.res_bus.set_index("bus_id").vm_pu, net.res_bus.vm_pu, atol=1e-8, rtol=0)

    def test_closed_transformer_switch_targets_and_reimport(self):
        net = pn.create_cigre_network_mv(with_der=False)
        mapping = {int(i): 100 + 7 * int(i) for i in net.trafo.index}
        net.trafo.index = [mapping[int(i)] for i in net.trafo.index]
        net.trafo = net.trafo.iloc[::-1]
        mask = net.switch.et == "t"
        net.switch.loc[mask, "element"] = net.switch.loc[mask, "element"].map(mapping)
        grid = self.assert_matches(net)
        self.assertEqual(grid.validate(), [])
        old = grid.bus(0)
        switch = net.switch.index[mask][0]
        target = int(net.switch.at[switch, "element"])
        for field, invalid, pattern in (
            ("element", 999, "missing transformer"),
            ("bus", 14, "transformer terminal"),
        ):
            with self.subTest(field=field):
                previous = net.switch.at[switch, field]
                net.switch.at[switch, field] = invalid
                with self.assertRaisesRegex(ValueError, pattern):
                    grid.from_pp_net(net)
                self.assertEqual(old.id, 0)
                self.assertTrue(grid.solve(tol=1e-10).converged)
                net.switch.at[switch, field] = previous
        # The whole network, including closed transformer switches, still loads.
        grid.from_pp_net(net)
        self.assertTrue(grid.solve(tol=1e-10).converged)
        self.assertEqual(int(net.switch.at[switch, "element"]), target)

    def test_invalid_switch_references(self):
        net = pn.case14()
        pp.create_switch(net, int(net.line.at[0, "from_bus"]), 0, et="l", closed=False)
        net.switch.at[0, "element"] = 999
        with self.assertRaisesRegex(ValueError, "missing line"):
            rp.PowerGrid.from_pandapower(net, switch_model="merge")
        net.switch.at[0, "element"] = 0
        net.switch.at[0, "bus"] = 13
        with self.assertRaisesRegex(ValueError, "not an endpoint"):
            rp.PowerGrid.from_pandapower(net, switch_model="merge")


if __name__ == "__main__":
    unittest.main()
