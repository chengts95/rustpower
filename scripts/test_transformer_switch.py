"""Two-winding transformer terminal switching against pandapower."""
import unittest

import numpy as np
import pandapower as pp
import rustpower as rp


class TransformerSwitchTests(unittest.TestCase):
    def network(self, core=True, tap_side=None):
        net = pp.create_empty_network(sn_mva=100.0)
        a = pp.create_bus(net, 110.0, index=100)
        b = pp.create_bus(net, 20.0, index=700)
        pp.create_ext_grid(net, a, vm_pu=1.02)
        args = dict(sn_mva=40.0, vn_hv_kv=110.0, vn_lv_kv=20.0,
                    vk_percent=10.0, vkr_percent=0.5, pfe_kw=80.0 if core else 0.0,
                    i0_percent=0.8 if core else 0.0, shift_degree=17.0,
                    parallel=2)
        pp.create_transformer_from_parameters(net, a, b, index=901, **args)
        pp.create_transformer_from_parameters(net, a, b, index=903, **args)
        pp.create_load(net, b, 5.0, 1.0)
        if tap_side is not None:
            for key, value in dict(tap_side=tap_side, tap_pos=2.0, tap_neutral=0.0,
                                   tap_step_percent=1.25, tap_changer_type="Ratio").items():
                net.trafo.at[901, key] = value
        return net

    def assert_matches(self, net, grid=None):
        pp.runpp(net, init="flat", tolerance_mva=1e-10)
        if grid is None:
            grid = rp.PowerGrid.from_pandapower(net, switch_model="merge")
        else:
            grid.from_pp_net(net)
        self.assertTrue(grid.solve(tol=1e-10).converged)
        result = grid.res_bus.set_index("pp_bus_id").reindex(net.bus.index)
        np.testing.assert_allclose(result.vm_pu, net.res_bus.vm_pu, atol=1e-8, rtol=0)
        np.testing.assert_allclose(result.va_degree, net.res_bus.va_degree, atol=1e-6, rtol=0)
        trafos = grid.res_trafo.set_index("trafo_id")
        for label in net.trafo.index:
            if not net.trafo.at[label, "in_service"]:
                # RustPower publishes zero device results when out of service.
                np.testing.assert_array_equal(trafos.loc[label, net.res_trafo.columns], 0.0)
                continue
            for column, expected in net.res_trafo.loc[label].items():
                value = trafos.loc[label, column]
                if np.isnan(expected):
                    # Same convention as disconnected lines: publish zeros.
                    self.assertEqual(value, 0.0, msg=column)
                else:
                    self.assertAlmostEqual(value, expected, delta=1e-6, msg=column)
        return grid

    def test_open_terminals_core_taps_and_reclosing(self):
        for core, tap in ((False, None), (True, None), (True, "hv"), (True, "lv")):
            for ends in ((100,), (700,), (100, 700)):
                with self.subTest(core=core, tap=tap, ends=ends):
                    net = self.network(core, tap)
                    for bus in ends:
                        pp.create_switch(net, bus, 901, et="t", closed=False)
                    # Any open switch wins over another closed switch at that end.
                    pp.create_switch(net, ends[0], 901, et="t", closed=True)
                    grid = self.assert_matches(net)
                    # Repeated initialization must not reduce the patch twice.
                    voltage = grid.v.copy()
                    grid.init_pf()
                    self.assertTrue(grid.solve(tol=1e-10).converged)
                    np.testing.assert_allclose(grid.v, voltage, atol=1e-10)
                    load = grid.loads()[0]
                    load.p_mw *= 1.01
                    self.assertEqual(grid.solve(tol=1e-10).rebuild, "incremental")
                    # Full reimport restores the normal patch on closing/removal.
                    net.switch.closed = True
                    self.assert_matches(net, grid)
                    net.switch = net.switch.iloc[:0]
                    self.assert_matches(net, grid)

    @unittest.skipUnless("arrow" in rp.features(), "requires Parquet support")
    def test_archive_dc_and_out_of_service(self):
        net = self.network(tap_side="lv")
        pp.create_switch(net, 100, 901, et="t", closed=False)
        grid = self.assert_matches(net)
        restored = rp.PowerGrid(switch_model="merge")
        restored.load_parquet_case(grid.get_parquet_case())
        self.assertTrue(restored.solve(tol=1e-10).converged)
        np.testing.assert_allclose(restored.res_trafo.set_index("trafo_id").sort_index(),
                                   grid.res_trafo.set_index("trafo_id").sort_index(), atol=1e-9)
        self.assertEqual(restored.compute_dcpf_v().shape, (2,))
        restored.apply_dcpf_init()
        self.assertTrue(restored.solve(tol=1e-10).converged)
        net.trafo.at[901, "in_service"] = False
        self.assert_matches(net, grid)
        net.switch.closed = True
        self.assert_matches(net, grid)


if __name__ == "__main__":
    unittest.main()
