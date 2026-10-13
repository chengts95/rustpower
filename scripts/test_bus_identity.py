"""Core identity, dense offsets, and transactional graph regressions."""
import unittest

import numpy as np
import pandapower as pp
import rustpower as rp


class BusIdentityTests(unittest.TestCase):
    def native_grid(self, gap=True):
        grid = rp.PowerGrid()
        with grid.edit() as edit:
            a, ha = edit.add_bus(110.0)
            if gap:
                middle, hm = edit.add_bus(110.0)
            b, hb = edit.add_bus(110.0)
            slack = edit.add_ext_grid(a)
            line = edit.add_line(a, b, 1.0, max_i_ka=1.0)
            load = edit.add_load(b, 1.0, 0.1)
        if gap:
            with grid.edit() as edit:
                edit.remove(hm)
        return grid, ha, hb, line, load

    def test_remove_unused_middle_bus_and_publish_stable_labels(self):
        grid, a, b, line, load = self.native_grid()
        fresh, *_ = self.native_grid(gap=False)
        self.assertEqual(b.id, 2)
        self.assertEqual(b.matrix_index, 1)
        self.assertEqual(line.to_bus, 2)
        self.assertTrue(grid.solve(tol=1e-10).converged)
        self.assertTrue(fresh.solve(tol=1e-10).converged)
        self.assertEqual(set(grid.res_bus.bus_id), {0, 2})
        np.testing.assert_allclose(grid.v, fresh.v, atol=1e-10)
        np.testing.assert_allclose(grid.res_line.p_from_mw, fresh.res_line.p_from_mw, atol=1e-10)
        np.testing.assert_allclose(grid.compute_dcpf_v(), fresh.compute_dcpf_v(), atol=1e-10)
        grid.apply_dcpf_init()
        self.assertTrue(grid.solve(tol=1e-10).converged)
        load.p_mw = 2.0
        self.assertEqual(grid.solve(tol=1e-10).rebuild, "incremental")
        self.assertAlmostEqual(float(grid.res_bus.set_index("bus_id").loc[2, "p_mw"]), 2.0, delta=1e-8)

    def test_invalid_references_reject_commit_and_roll_back_additions(self):
        for kind, label in (("line", 999), ("line", -1), ("load", 999)):
            with self.subTest(kind=kind, label=label):
                grid = rp.PowerGrid()
                with self.assertRaisesRegex(ValueError, "missing bus"):
                    with grid.edit() as edit:
                        a, old_handle = edit.add_bus(110.0)
                        edit.add_ext_grid(a)
                        if kind == "line":
                            edit.add_line(a, label, 1.0)
                        else:
                            edit.add_load(label, 1.0, 0.1)
                self.assertEqual(grid.n_bus, 0)
                with grid.edit() as edit:
                    replacement, new_handle = edit.add_bus(110.0)
                    edit.add_ext_grid(replacement)
                self.assertEqual(replacement, 0)
                with self.assertRaises(ValueError):
                    _ = old_handle.id
                self.assertTrue(grid.solve().converged)

    def test_remove_bus_keeps_connected_elements_disabled(self):
        grid, a, b, line, load = self.native_grid()
        self.assertTrue(grid.solve().converged)
        with grid.edit() as edit:
            edit.remove(b)
        self.assertEqual(grid.n_bus, 1)
        self.assertEqual(line.to_bus, 2)
        self.assertEqual(load.bus, 2)
        self.assertFalse(line.in_service)
        self.assertFalse(load.in_service)
        self.assertTrue(grid.solve().converged)
        self.assertEqual(float(grid.res_line.i_from_ka.iloc[0]), 0.0)
        for element in (line, load):
            with self.assertRaisesRegex(ValueError, "deleted bus entity"):
                element.in_service = True
            self.assertFalse(element.in_service)
        load.p_mw = 4.0
        self.assertEqual(load.p_mw, 4.0)
        self.assertTrue(grid.solve().converged)
        self.assertAlmostEqual(float(grid.res_bus.p_mw.iloc[0]), 0.0)

    def test_abort_bus_removal_keeps_connections_enabled(self):
        grid, a, b, line, load = self.native_grid()
        with self.assertRaises(RuntimeError):
            with grid.edit() as edit:
                edit.remove(b)
                raise RuntimeError("cancel")
        self.assertEqual(b.id, 2)
        self.assertTrue(line.in_service)
        self.assertTrue(load.in_service)
        self.assertTrue(grid.solve().converged)

    def test_failed_commit_does_not_disable_connections(self):
        grid, a, b, line, load = self.native_grid()
        with self.assertRaisesRegex(ValueError, "missing bus"):
            with grid.edit() as edit:
                edit.remove(b)
                edit.add_load(999, 1.0, 0.1)
        self.assertEqual(b.id, 2)
        self.assertTrue(line.in_service)
        self.assertTrue(load.in_service)
        self.assertTrue(grid.solve().converged)

    @unittest.skipUnless("arrow" in rp.features(), "requires Parquet support")
    def test_archive_keeps_disabled_elements_after_bus_removal(self):
        grid, a, b, line, load = self.native_grid()
        with grid.edit() as edit:
            edit.remove(b)
        self.assertTrue(grid.solve().converged)
        restored = rp.PowerGrid()
        restored.load_parquet_case(grid.get_parquet_case())
        kept_load = restored.load(bus=2)
        kept_line = restored.line(0, 2)
        self.assertFalse(kept_load.in_service)
        self.assertFalse(kept_line.in_service)
        self.assertTrue(restored.solve().converged)
        kept_load.p_mw = 3.0
        self.assertEqual(kept_load.p_mw, 3.0)
        self.assertTrue(restored.solve().converged)
        with self.assertRaisesRegex(ValueError, "missing bus"):
            kept_load.in_service = True

    def test_import_disabled_elements_with_missing_bus(self):
        net = pp.create_empty_network(sn_mva=100.0)
        live = pp.create_bus(net, 110.0, index=100)
        missing = pp.create_bus(net, 110.0, index=999)
        pp.create_ext_grid(net, live)
        pp.create_load(net, missing, 1.0, 0.1, in_service=False)
        pp.create_line_from_parameters(net, live, missing, 1.0, 0.1, 0.1, 0.0, 1.0, in_service=False)
        net.bus = net.bus.drop(missing)
        grid = rp.PowerGrid.from_pandapower(net)
        self.assertTrue(grid.solve().converged)
        load = grid.loads()[0]
        self.assertFalse(load.in_service)
        self.assertLess(load.bus, -1)
        with self.assertRaisesRegex(ValueError, "missing bus"):
            load.in_service = True
        self.assertFalse(load.in_service)

    def test_remove_bus_keeps_generator_transformer_and_shunt(self):
        grid, a, b, line, load = self.native_grid()
        with grid.edit() as edit:
            gen = edit.add_gen(b.id, 0.1, 1.0)
            trafo = edit.add_trafo(a.id, b.id, sn_mva=10.0, vn_hv_kv=110.0, vn_lv_kv=110.0)
            shunt = edit.add_shunt(b.id, 0.1)
        self.assertTrue(grid.solve().converged)
        with grid.edit() as edit:
            edit.remove(b)
        self.assertFalse(gen.in_service)
        gen.p_mw = 0.2
        gen.vm_pu = 1.01
        self.assertEqual(gen.p_mw, 0.2)
        self.assertTrue(grid.solve().converged)
        # Handles still refer to retained elements and remain removable.
        with grid.edit() as edit:
            edit.remove(gen)
            edit.remove(trafo)
            edit.remove(shunt)
        self.assertTrue(grid.solve().converged)

    def test_optional_validation_reports_disabled_connections_without_changes(self):
        grid, a, b, line, load = self.native_grid()
        self.assertEqual(grid.validate(), [])
        with grid.edit() as edit:
            edit.remove(b)
        problems = grid.validate()
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("TargetBus" in problem for problem in problems))
        self.assertTrue(any("ToBus" in problem for problem in problems))
        self.assertFalse(line.in_service)
        self.assertFalse(load.in_service)
        # Reporting does not prevent solving or saving an incomplete model.
        self.assertTrue(grid.solve().converged)
        self.assertTrue(grid.get_parquet_case())
        self.assertEqual(grid.validate(), problems)
        with grid.edit() as edit:
            edit.remove(line)
            edit.remove(load)
        self.assertEqual(grid.validate(), [])

    def test_abort_preserves_removals(self):
        grid, a, b, line, load = self.native_grid()
        with self.assertRaisesRegex(RuntimeError, "abort"):
            with grid.edit() as edit:
                edit.remove(line)
                edit.remove(load)
                edit.remove(b)
                raise RuntimeError("abort")
        self.assertEqual(grid.n_bus, 2)
        self.assertEqual(line.to_bus, 2)
        self.assertTrue(grid.solve().converged)

    def test_foreign_grid_handles_are_rejected(self):
        one, a, *_ = self.native_grid()
        two, *_ = self.native_grid()
        with self.assertRaisesRegex(ValueError, "another grid"):
            with two.edit() as edit:
                edit.remove(a)
        self.assertEqual(one.n_bus, 2)
        self.assertEqual(two.n_bus, 2)
        self.assertTrue(two.solve().converged)

    def test_sparse_native_archive_rebuilds_entity_references(self):
        grid, a, b, line, load = self.native_grid()
        self.assertTrue(grid.solve(tol=1e-10).converged)
        restored = rp.PowerGrid()
        restored.load_parquet_case(grid.get_parquet_case())
        self.assertTrue(restored.solve(tol=1e-10).converged)
        self.assertEqual(restored.bus(2).matrix_index, 1)
        self.assertIsNone(restored.bus(2).pp_id)
        np.testing.assert_allclose(restored.v, grid.v, atol=1e-10)
        np.testing.assert_allclose(restored.res_line.i_from_ka, grid.res_line.i_from_ka, atol=1e-10)

    def test_remove_shunt_and_its_bus_removes_derived_ground_branch(self):
        grid, a, b, line, load = self.native_grid()
        with grid.edit() as edit:
            shunt = edit.add_shunt(b.id, 0.1)
        self.assertTrue(grid.solve().converged)
        with grid.edit() as edit:
            edit.remove(line)
            edit.remove(load)
            edit.remove(shunt)
            edit.remove(b)
        self.assertEqual(grid.n_bus, 1)
        self.assertTrue(grid.solve().converged)

    def test_bad_archive_preserves_existing_grid(self):
        grid, a, b, line, load = self.native_grid()
        self.assertTrue(grid.solve().converged)
        with self.assertRaises(RuntimeError):
            grid.load_parquet_case(b"invalid zip")
        self.assertEqual(grid.n_bus, 2)
        self.assertEqual(b.id, 2)
        self.assertTrue(grid.solve().converged)

    def test_same_size_replacement_refreshes_bus_scales(self):
        net = pp.create_empty_network(sn_mva=100)
        a = pp.create_bus(net, vn_kv=110)
        b = pp.create_bus(net, vn_kv=110)
        pp.create_ext_grid(net, a)
        pp.create_load(net, b, p_mw=1, q_mvar=0.1)
        pp.create_line_from_parameters(net, a, b, 1, 0.1, 0.1, 0, 1)
        grid = rp.PowerGrid.from_pandapower(net)
        grid.solve(tol=1e-10)
        _ = grid.res_line
        net.bus.vn_kv = 220.0
        grid.from_pp_net(net)
        grid.solve(tol=1e-10)
        fresh = rp.PowerGrid.from_pandapower(net)
        fresh.solve(tol=1e-10)
        np.testing.assert_allclose(grid.res_line.p_from_mw, fresh.res_line.p_from_mw, atol=1e-10)
        np.testing.assert_allclose(grid.res_line.i_from_ka, fresh.res_line.i_from_ka, atol=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
