import unittest

import numpy as np
import pandapower.networks as networks
import rustpower as rp


class CpfApiTests(unittest.TestCase):
    def test_ieee39_curve_and_result_arrays(self):
        grid = rp.PowerGrid.from_pandapower(networks.case39())
        report = grid.run_cpf()
        self.assertTrue(report.passed_nose)
        self.assertEqual(report.stop_reason, "nose-passed")
        self.assertIsNotNone(report.nose)
        self.assertGreater(2.14, report.nose.lambda_)
        self.assertLess(2.13, report.nose.lambda_)
        self.assertEqual(report.voltages.shape, (len(report.lambda_values), 39))
        self.assertEqual(report.corrector_iterations.shape, report.lambda_values.shape)
        self.assertTrue(np.iscomplexobj(report.voltages))
        self.assertTrue(np.isfinite(report.voltages).all())
        self.assertGreater(report.stats["factorizations"], 0)
        self.assertGreaterEqual(report.stats["elapsed_seconds"], 0.0)

    def test_truncated_curve_has_no_nose(self):
        grid = rp.PowerGrid.from_pandapower(networks.case39())
        report = grid.run_cpf(max_points=1)
        self.assertFalse(report.passed_nose)
        self.assertIsNone(report.nose)
        self.assertEqual(report.stop_reason, "max-points-reached")
        self.assertEqual(report.lambda_max, 1.0)

    def test_invalid_options(self):
        grid = rp.PowerGrid.from_pandapower(networks.case39())
        with self.assertRaises(ValueError):
            grid.run_cpf(step0=0.0)
        with self.assertRaises(ValueError):
            grid.run_cpf(step_shrink=1.0)
        with self.assertRaises(ValueError):
            grid.run_cpf(base_max_iter=0)


if __name__ == "__main__":
    unittest.main()
