import unittest

import numpy as np
import pandapower.networks as networks
import rustpower as rp


class LmpfApiTests(unittest.TestCase):
    def test_prepared_study_and_reuse(self):
        grid = rp.PowerGrid.from_pandapower(networks.case39())
        study = grid.prepare_lmpf()
        self.assertIs(study.owner, grid)
        first = study.run()
        self.assertTrue(first.converged)
        self.assertEqual(first.status, "converged")
        self.assertLess(first.residual_norm, 1e-6)
        self.assertEqual(first.voltage.shape, (39,))
        self.assertTrue(np.isfinite(first.voltage).all())
        second = study.run()
        np.testing.assert_allclose(first.voltage, second.voltage, rtol=0.0, atol=1e-12)

    def test_gauss_newton_option_and_reset(self):
        grid = rp.PowerGrid.from_pandapower(networks.case39())
        study = grid.prepare_lmpf()
        report = study.run(exact=False)
        self.assertTrue(report.converged)
        study.reset()
        self.assertTrue(study.run().converged)

    def test_invalid_options(self):
        study = rp.PowerGrid.from_pandapower(networks.case39()).prepare_lmpf()
        with self.assertRaises(ValueError):
            study.run(max_iter=0)
        with self.assertRaises(ValueError):
            study.run(initial_mu=float("nan"))
        with self.assertRaises(ValueError):
            study.run(max_trials=0)


if __name__ == "__main__":
    unittest.main()
