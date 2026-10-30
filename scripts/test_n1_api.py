"""Python N-1 ownership, sparse output, native retry and column-link regressions.

Build with python,experimental_n1 and run with the desired extension on PYTHONPATH.
"""
import gc
import os
import threading
import unittest

import numpy as np
import pandapower as pp
import pandapower.networks as networks
import rustpower as rp


def grid(net=None):
    return rp.PowerGrid.from_pandapower(networks.case14() if net is None else net)


def first_line(g):
    return g.line(0, 1)


class N1ApiTests(unittest.TestCase):
    def test_selection_sparse_details_and_ownership(self):
        g = grid()
        line = first_line(g)
        study = g.prepare_n1([line, line])
        self.assertEqual(len(study.faults), 1)
        status = study.run(workers=2)
        self.assertEqual(status.summary['passed'], 1)
        self.assertEqual(len(status.events['case_id']), 0)
        detail = study.details(study.faults, workers=2)
        self.assertEqual(detail.events['outcome'], ['Passed'])
        self.assertIsInstance(detail.events['outage_entity'], np.ndarray)
        self.assertEqual(detail.events['outage_entity'][0], study.faults.entity_ids[0])
        # Output arrays own their memory, and the prepared snapshot retains its grid.
        columns = detail.events
        del g, line, detail
        gc.collect()
        self.assertEqual(study.run().summary['passed'], 1)
        np.testing.assert_array_equal(columns['case_id'], [0])

    def test_failure_selection_dc_retry_and_empty(self):
        g = grid()
        study = g.prepare_n1([first_line(g)])
        failed = study.run(max_iter=0)
        self.assertEqual(failed.summary['numerical_failed'], 1)
        self.assertEqual(len(failed.numerical_failures), 1)
        retry = study.run(failed.numerical_failures, initialization='dc')
        self.assertEqual(retry.summary['passed'], 1)
        self.assertEqual(retry.summary['numerical_failed'], 0)
        self.assertEqual(len(retry.events['case_id']), 0)
        self.assertEqual(study.run([]).summary['processed'], 0)
        self.assertEqual(g.prepare_n1([]).run().summary['processed'], 0)

    def test_all_faults_islands_and_workers_agree(self):
        g = grid()
        study = g.prepare_n1()
        one = study.run(workers=1)
        four = study.run(workers=4)
        for key in ['processed', 'passed', 'numerical_failed', 'islanded', 'total_iterations']:
            self.assertEqual(one.summary[key], four.summary[key])
        self.assertEqual(one.summary['processed'], len(study.faults))
        self.assertGreater(one.summary['islanded'], 0)
        self.assertEqual(len(one.island_faults), one.summary['islanded'])

    def test_rules_columns_and_stale_source_snapshot(self):
        net = networks.case14()
        pp.runpp(net, tolerance_mva=1e-10)
        net.bus['min_vm_pu'] = net.res_bus.vm_pu - 1e-4
        net.bus['max_vm_pu'] = net.res_bus.vm_pu + 1e-4
        # Each configured line is just within its baseline rating.
        net.line['max_i_ka'] = np.maximum(net.res_line.i_ka.to_numpy() * 1.001, 1e-6)
        g = grid(net)
        study = g.prepare_n1(voltage_limits=True, line_loading_limit=100.)
        status = study.run(workers=2)
        self.assertGreater(status.summary['violated'], 0)
        detail = study.details(status.rejected_faults, workers=2)
        hits, flows, events = detail.hits, detail.branch_results, detail.events
        self.assertGreater(len(hits['rule_id']), 0)
        self.assertTrue(np.all(hits['event_row'] < len(events['case_id'])))
        mask = hits['result_index'] >= 0
        self.assertTrue(np.any(mask))
        self.assertTrue(np.all(hits['result_index'][mask] < len(flows['event_row'])))
        np.testing.assert_array_equal(flows['event_row'][hits['result_index'][mask]], hits['event_row'][mask])
        self.assertIn('Bus', hits['target_kind'])
        # Snapshot rules were temporary source components; a new default study is silent.
        self.assertEqual(g.prepare_n1().run().summary['violated'], 0)
        first_line(g).in_service = False
        self.assertEqual(study.run().summary['processed'], status.summary['processed'])

    def test_input_validation(self):
        g, other = grid(), grid()
        with self.assertRaisesRegex(ValueError, 'another grid'):
            g.prepare_n1([first_line(other)])
        with self.assertRaises(TypeError):
            g.prepare_n1([g.bus(0)])
        with self.assertRaises(ValueError):
            g.prepare_n1(line_loading_limit=float('nan'))
        with self.assertRaises(ValueError):
            g.prepare_n1(initialization='invalid')
        study = g.prepare_n1([first_line(g)])
        with self.assertRaisesRegex(RuntimeError, 'worker'):
            study.run(workers=0, backend='cpu')
        with self.assertRaisesRegex(ValueError, 'another grid'):
            study.run(other.prepare_n1().faults)
        with self.assertRaisesRegex(ValueError, 'qlim=False'):
            rp.PowerGrid.from_pandapower(networks.case14(), qlim=True).prepare_n1()
        with self.assertRaisesRegex(ValueError, 'backend'):
            study.run(backend='invalid')

    def test_auto_cpu_fallback_and_explicit_cuda_error(self):
        previous = os.environ.get('RUSTPOWER_N1_DISABLE_CUDA')
        os.environ['RUSTPOWER_N1_DISABLE_CUDA'] = '1'
        try:
            available, diagnostic = rp.n1_cuda_status()
            self.assertFalse(available)
            self.assertIn('disabled', diagnostic)
            g = grid()
            study = g.prepare_n1([first_line(g)])
            report = study.run(backend='auto')
            self.assertEqual(report.summary['backend'], 'cpu')
            self.assertEqual(report.summary['processed'], 1)
            with self.assertRaises(RuntimeError):
                study.run(backend='cuda')
            with self.assertRaises(RuntimeError):
                study.run(backend='hybrid')
        finally:
            if previous is None:
                os.environ.pop('RUSTPOWER_N1_DISABLE_CUDA', None)
            else:
                os.environ['RUSTPOWER_N1_DISABLE_CUDA'] = previous

    def test_cuda_and_hybrid_match_cpu_when_available(self):
        available, diagnostic = rp.n1_cuda_status()
        if not available:
            self.skipTest(diagnostic)
        g = grid()
        study = g.prepare_n1()
        cpu = study.run(workers=2, backend='cpu')
        cuda = study.run(backend='cuda')
        hybrid = study.run(workers=2, backend='hybrid')
        for accelerated in [cuda, hybrid]:
            for key in ['processed', 'passed', 'violated', 'numerical_failed', 'islanded', 'total_iterations']:
                self.assertEqual(accelerated.summary[key], cpu.summary[key])
        self.assertEqual(cuda.summary['backend'], 'cuda')
        self.assertEqual(hybrid.summary['backend'], 'hybrid')
        details = study.details(backend='cuda')
        self.assertEqual(len(details.events['case_id']), details.summary['numerically_solved'])

    def test_native_run_releases_gil(self):
        # A Python thread blocked before the native call must run during it.
        study = grid(networks.case300()).prepare_n1()
        ready, go = threading.Event(), threading.Event()
        progress = []
        def observer():
            ready.set()
            go.wait()
            progress.append(True)
        t = threading.Thread(target=observer)
        t.start()
        ready.wait()
        import sys
        previous = sys.getswitchinterval()
        try:
            sys.setswitchinterval(100.)
            go.set()
            study.run(workers=2)
            self.assertTrue(progress)
        finally:
            sys.setswitchinterval(previous)
            t.join()


if __name__ == '__main__':
    unittest.main(verbosity=2)
