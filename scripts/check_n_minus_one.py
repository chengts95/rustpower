"""N-1 correctness survey on small pandapower cases; writes per-contingency results.

Uses the explicitly imported rustpower build. Whole-device outages, no switches.
118 is intentionally excluded. This is a correctness check, not a benchmark.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.topology import unsupplied_buses
import rustpower as rp


def solve(grid):
    report = grid.solve(max_iter=50, tol=1e-10)
    if not report.converged:
        raise AssertionError('RustPower did not converge')
    return grid.res_bus.set_index('pp_bus_id').sort_index()


def check_bus(result, expected):
    result = result.reindex(expected.index)
    np.testing.assert_allclose(result.vm_pu, expected.vm_pu, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.va_degree, expected.va_degree, atol=1e-6, rtol=0)



def check_device_results(grid, net):
    lines = grid.res_line
    for label, branch in net.line.iterrows():
        selected = lines[(lines.from_bus == branch.from_bus) & (lines.to_bus == branch.to_bus)]
        # Endpoint identity only supports unique pairs in the current public API.
        if len(selected) != 1:
            continue
        actual = selected.iloc[0]
        cols = ['p_from_mw','q_from_mvar','p_to_mw','q_to_mvar','i_from_ka','i_to_ka','loading_percent']
        expected = net.res_line.loc[label, cols] if branch.in_service else np.zeros(len(cols))
        np.testing.assert_allclose(actual[cols].astype(float), expected, atol=1e-6, rtol=1e-8)
    trafos = grid.res_trafo.set_index('trafo_id')
    cols = ['p_hv_mw','q_hv_mvar','p_lv_mw','q_lv_mvar','i_hv_ka','i_lv_ka','loading_percent']
    for label, branch in net.trafo.iterrows():
        expected = net.res_trafo.loc[label, cols] if branch.in_service else np.zeros(len(cols))
        np.testing.assert_allclose(trafos.loc[label, cols].astype(float), expected, atol=1e-6, rtol=1e-8)


def attempt(fn):
    try:
        fn()
        return 'ok'
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return f'{type(exc).__name__}: {exc}'


def survey(factory, cache):
    net = factory()
    pp.runpp(net, init='flat', tolerance_mva=1e-10, max_iteration=50, lightsim2grid=False)
    baseline = net.res_bus.copy()
    reused = rp.PowerGrid.from_pandapower(net)
    reused.enable_cache(cache)
    check_bus(solve(reused), baseline)
    live = rp.PowerGrid.from_pandapower(net)
    live.enable_cache(cache)
    check_bus(solve(live), baseline)
    counts = Counter(tuple(sorted((int(row.from_bus), int(row.to_bus)))) for _, row in net.line.iterrows())
    for kind in ('line', 'trafo'):
        table = net[kind]
        for label in table.index[table.in_service]:
            table.at[label, 'in_service'] = False
            row = dict(case=factory.__name__, cache=cache, kind=kind, label=int(label))
            missing = sorted(int(i) for i in unsupplied_buses(net))
            row['unsupplied_buses'] = missing
            if missing:
                # Survey island behavior separately; voltage comparisons require a supplied graph.
                def island():
                    g = rp.PowerGrid.from_pandapower(net)
                    report = g.solve(max_iter=50, tol=1e-10)
                    row['island_converged'] = bool(report.converged)
                    row['island_voltage_finite'] = bool(np.isfinite(g.res_bus.vm_pu).all())
                row['island_behavior'] = attempt(island)
            else:
                try:
                    pp.runpp(net, init='flat', tolerance_mva=1e-10, max_iteration=50, lightsim2grid=False)
                    expected = net.res_bus.copy()
                    row['pandapower'] = 'ok'
                except Exception as exc:
                    row['pandapower'] = f'{type(exc).__name__}: {exc}'
                    expected = None
                if expected is not None:
                    def fresh():
                        g = rp.PowerGrid.from_pandapower(net)
                        g.enable_cache(cache)
                        check_bus(solve(g), expected)
                        check_device_results(g, net)
                    row['fresh'] = attempt(fresh)
                    def reimport():
                        reused.from_pp_net(net)
                        check_bus(solve(reused), expected)
                        check_device_results(reused, net)
                    row['reimport'] = attempt(reimport)
                    if kind == 'line':
                        branch = table.loc[label]
                        pair = tuple(sorted((int(branch.from_bus), int(branch.to_bus))))
                        if counts[pair] == 1:
                            handle = live.line(int(branch.from_bus), int(branch.to_bus))
                            def outage():
                                handle.in_service = False
                                check_bus(solve(live), expected)
                                check_device_results(live, net)
                                results = live.res_line
                                hit = results[((results.from_bus == branch.from_bus) & (results.to_bus == branch.to_bus)) |
                                              ((results.to_bus == branch.from_bus) & (results.from_bus == branch.to_bus))]
                                if len(hit) != 1:
                                    raise AssertionError('outaged line result identity ambiguous')
                                np.testing.assert_allclose(hit[['p_from_mw','q_from_mvar','p_to_mw','q_to_mvar']], 0, atol=1e-10)
                            row['live_outage'] = attempt(outage)
                            def restore():
                                handle.in_service = True
                                check_bus(solve(live), baseline)
                            row['live_restore'] = attempt(restore)
                        else:
                            row['live_lookup'] = 'parallel branch: endpoint API is ambiguous'
            table.at[label, 'in_service'] = True
            def restore_import():
                reused.from_pp_net(net)
                check_bus(solve(reused), baseline)
            row['reimport_restore'] = attempt(restore_import)
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for factory in (pn.case14, pn.case39, pn.case57):
        for cache in (False, True):
            rows.extend(survey(factory, cache))
            args.output.write_text(json.dumps(dict(features=rp.features(), module=rp.__file__, records=rows), indent=2))
            failures = [(r['kind'],r['label'],k) for r in rows if r['case']==factory.__name__ and r['cache']==cache
                        for k in ('fresh','reimport','live_outage','live_restore','reimport_restore') if k in r and r[k]!='ok']
            print(factory.__name__, 'cache', cache, 'failures', failures, flush=True)
    bad = [r for r in rows if any(r.get(k,'ok')!='ok' for k in ('fresh','reimport','live_outage','live_restore','reimport_restore'))]
    print('records',len(rows),'connected',sum(not r['unsupplied_buses'] for r in rows),'failed records',len(bad),flush=True)
    panics = [r for r in rows if r.get('island_behavior','').startswith('PanicException:')]
    print('island panic records',len(panics),flush=True)
    return bool(bad or panics)

if __name__ == '__main__':
    raise SystemExit(main())
