"""Export the exact pandapower NR inputs for flat/DC starts on case6515rte.

Run with the environment containing pandapower; then use example audit_lm.
No results initialization, reactive-limit switching, distributed slack or ZIP loads.
"""
import argparse
import importlib
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import pandapower as pp
import pandapower.networks as pn

parser = argparse.ArgumentParser()
parser.add_argument('--phase-scale', type=float, default=1.0)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'target/research/lm_audit')
cli = parser.parse_args()
out = cli.output
if cli.phase_scale != 1.0:
    out = out / f'phase_scale_{cli.phase_scale:g}'
out.mkdir(exist_ok=True, parents=True)
mod = importlib.import_module('pandapower.pf.run_newton_raphson_pf')
original = mod.newtonpf
records = []
models = []
for init in ['flat', 'dc']:
    net = pn.case6515rte()
    net.trafo['shift_degree'] *= cli.phase_scale
    captured = {}
    def capture(Y, S, V, ref, pv, pq, ppci, options, *args):
        order = np.r_[pq, pv, ref].astype(int)
        assert len(np.unique(order)) == Y.shape[0]
        y = Y.tocsc()[order, :][:, order].tocsc()
        y.sort_indices()
        captured.update(dict(case='6515rte', phase_scale=cli.phase_scale, init=init, nb=len(order), npq=len(pq), npv=len(pv),
            base_mva=float(ppci['baseMVA']), tolerance_pu=float(options['tolerance_mva']), max_iter=300,
            order=order.tolist(), cp=y.indptr.tolist(), ri=y.indices.tolist(),
            y_re=y.data.real.tolist(), y_im=y.data.imag.tolist(),
            s_re=S[order].real.tolist(), s_im=S[order].imag.tolist(),
            v_re=V[order].real.tolist(), v_im=V[order].imag.tolist()))
        t = perf_counter()
        result = original(Y, S, V, ref, pv, pq, ppci, options, *args)
        elapsed = perf_counter()-t
        mis = result[0] * np.conj(Y @ result[0]) - S
        residual = np.max(np.abs(np.r_[mis[np.r_[pq,pv]].real, mis[pq].imag]))
        captured['pp_result'] = dict(converged=bool(result[1]), iterations=int(result[2]),
            residual_inf=float(residual) if np.isfinite(residual) else None, core_ms=elapsed*1000,
            v_re=[float(x) if np.isfinite(x) else None for x in result[0][order].real],
            v_im=[float(x) if np.isfinite(x) else None for x in result[0][order].imag])
        return result
    mod.newtonpf = capture
    try:
        pp.runpp(net, init=init, calculate_voltage_angles=True, algorithm='nr',
                 max_iteration=300, tolerance_mva=1e-8,
                 enforce_q_lims=False, distributed_slack=False,
                 voltage_depend_loads=False, numba=False, lightsim2grid=False)
        error = None
    except pp.LoadflowNotConverged as exc:
        error = str(exc)
    finally:
        mod.newtonpf = original
    if 'pp_result' not in captured:
        raise RuntimeError('NR interception did not complete')
    path = out / f'6515rte_{init}.json'
    path.write_text(json.dumps(captured, allow_nan=False))
    models.append(captured)
    summary = {k:v for k,v in captured['pp_result'].items() if not k.startswith('v_')}
    records.append(dict(init=init, error=error, **summary))
    print(records[-1], flush=True)
for key in ['order','cp','ri','y_re','y_im']:
    assert models[0][key] == models[1][key], f'Flat/DC numerical models differ: {key}'
# DC initialization updates slack generation; its specified injection is not
# a PF equation. Require exact equality of every retained equation instead.
nact = models[0]['npq'] + models[0]['npv']
assert models[0]['s_re'][:nact] == models[1]['s_re'][:nact]
assert models[0]['s_im'][:models[0]['npq']] == models[1]['s_im'][:models[0]['npq']]
(out/'pandapower.json').write_text(json.dumps(dict(pandapower=pp.__version__, numpy=np.__version__,
    python=platform.python_version(), records=records), indent=2))
