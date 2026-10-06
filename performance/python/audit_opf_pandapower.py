"""Export identical pandapower PIPS inputs and time its core and public runopp API.
Run before examples/audit_opf.rs; then pass --compare to evaluate Rust results.
"""
import argparse, importlib, json, platform, time, hashlib, os, subprocess
from pathlib import Path
import numpy as np
import scipy
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
from scipy.sparse import spmatrix

# pandapower 3.5.4 OPF uses the legacy SciPy .H alias removed in SciPy 1.14.
# Restore only that spelling, with identical conjugate-transpose semantics.
SCIPY_H_SHIM = not hasattr(spmatrix, 'H')
if SCIPY_H_SHIM:
    spmatrix.H = property(lambda self: self.conjugate().transpose())

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'target/research/opf_audit'

def sparse(m):
    m = m.tocsc(); m.sort_indices()
    return dict(rows=m.shape[0], cols=m.shape[1], cp=m.indptr.tolist(), ri=m.indices.tolist(), re=m.data.real.tolist(), im=m.data.imag.tolist())

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def export(case, repeats):
    net = getattr(pn, case)()
    net.ext_grid.loc[:, 'va_degree'] = 0.0  # common phase reference, physical flows unchanged
    mod = importlib.import_module('pandapower.pypower.pipsopf_solver')
    original = mod.pips
    captured = {}
    def capture(*args, **kwargs):
        captured['args'] = args
        # Closure supplies the exact internal model passed to PIPS.
        closure = dict(zip(args[0].__code__.co_freevars, [c.cell_contents for c in args[0].__closure__]))
        captured['om'] = closure['om']
        t = time.perf_counter(); result = original(*args, **kwargs)
        captured['core_ms'] = 1000*(time.perf_counter()-t)
        captured['result'] = result
        return result
    mod.pips = capture
    opts = dict(init='flat', calculate_voltage_angles=True, OPF_FLOW_LIM=0,
                PDIPM_FEASTOL=1e-6, PDIPM_GRADTOL=1e-6,
                PDIPM_COMPTOL=1e-6, PDIPM_COSTTOL=1e-6, PDIPM_MAX_IT=150)
    try:
        try:
            pp.runopp(net, **opts)  # untimed warm-up, including imports/JIT
        except pp.OPFNotConverged:
            r=captured.get('result', {})
            dump(OUT/f'{case}_failure.json', dict(case=case, stage='pandapower warm-up', options=opts,
                converged=False, iterations=int(r.get('output',{}).get('iterations',-1)),
                message=r.get('output',{}).get('message','OPFNotConverged'),
                objective=float(r['f']) if np.isfinite(r.get('f',np.nan)) else None))
            print(case, 'pandapower did not converge; failure recorded, no speed ratio', flush=True)
            return
        api_times, core_times = [], []
        for _ in range(repeats):
            t = time.perf_counter(); pp.runopp(net, **opts)
            api_times.append(1000*(time.perf_counter()-t)); core_times.append(captured['core_ms'])
    finally:
        mod.pips = original
    args = captured['args']; om = captured['om']; ppc = om.get_ppc()
    b, g, br, cost = [ppc[k].real for k in ('bus','gen','branch','gencost')]
    nb, ng, nl = len(b),len(g),len(br)
    assert np.array_equal(b[:,0], np.arange(nb))
    assert cost.shape[0] == ng and np.all(cost[:,0] == 2) and np.all(cost[:,3] <= 3)
    assert args[2] is None or args[2].shape[0] == 0, 'Additional linear constraints require a wider Rust formulation'
    assert np.all((br[:,5] > 0) & (br[:,5] < 1e10)), 'This audit requires all branch limits finite'
    ybus,yf,yt = makeYbus(ppc['baseMVA'], b, br)
    costs = [np.pad(row[4:4+int(row[3])], (3-int(row[3]),0)).tolist() for row in cost]
    ref = np.flatnonzero(b[:,1] == 3); assert len(ref) == 1 and abs(args[5][ref[0]]) < 1e-14, (ref, args[5][ref], args[6][ref])
    # Finite proxies only for unbounded angles; bounds below exclude these proxies.
    data = dict(case=case, nb=nb,ng=ng,nl=nl,base_mva=ppc['baseMVA'],ref_bus=int(ref[0]),
        ybus=sparse(ybus),yf=sparse(yf),yt=sparse(yt),f_buses=br[:,0].astype(int).tolist(),t_buses=br[:,1].astype(int).tolist(),
        s_load_re=(b[:,2]/ppc['baseMVA']).tolist(), s_load_im=(b[:,3]/ppc['baseMVA']).tolist(),
        gen_bus=g[:,0].astype(int).tolist(), rate_a=(br[:,5]/ppc['baseMVA']).tolist(), cost_coeffs=costs,
        xmin=np.where(np.isfinite(args[5]),args[5],-1e20).tolist(),xmax=np.where(np.isfinite(args[6]),args[6],1e20).tolist(),x0=args[1].tolist())
    network = dict(sn_mva=float(net.sn_mva), f_hz=float(net.f_hz))
    for key in ('bus','gen','ext_grid','load','sgen','shunt','line','trafo','switch'):
        frame=net[key].copy()
        if key == 'trafo' and 'tap_phase_shifter' not in frame:
            frame['tap_phase_shifter'] = False
        for column in ('bus','from_bus','to_bus','hv_bus','lv_bus','parallel','step','max_step','element'):
            if column in frame:
                frame[column]=frame[column].astype('int64')
        frame['index']=frame.index
        network[key]=json.loads(frame.to_json(orient='records',double_precision=15))
    data['network']=network
    probe_x=args[1].copy()
    probe_x[:nb] += 0.01*np.sin(np.arange(nb)); probe_x[ref[0]]=0.0
    probe_x[nb:2*nb] += 0.005*np.cos(np.arange(nb))
    probe_lam=0.1+0.03*np.sin(np.arange(2*nb))
    probe_mu=0.05+0.01*np.cos(np.arange(2*nl))
    probe_h=args[8](probe_x, dict(eqnonlin=probe_lam, ineqnonlin=probe_mu), 1e-4)
    data['hessian_probe']=dict(x=probe_x.tolist(),lam=probe_lam.tolist(),mu=probe_mu.tolist(),hessian=sparse(probe_h))
    dump(OUT/f'{case}_input.json',data)
    result=captured['result']
    reference=dict(case=case,converged=bool(result['eflag']>0),f=float(result['f']),x=result['x'].tolist(),iterations=int(result['output']['iterations']),
        core_ms=core_times,runopp_ms=api_times,options=opts,pips_options=args[9],final_measures={k:float(v) for k,v in result['output']['hist'][-1].items()},
        input_sha256=hashlib.sha256((OUT/f'{case}_input.json').read_bytes()).hexdigest())
    dump(OUT/f'{case}_pandapower.json',reference)
    print(case,reference['f'],reference['iterations'],'core median ms',np.median(core_times),flush=True)

def compare(case):
    if not (OUT/f'{case}_input.json').exists():
        print(case, 'no matched-input trial; see failure record'); return
    d=json.loads((OUT/f'{case}_input.json').read_text()); ref=json.loads((OUT/f'{case}_pandapower.json').read_text())
    assert hashlib.sha256((OUT/f'{case}_input.json').read_bytes()).hexdigest() == ref['input_sha256'], 'Input changed since reference solve'
    from scipy.sparse import csc_matrix
    def matrix(k):
        a=d[k]; return csc_matrix((np.array(a['re'])+1j*np.array(a['im']),a['ri'],a['cp']),shape=(a['rows'],a['cols']))
    y,yf,yt=[matrix(k) for k in ('ybus','yf','yt')]
    nb,ng=d['nb'],d['ng']; base=d['base_mva']; xr=np.array(ref['x'])
    def assess(x):
        x=np.array(x); assert x.shape == (2*nb+2*ng,) and np.all(np.isfinite(x)), 'Nonfinite or malformed solution'
        v=x[nb:2*nb]*np.exp(1j*x[:nb]); s=v*np.conj(y@v)+np.array(d['s_load_re'])+1j*np.array(d['s_load_im'])
        np.add.at(s,d['gen_bus'],-x[2*nb:2*nb+ng]-1j*x[2*nb+ng:])
        sf=v[d['f_buses']]*np.conj(yf@v); st=v[d['t_buses']]*np.conj(yt@v)
        return dict(balance_inf_pu=float(max(abs(s.real).max(),abs(s.imag).max())),flow_violation_pu=float(max(0,np.max(np.abs(sf)-d['rate_a']),np.max(np.abs(st)-d['rate_a']))),bound_violation=float(max(0,np.max(np.array(d['xmin'])-x),np.max(x-np.array(d['xmax'])))),objective_recomputed=float(sum(c[0]*(p*base)**2+c[1]*p*base+c[2] for c,p in zip(d['cost_coeffs'],x[2*nb:2*nb+ng]))))
    rows=[]
    for line in (OUT/f'{case}_rust.jsonl').read_text().splitlines():
        if not line.startswith('{'): continue
        r=json.loads(line); x=np.array(r.pop('x')); r.update(assess(x)); r.update(objective_abs_diff=abs(r['f']-ref['f']),vm_max_diff=float(max(abs(x[nb:2*nb]-xr[nb:2*nb]))),va_max_diff_rad=float(max(abs(x[:nb]-xr[:nb]))),pg_max_diff_mw=float(base*max(abs(x[2*nb:2*nb+ng]-xr[2*nb:2*nb+ng]))),qg_max_diff_mvar=float(base*max(abs(x[2*nb+ng:]-xr[2*nb+ng:]))))
        rows.append(r)
    assert len(rows) == len(ref['core_ms'])*7, f'Missing Rust results for {case}: {len(rows)}'
    assert set(r['version'] for r in rows) == {'V1','V4','V5.0','V5.2','V5.3','V5.5','V5.6'}
    assert all(np.isfinite(r['f']) and abs(r['f']-r['objective_recomputed']) < 1e-8*max(1,abs(r['f'])) for r in rows), 'Reported objective disagrees with independent computation'
    assert all(r['converged'] for r in rows), 'A Rust variant failed; inspect raw results'
    assert all(r['objective_abs_diff']/max(1,abs(ref['f'])) < 1e-6 and r['balance_inf_pu'] < 1e-5 and r['flow_violation_pu'] < 1e-7 and r['bound_violation'] < 1e-7 for r in rows), 'Independent solution check failed'
    dump(OUT/f'{case}_comparison.json',dict(pandapower=assess(xr),rust=rows))
    for version in sorted(set(r['version'] for r in rows)):
        a=[r for r in rows if r['version']==version]; print(case,version,'converged',all(r['converged'] for r in a),'ms',np.median([r['total_ms'] for r in a]),'df',a[0]['objective_abs_diff'],'balance',a[0]['balance_inf_pu'])

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--compare',action='store_true'); p.add_argument('--cases',nargs='+',default=['case39','case118','case300']); p.add_argument('--repeats',type=int,default=5); a=p.parse_args(); OUT.mkdir(exist_ok=True,parents=True)
    if not a.compare:
        dump(OUT/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),pandapower=pp.__version__,numpy=np.__version__,scipy=scipy.__version__,repeats=a.repeats, scipy_H_compatibility_alias=SCIPY_H_SHIM, commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),rustc=subprocess.check_output(['rustc','--version'],text=True).strip(), cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_model=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),omp_threads=os.environ.get('OMP_NUM_THREADS')))
    for case in a.cases:
        compare(case) if a.compare else export(case,a.repeats)
