import json,sys,statistics,random
paths=sys.argv[1:] or ['early-amd-round1.jsonl','early-amd-round2.jsonl']
rows=[]
for p in paths:
 for line in open(p):r=json.loads(line);r['round']=p;rows.append(r)
def dist(v):
 v=sorted(v);return dict(n=len(v),median=statistics.median(v),mean=statistics.mean(v),p25=v[int((len(v)-1)*.25)],p75=v[int((len(v)-1)*.75)],min=v[0],max=v[-1])
s=[]
for c in ['IEEE39','IEEE118','PEGASE9241']:
 for mode in ['cold','reuse']:
  f={i:[r for r in rows if r['case']==c and r['mode']==mode and r['kind']=='full' and r['path']==i] for i in (0,1)}
  t={i:[r for r in rows if r['case']==c and r['mode']==mode and r['kind']=='stages' and r['path']==i] for i in (0,1)}
  full={str(i):dist([r['us'] for r in f[i]]) for i in (0,1)}
  stage={str(i):{k:dist([r[k] for r in t[i]]) for k in ['analyze_us','factor_us','refactor_us','triangular_us','prepare_us']} for i in (0,1)}
  pairs={}
  for i in (0,1):
   for r in f[i]:pairs.setdefault((r['round'],r['pair']),{})[i]=r['us']
  deltas=[100*(p[1]/p[0]-1) for p in pairs.values()]
  rng=random.Random(1729);boots=sorted(statistics.median(rng.choices(deltas,k=len(deltas))) for _ in range(2000))
  rec=dict(case=c,mode=mode,full_us=full,stages_us=stage,early_vs_base_pct=100*(full['1']['median']/full['0']['median']-1),paired_pct=dist(deltas),paired_median_bootstrap95=[boots[49],boots[1949]],
   stats={str(i):f[i][0]['stats'] for i in (0,1)},amd_us={str(i):dist([r['stats']['amd_us'] for r in f[i]]) for i in (0,1)},
   iterations={str(i):sorted({r['iterations'] for r in f[i]}) for i in (0,1)},max_residual=max(r['residual'] for r in f[0]+f[1]),max_voltage_error=max(r['max_voltage_error'] for r in f[0]+f[1]),
   per_round={p:{str(i):dist([r['us'] for r in f[i] if r['round']==p]) for i in (0,1)} for p in paths})
  s.append(rec)
  print(f"{c:10} {mode:5} n={full['0']['n']}: baseline={full['0']['median']:.3f}us; early_exact_AMD={full['1']['median']:.3f}us; change={rec['early_vs_base_pct']:+.3f}%; paired={rec['paired_pct']['median']:+.3f}% CI95={rec['paired_median_bootstrap95']}")
  print('  stages(us)',{i:{k:round(stage[str(i)][k]['median'],3) for k in stage[str(i)]} for i in (0,1)})
  print('  KLU peak bytes',{i:rec['stats'][str(i)]['mempeak'] for i in (0,1)},'AMD(us)',{i:rec['amd_us'][str(i)]['median'] for i in (0,1)})
  print('  iters',rec['iterations'],'voltage error',rec['max_voltage_error'])
json.dump(s,open('early-amd-summary.json','w'),indent=2)
