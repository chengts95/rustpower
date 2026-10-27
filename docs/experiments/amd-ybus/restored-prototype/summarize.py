import json,sys,statistics,random
paths=sys.argv[1:] or ['super-amd-round1.jsonl','super-amd-round2.jsonl']
r=[]
for p in paths:
 for line in open(p):x=json.loads(line);x['round']=p;r.append(x)
def dist(v):
 v=sorted(v);return dict(n=len(v),median=statistics.median(v),mean=statistics.mean(v),p25=v[int((len(v)-1)*.25)],p75=v[int((len(v)-1)*.75)],min=v[0],max=v[-1])
out=[]
for mode in ['cold','reuse']:
 f={i:[x for x in r if x['kind']=='full' and x['mode']==mode and x['path']==i] for i in range(3)}
 t={i:[x for x in r if x['kind']=='stages' and x['mode']==mode and x['path']==i] for i in range(3)}
 if not all(f.values()) or not all(t.values()):continue
 full={str(i):dist([x['us'] for x in f[i]]) for i in range(3)}
 stage={str(i):{k:dist([x[k] for x in t[i]]) for k in ['analyze_us','factor_us','refactor_us','triangular_us','prepare_us']} for i in range(3)}
 amd={str(i):{k:dist([x['stats'][k] for x in f[i]]) for k in ['amd_us','aat_us','elim_us','direct_prep_us']} for i in range(3)}
 amd['0']['amd1_graph_us']=dist([x['stats']['amd1_graph_us'] for x in f[0]])
 pairs={}
 for i in range(3):
  for x in f[i]:pairs.setdefault((x['round'],x['pair']),{})[i]=x['us']
 delta={str(i):[100*(p[i]/p[0]-1) for p in pairs.values()] for i in (1,2)}
 rec=dict(case='PEGASE9241',mode=mode,labels={'0':'previous_exact_early_AMD','1':'direct_Ybus_AMD2','2':'seeded_supervariable_AMD2'},full_us=full,stage_us=stage,amd_components_us=amd,
   relative_median_pct={str(i):100*(full[str(i)]['median']/full['0']['median']-1) for i in (1,2)},paired_pct={i:dist(v) for i,v in delta.items()},
   stats={str(i):f[i][0]['stats'] for i in range(3)},iterations={str(i):sorted({x['iterations'] for x in f[i]}) for i in range(3)},max_voltage_error=max(x['max_voltage_error'] for x in sum(f.values(),[])),max_residual=max(x['residual'] for x in sum(f.values(),[])),
   per_round={p:{str(i):dist([x['us'] for x in f[i] if x['round']==p]) for i in range(3)} for p in paths})
 out.append(rec)
 print(mode,'NR(us)',{i:round(full[str(i)]['median'],3) for i in range(3)},'changes',rec['relative_median_pct'])
 print('AMD components(us)',{i:{k:round(v['median'],3) for k,v in amd[str(i)].items()} for i in range(3)})
 print('KLU stages(us)',{i:{k:round(v['median'],3) for k,v in stage[str(i)].items()} for i in range(3)})
 print('L/U',[(i,rec['stats'][str(i)]['lnz'],rec['stats'][str(i)]['unz']) for i in range(3)],'iters',rec['iterations'],'maxVerror',rec['max_voltage_error'])
json.dump(out,open('super-amd-summary.json','w'),indent=2)
