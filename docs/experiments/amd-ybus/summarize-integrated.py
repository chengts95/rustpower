import json,sys,pathlib,statistics
root=pathlib.Path(sys.argv[1]);rows=[]
for p in sorted(root.glob('btf*-round*.jsonl')):
 for line in p.open():r=json.loads(line);r['round']=p.name;rows.append(r)
summary=[]
for btf in [1,0]:
 for case in ['IEEE39','IEEE118','PEGASE9241']:
  for mode in ['cold','reuse']:
   g={i:[r for r in rows if r['case']==case and r['mode']==mode and r['btf']==btf and r['path']==i] for i in [0,1]}
   if not all(g.values()):continue
   fields=['permutation_us','ordering_us','j_symbolic_us','net_preparation_us','full_nr_us','full_with_permutation_us','amd_call_us']
   stats={str(i):{k:(statistics.median(r[k] for r in g[i]) if all(r[k] is not None for r in g[i]) else None) for k in fields} for i in [0,1]}
   rec=dict(case=case,mode=mode,btf=btf,count_per_path=len(g[0]),medians=stats,
    net_preparation_includes_j=all(r.get("net_preparation_includes_j", True) for r in g[0]+g[1]),
    full_nr_change_pct=100*(stats['1']['full_nr_us']/stats['0']['full_nr_us']-1),ordering_saving_us=stats['0']['ordering_us']-stats['1']['ordering_us'],
    permutation_cost_us=statistics.median(r['permutation_us'] for r in g[0]+g[1]),
    iterations={str(i):sorted({r['iterations'] for r in g[i]}) for i in [0,1]},max_voltage_error=max(r['max_voltage_error'] for r in g[0]+g[1]),
    pattern_key_bytes=g[1][0]['pattern_key_bytes'],fill={str(i):[g[i][0]['lnz'],g[i][0]['unz']] for i in [0,1]},
    actual_cpus=sorted({r['cpu_before'] for r in g[0]+g[1]}|{r['cpu_after'] for r in g[0]+g[1]}))
   summary.append(rec)
   print(f"BTF={btf} {case} {mode}: base/new NR {stats['0']['full_nr_us']:.3f}/{stats['1']['full_nr_us']:.3f} us ({rec['full_nr_change_pct']:+.2f}%), permutation {rec['permutation_cost_us']:.3f} us, ordering saving {rec['ordering_saving_us']:.3f} us")
   print('  prepare(base/new)',stats['0']['net_preparation_us'],stats['1']['net_preparation_us'],'ordering(base/new)',stats['0']['ordering_us'],stats['1']['ordering_us'],'J symbolic(base/new)',stats['0']['j_symbolic_us'],stats['1']['j_symbolic_us'])
   print('  iterations',rec['iterations'],'V error',rec['max_voltage_error'],'fill',rec['fill'],'key bytes',rec['pattern_key_bytes'])
json.dump(summary,(root/'summary.json').open('w'),indent=2)
assert rows and all(r['converged'] and r['residual']<1e-10 and r['max_voltage_error']<1e-8 for r in rows)
assert all(r['cpu_before']==r['cpu_after'] for r in rows)
assert all(r['optimized'] and r['amd_calls']==0 for r in rows if r['path']==1)
assert all(r['amd_calls']==(1 if r['mode']=='cold' else 0) for r in rows if r['path']==0)
print('Validated records',len(rows))
