import json,sys,statistics,collections,pathlib
root=pathlib.Path(sys.argv[1]); groups=collections.defaultdict(list)
for p in sorted(root.glob('round*.jsonl')):
    for line in p.read_text().splitlines():
        r=json.loads(line)
        assert r['cpu_before']==r['cpu_after'] and r['btf']==0
        assert r['permutation_matches_c'] and r['voltage_error']<1e-9
        groups[r['case'],r['path']].append(r)
summary=[]
for (case,path),rs in sorted(groups.items()):
    item={'case':case,'path':path,'samples':len(rs),'iterations':sorted(set(r['iterations'] for r in rs)),'lnz':sorted(set(r['lnz'] for r in rs)),'unz':sorted(set(r['unz'] for r in rs)),'max_voltage_error':max(r['voltage_error'] for r in rs)}
    for key in ['ordering_us','prepare_symbolic_us','j_symbolic_us','cold_us','reuse_us']:
        vals=[r[key] for r in rs if r[key] is not None]
        item[key]=statistics.median(vals) if vals else None
    summary.append(item)
for case in ['39','118','9241']:
    c=next(r for r in summary if r['case']==case and r['path']=='c_scalar')
    for r in (x for x in summary if x['case']==case):
        r['cold_change_pct']=100*(r['cold_us']/c['cold_us']-1)
        print(case,r['path'],'N=',r['samples'],'order=',r['ordering_us'],'prep=',round(r['prepare_symbolic_us'],3),'cold=',round(r['cold_us'],3),'reuse=',round(r['reuse_us'],3),'fill=',r['lnz'],r['unz'],'change=',round(r['cold_change_pct'],2))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
