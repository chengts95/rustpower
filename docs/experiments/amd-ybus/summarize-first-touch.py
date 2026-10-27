import sys,pathlib,json,collections,statistics
p=pathlib.Path(sys.argv[1]);g=collections.defaultdict(list)
for f in p.glob('round*.jsonl'):
 for line in f.read_text().splitlines():
  r=json.loads(line);assert r['permutation_exact'] and r['cpu']==0;g[r['case'],r['variant']].append(r)
s=[]
for (case,variant),rs in sorted(g.items()):
 r={'case':case,'variant':variant,'samples':len(rs),'us':statistics.median(x['us'] for x in rs)}
 for key in ['input_us','kernel_us']:
  vals=[x[key] for x in rs if x[key] is not None];r[key]=statistics.median(vals) if vals else None
 for key in ['pristine_updates','payloads','pair_reuses','own_twin_skips','seed_scanned','payload_written','suffix_scans_avoided','lnz_estimate']:
  vals=set(x[key] for x in rs);assert len(vals)==1;r[key]=vals.pop()
 s.append(r)
for case in ['39','118','9241']:
 b=next(x for x in s if x['case']==case and x['variant']=='original_scalar')
 for r in (x for x in s if x['case']==case):
  r['change_pct']=100*(r['us']/b['us']-1);print(case,r['variant'],'us',round(r['us'],3),'delta%',round(r['change_pct'],2),'input',r['input_us'],'kernel',r['kernel_us'],'reuses',r['pair_reuses'],'scan avoided',r['suffix_scans_avoided'])
(p/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
