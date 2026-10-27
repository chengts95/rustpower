import json,pathlib,sys,collections,statistics
p=pathlib.Path(sys.argv[1]);g=collections.defaultdict(list)
for f in p.glob('round*.jsonl'):
 for l in f.read_text().splitlines():
  r=json.loads(l);assert r['permutation_exact'];g[r['case'],r['mode'],r['variant']].append(r)
out=[]
for (case,mode,variant),rs in sorted(g.items()):
 vals=[x['us'] for x in rs];qs=statistics.quantiles(vals,n=10)
 r={'case':case,'mode':mode,'variant':variant,'samples':len(vals),'median_us':statistics.median(vals),'p10_us':qs[0],'p90_us':qs[-1],'min_us':min(vals),'max_us':max(vals)}
 for key in ['scalar_variables','full_initial_entries','initial_neighbor_reads','initial_neighbor_ids_saved','dynamic_writes','distinct_dynamic_slots','initial_variables_read','allocated_value_slots','arena_compactions']:
  v=set(x[key] for x in rs);assert len(v)==1,(key,v);r[key]=v.pop()
 out.append(r)
for case in ['39','118','9241']:
 for mode in ['fresh','reuse']:
  a=next(x for x in out if (x['case'],x['mode'],x['variant'])==(case,mode,'full_initial_table'));b=next(x for x in out if(x['case'],x['mode'],x['variant'])==(case,mode,'on_demand_neighbors'))
  print(case,mode,round(a['median_us'],3),'->',round(b['median_us'],3),'change%',round(100*(b['median_us']/a['median_us']-1),2),'saved_initial',b['initial_neighbor_ids_saved'],'reads',b['initial_neighbor_reads'],'dynamic_writes',b['dynamic_writes'],'distinct',b['distinct_dynamic_slots'],'alloc slots',b['allocated_value_slots'])
(p/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
