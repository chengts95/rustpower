#!/usr/bin/env python3
import json,pathlib,statistics,sys
p=pathlib.Path(sys.argv[1]);out={'wall':{},'phases':{},'counters':{}}
def summary(v):
 s=sorted(v);return {'median':statistics.median(s),'p10':s[int(.1*(len(s)-1))],'p90':s[int(.9*(len(s)-1))],'n':len(s)}
for f in sorted(p.glob('time-*.json')):
 d=json.loads(f.read_text());k=f'{d["case"]}/{d["mode"]}/{d["variant"]}'
 out['wall'].setdefault(k,[]).append({'batch':f.name,**summary([x*1e6 for x in d['seconds']])})
for f in sorted(p.glob('phases-*.json')):
 d=json.loads(f.read_text());out['phases'][f'{d["case"]}/{d["mode"]}']={name:summary([r[i]*1e6 for r in d['phases_seconds']]) for i,name in enumerate(d['phase_names'])}
for f in sorted(p.glob('stat-*.csv')):
 d=json.loads(f.with_name(f.stem+'-driver.json').read_text());v={}
 for line in f.read_text().splitlines():
  a=line.split(';')
  if len(a)<5 or not a[0] or a[0].startswith('#'):continue
  try:v[a[2]]={'per_order':float(a[0])/d['reps'],'running_percent':a[4]}
  except ValueError:v[a[2]]={'unavailable':a[0]}
 out['counters'][f.stem]=v
(p/'summary.json').write_text(json.dumps(out,indent=2))
for k,v in out['wall'].items():print(k,[(round(x['median'],3),round(x['p10'],3),round(x['p90'],3)) for x in v])
for k,v in out['phases'].items():print('phase',k,{n:round(x['median'],3) for n,x in v.items()})
for k,v in out['counters'].items():
 if k.endswith('basic'):print(k,{n:round(x['per_order'],1) for n,x in v.items() if 'per_order' in x})
