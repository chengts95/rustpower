#!/usr/bin/env python3
import pathlib,subprocess,collections,re,json,sys,html
p=pathlib.Path(sys.argv[1]);binary=p/'profile_amd'
nm=subprocess.check_output(['nm','-n','-C',str(binary)],text=True)
base={parts[2]:int(parts[0],16) for l in nm.splitlines() if len(parts:=l.split(None,2))==3 and re.fullmatch('[0-9a-fA-F]+',parts[0])}
results={}
for data in sorted(p.glob('flat-[12]-*.data')):
 if not data.with_name(data.stem+'-driver.json').exists():continue
 text=subprocess.check_output(['perf','script','-i',str(data),'--no-inline','-F','period,ip,sym,symoff'],text=True,stderr=subprocess.DEVNULL)
 data.with_name(data.stem+'-leaf.txt').write_text(text)
 raw=[];addresses=set()
 for line in text.splitlines():
  fields=line.split(None,2)
  if len(fields)!=3:continue
  weight=int(fields[0]);m=re.match(r'(.+)\+0x([0-9a-f]+)$',fields[2]);addr=None
  sym=m[1] if m else fields[2]
  if m and sym in base:addr=base[sym]+int(m[2],16);addresses.add(addr)
  raw.append((weight,sym,addr))
 mapping={}
 if addresses:
  args=['addr2line','-a','-i','-f','-C','-e',str(binary)]+[hex(a) for a in sorted(addresses)]
  resolved=subprocess.check_output(args,text=True);current=None;frames=[]
  for l in resolved.splitlines()+['0xffffffffffffffff']:
   if re.fullmatch('0x[0-9a-f]+',l):
    if current is not None:mapping[current]=frames
    current=int(l,16);frames=[]
   else:frames.append(l)
 totals=collections.Counter();symbols=collections.Counter();lines=collections.Counter();total=sum(w for w,_,_ in raw)
 for weight,sym,addr in raw:
  symbols[sym]+=weight
  frames=mapping.get(addr,[])
  locs=[x for x in frames if '/rustpower-amd/src/' in x]
  loc=locs[-1] if locs else ''
  if loc:lines[loc]+=weight
  fn=loc.rsplit('/',1)[-1].split(':')[0];m=re.search(r':(\d+)(?: |$)',loc);line=int(m[1]) if m else 0
  if fn=='amd_2.rs' and line:
   if line<211:g='kernel initialization'
   elif line<281:g='pivot selection and bucket removal'
   elif line<525:g='construct elimination element / arena'
   elif line<577:g='Scan1 external element degrees'
   elif line<653:g='Scan2 element filtering / absorption'
   elif line<692:g='Scan2 variable filtering / degree / hash'
   elif line<814:g='degree bounds / hash-bucket insertion'
   elif line<933:g='supervariable comparison / merging'
   elif line<999:g='restore degree buckets'
   elif line<1109:g='factor statistics'
   else:g='kernel output / parent compression'
  elif fn=='amd_2.rs':g='kernel unmapped line0'
  elif fn in ('postorder.rs','post_tree.rs'):g='postorder'
  elif fn=='powerflow.rs':g='Ybus validation / column metadata'
  elif fn=='on_demand.rs':g='entry table or on-demand storage'
  elif fn=='lazy_core.rs':g='on-demand kernel'
  else:g='other / unassigned'
  totals[g]+=weight
 result={'samples':len(raw),'weighted_period':total,'operations_self_percent':{k:100*v/total for k,v in totals.most_common()},'symbols_self_percent':{k:100*v/total for k,v in symbols.most_common()},'source_lines_self_percent':{k:100*v/total for k,v in lines.most_common()}}
 results[data.stem]=result
 data.with_name(data.stem+'-operations.json').write_text(json.dumps(result,indent=2))
 if data.stem.endswith('full'):print(data.stem,[(k,round(v,2)) for k,v in result['operations_self_percent'].items()])
 # A leaf-only sample chart, explicitly not a call-stack flamegraph.
 items=list(result['symbols_self_percent'].items())[:12];svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="420">','<rect width="100%" height="100%" fill="white"/>','<text x="15" y="24" font-family="sans-serif" font-size="17">CPU-clock leaf samples (no call stacks; not a flame graph)</text>']
 for i,(name,percent) in enumerate(items):
  y=48+i*28;svg.append(f'<rect x="440" y="{y}" width="{percent*9:.2f}" height="20" fill="#ee9955"/><text x="12" y="{y+15}" font-size="12" font-family="monospace">{html.escape(name[:55])}</text><text x="{445+percent*9:.2f}" y="{y+15}" font-size="12">{percent:.2f}%</text>')
 svg.append('</svg>');data.with_name(data.stem+'-hotspots.svg').write_text('\n'.join(svg))
(p/'operations-summary.json').write_text(json.dumps(results,indent=2))
