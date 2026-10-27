#!/usr/bin/env python3
"""Render perf's stackcollapse output as a self-contained, sample-count flame SVG."""
import pathlib,sys,html,hashlib,json
title=sys.argv[3] if len(sys.argv)>3 else 'Original scalar AMD on prebuilt J — CPU-clock call-stack samples'
folded=pathlib.Path(sys.argv[1]);dest=pathlib.Path(sys.argv[2]);root={'name':'all samples','value':0,'children':{}}
for line in folded.read_text().splitlines():
 try:stack,count=line.rsplit(' ',1);count=int(count)
 except ValueError:continue
 parts=stack.split(';')
 # Keep the complete AMD->driver chain; omit unreliable startup frames above C main.
 if 'main' in parts:parts=[parts[0]]+parts[parts.index('main'):]
 n=root;n['value']+=count
 for name in parts:
  n=n['children'].setdefault(name,{'name':name,'value':0,'children':{}});n['value']+=count
assert root['value']>0,'No stack samples; refuse to draw an empty flame graph'
def depth(n):return 1+max(map(depth,n['children'].values()),default=0)
w=1500;h=70+depth(root)*23
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">','<rect width="100%" height="100%" fill="#fffff5"/>','<style>text{font-family:monospace;font-size:11px}rect:hover{stroke:black;stroke-width:1}</style>',f'<text x="12" y="22" style="font-size:17px">{html.escape(title)}</text>',f'<text x="12" y="42">{root["value"]} samples; width = inclusive count; frames above C main omitted; hover for full name</text>']
def draw(n,x,d):
 width=(w-20)*n['value']/root['value'];y=h-15-(d+1)*23
 if width<.5:return
 rgb=int(hashlib.sha256(n['name'].encode()).hexdigest()[:2],16);color=f'rgb(245,{110+rgb//3},{55+rgb//5})';label=html.escape(n['name']);percent=100*n['value']/root['value']
 svg.append(f'<g><title>{label}: {n["value"]} samples ({percent:.2f}% inclusive)</title><rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="22" fill="{color}" stroke="white" stroke-width=".3"/>')
 cut=int((width-5)/6.6)
 if cut>2:svg.append(f'<text x="{x+3:.3f}" y="{y+15}">{html.escape(n["name"][:cut])}</text>')
 svg.append('</g>');offset=x
 for child in sorted(n['children'].values(),key=lambda v:v['name']):draw(child,offset,d+1);offset+=(w-20)*child['value']/root['value']
draw(root,10,0);svg.append('</svg>');dest.write_text('\n'.join(svg));dest.with_suffix('.tree.json').write_text(json.dumps(root))
