import json,itertools,pathlib
root=pathlib.Path(__file__).resolve().parent
reports=[]
for case in ['39','118','9241']:
 data=json.loads((root/f'ordering-correspondence/case{case}-scalar_J.json').read_text())
 p=data['bus_pattern_offsets'];r=data['bus_pattern_rows'];n=len(p)-1;q=data['summary']['npq']
 trace=json.loads((root/f'ordering-trace/case{case}.json').read_text())
 adj=[set(r[p[b]:p[b+1]])-{b} for b in range(n)]
 weights=[2 if b<q else 1 for b in range(n)]
 simplicial=[];leaves=[]
 for b,nei in enumerate(adj):
  missing=[(i,j) for i,j in itertools.combinations(nei,2) if j not in adj[i]]
  assert (not missing)==(sum(weights[i]*weights[j] for i,j in missing)==0)
  if not missing:simplicial.append(b)
  if len(nei)==1:leaves.append(b)
 pair=trace['summary']['first_pivot_divergence'];examples=[]
 if pair and pair['pivot_step']==0:
  for b in [pair['bus_choice'],pair['seeded_choice']]:
   examples.append({'bus':b,'type':'PQ' if b<q else 'PV','neighbors':sorted(adj[b]),'neighbor_types':['PQ' if i<q else 'PV' for i in sorted(adj[b])],'unweighted_degree':len(adj[b]),'expanded_group_external_degree':sum(weights[i] for i in adj[b])})
 result={'case':case,'active_buses':n,'leaves':len(leaves),'initial_simplicial_buses':len(simplicial),'no_fill_predicate_equivalence_checked':n,'first_divergent_choices':examples,'zero_fill_buses':simplicial,'leaf_buses':leaves}
 reports.append(result)
 print(case,'leaves',len(leaves),'zero-fill bus groups',len(simplicial),'first choices',examples)
(root/'ordering-trace/structural-invariants.json').write_text(json.dumps(reports,indent=2)+'\n')
