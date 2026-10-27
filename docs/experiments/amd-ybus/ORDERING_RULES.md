# Concrete rules relating bus AMD and J AMD

Scope: derive usable relationships, not output similarity scores. Original
PQ/PV/Slack assembly/layout is untouched. Diagnostics are an opt-in trace
feature in the experimental Rust crate. Pivots are observed before removing the
chosen degree-list head; no selection/update/absorption/postorder logic changes.
All six final orders are asserted equal to the already saved C scalar/seeded
references. These experiments do not claim a completed ordering reuse algorithm.

## 1. Initial degree buckets split by neighbor DOFs

Let d(b) be off-diagonal bus degree; t(b) the number of PV neighbors. For
PQ weight2 and PV weight1:

    J group external degree = 2*d(b) - t(b)
    J scalar variable degree = 2*d(b) - t(b) + own_weight(b) - 1

Verified for all38/117/9240 active bus columns. Ordinary Ybus degree alone
throws away t(b); an unweighted bus bucket does not map to one weighted bucket.
E.g. degree2 splits into weighted degrees2/3/4 on118 and9241. This explains an
actual change in decisions, rather than merely observing unequal permutations.

Trace9241 first choice (indices in the unchanged prepermuted Ybus):

- Ordinary bus AMD selects PV9239, whose only neighbor is PQ558. Bus degree1,
  but expanded group external degree2.
- Seeded J AMD selects PV9238, whose only neighbor is PV8005. Bus degree1,
  expanded group external degree1.

Both candidates are tied in the unweighted graph; weighting separates them.
The original degree1 bucket has1552 candidates; its weighted degree1 subset has
203 candidates. Thus divergence starts at pivot0, not only at final tree
postorder.118 similarly diverges at0;39 has10 identical leading representative
pivots before a change. Representative-pivot streams include dynamic group
merging and are not complete expanded scalar elimination permutations.

The degree formula is an INITIAL rule. After fill and merging, a group may
contain several buses; current external degree is the sum of remaining group
weights over the union of external neighbors. One cannot keep using the
original t(b) after topology changes during symbolic elimination.

## 2. No-fill condition is invariant under the supported block expansion

For the same chosen WHOLE bus group b, symbolic elimination completes its
neighbor groups into a clique. If an absent neighbor-bus pair i,j is created,
it corresponds to w_i*w_j scalar edges in expanded J. All weights are positive.
Therefore:

    no new bus edges <=> no new scalar edges
    <=> the current bus neighbors already form a clique

This is a structural statement about grouped symmetric elimination of the
validated J pattern; it is not a promise about KLU numerical pivoting or
arbitrary scalar elimination interleavings.

Initial input counts:

|case|active buses|leaf buses|all no-fill bus groups (simplicial)|
|---|---:|---:|---:|
|39|38|8|9|
|118|117|7|19|
|9241|9240|1552|2015|

The same invariant applies to a common remaining quotient topology after
eliminating the same groups. Thus no-fill certificates/structure can be obtained
from the bus graph without rediscovering the duplicated J neighbors. Initial
certificates need updating when elimination changes the remaining graph.
Pre-removing groups would change the selected heuristic order and requires a
separate fill/solve/performance comparison; it has NOT been implemented here.

## What this supports, and what it does not

A useful path is sharing the bus structural representation and using weighted
candidate buckets/certificates. There is not yet a fixed transformation from a
completed unweighted bus permutation to the original scalar AMD permutation.
The experiments locate why a global permutation-only mapping loses information:
the unweighted sequence already made a tie decision before accounting for DOFs.
It remains possible that parts of elimination work can be reused; this report
does not conclude impossibility or substitute physical J reordering.

Reproduce:

    cargo run --offline --features experimental_amd_trace --example trace_ybus_j_order -- docs/experiments/amd-ybus/ordering-trace
    python3 docs/experiments/amd-ybus/analyze-order-invariants.py

Requires the saved ordering-correspondence reference files, reproducible via
compare_ybus_j_order. Trace stores complete pivot streams and first8 candidate
states. For feature-off builds the trace code and calls do not exist. No timers
or speedup assertions were added.
