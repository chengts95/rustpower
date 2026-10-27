# AMD(Ybus) versus AMD(reduced J): ordering correspondence

This is the requested ordering study. It does not change Ybus/PQ/PV/Slack layout,
J assembly, physical J storage, KLU, or any production path. No timing or fill
quality claim is made from permutation similarity.

Inputs: actual39/118/9241, existing [PQ|PV|Slack] Ybus. Remove Slack rows/columns,
retain active structural diagonal and sort order. System C amd_l_order orders
that bus graph B. Its output matches ordinary native Rust AMD exactly on each
case. Same default dense10/aggressive true for the full reduced J C ordering.
The existing seeded Rust J ordering is a second comparison, not substituted for
the ordinary J reference.

Compare bus permutation with (1) angle-only subsequence of J's scalar output,
which contains each active bus exactly once, and (2) first occurrence of either
variable of each bus. In these cases the two projections have identical metrics.
This is comparison of final postordered outputs, not traces of successive pivot
selection inside AMD.

|case|active buses|same angle positions, scalar J|same angle positions, seeded J|edge direction agreement, scalar J|edge direction agreement, seeded J|
|---|---:|---:|---:|---:|---:|
|39|38|34|23|95.56%|95.56%|
|118|117|1|3|85.55%|89.02%|
|9241|9240|24|114|91.50%|92.81%|

Edge direction agreement asks, for each connected pair of active buses, whether
the same endpoint appears earlier in both output permutations. This examines
local topological ordering rather than requiring globally identical positions.

Global all-pair ordering agreement is99.72/65.05/33.91% for ordinary J and
97.87/84.03/83.36% for seeded J. These metrics alone do not establish elimination
quality: postorder can reorder large branches while retaining many local
relations. A metric mismatch is not evidence of worse fill, and agreement is not
a proof that ordinary bus AMD can replace J AMD.

Observed: no exact direct equality of the output projections. Local edge-order
agreement is high in the large case despite substantial global rearrangement.
Seeded J is closer globally on118/9241, not on39. There is evidence of a local
relationship worth investigating, but no recovered mapping or reuse algorithm
yet. These results do not justify returning to physically reordering J.

The next mathematical distinction to resolve is pivot-selection order versus
assembly-tree postorder: current output permutations mix both. Existing raw
orders allow examining the local differences without changing the solver.

Reproduce from repository root:

    cargo run --offline --example compare_ybus_j_order -- docs/experiments/amd-ybus/ordering-correspondence

Full inputs B, bus order, scalar J order, both bus projections and metrics are
saved in ordering-correspondence/case{39,118,9241}-{scalar_J,seeded_J}.json.
Source: examples/compare_ybus_j_order.rs. No new production algorithm implemented.
