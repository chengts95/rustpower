# Ybus column structure to AMD-permuted J storage

This is a structural experiment, not a benchmark or production assembly change.
Uses the current checked Rust scalar AMD and seeded AMD on actual39/118/9241
networks, with default ordering parameters. No algorithm changes, alternate
network or new solver is used.

## Exact correspondence

Let a=npq+npv, q=npq. In the already-permuted Ybus, column b has A_b active
entries and Q_b PQ entries, including its structural diagonal. Its J column
length is L_b=A_b+Q_b. Both theta_b and (for PQ) magnitude_b have that length.

Let p[k] be the old scalar variable at new AMD position k; inv is its inverse.
Map the old scalar column c to bus b=c for c<a, otherwise b=c-a. Then:

    new_column_length[k] = L_bus(p[k])
    new_column_offset[0] = 0
    new_column_offset[k+1] = new_column_offset[k] + L_bus(p[k])

This holds for ANY permutation and needs no complete J row adjacency. It maps
the effect of a known AMD permutation; column lengths alone do not determine
the AMD permutation. Use active/PQ retained counts, not raw full-Ybus nnz which
also includes slack entries.

Column-only permutation (new column k=old column p[k], rows unchanged) preserves
every local stored position:

    new_value_position = new_column_offset[k] + old_local_position

For simultaneous row and column permutation, map old row r to inv[r]. Sorted
CSC requires sorting these mapped row IDs within each column; the local rank
usually changes:

    new_value_position = new_column_offset[k] + rank_of(inv[r])

This rank and complete scatter map can be computed from Ybus row segments and
the known permutation once per structural/order change, without assembling J
numeric values. It is not necessarily the same simple prefix offset as the
original [PQ|PV|Slack] layout.

## Independent verification

1. Predict original/reordered scalar column offsets solely from Ybus's shared
   column layout and AMD p. Compare with actual scalar J CSC.
2. Build the independent reference from the existing JacobianCache: assign each
   stored nonzero its unique original position tag; permute rows/columns through
   COO and convert to sorted CSC. This tests structure/storage, not electrical
   numerical values.
3. Independently generate reordered rows and original-to-new value scatter from
   Ybus active/PQ segments, using predicted offsets; compare every row and tag
   position with the reference. Also verify column-only local-offset invariance.
4. Check whether final scalar ordering can collapse to a single contiguous block
   per bus, allowing either internal theta/magnitude orientation. Pair distances
   directly answer that question; no assumption that initial seed groups remain
   contiguous in final scalar postorder expansion.

All six configurations pass exact offsets, direct row generation, complete
scatter map and column-only local-position checks. Full arrays, per-bus counts,
scalar permutations/inverses and pair positions are retained in mapping-results.
No timing or numerical NR performance claim is made.

## Observed seeded ordering

| case | scalar columns | J stored nnz | adjacent PQ pairs / total | max pair distance | columns retaining original local row order after symmetric permutation |
|---|---:|---:|---:|---:|---:|
|39|67|445|23/29|4|0/67|
|118|181|1051|63/64|5|0/181|
|9241|17036|129412|7309/7796|30|44/17036|

Thus a known scalar AMD p gives a simple exact mapping of column nnz/prefix
sums, but this implementation's output is not always the expansion of a bus
permutation into contiguous1/2-variable blocks. A general scalar map is needed
to preserve the current output exactly. Restricting the algorithm to a bus-block
output would be a separate candidate, not an assumed equivalent transformation.

## Scope correction

The user has rejected physically changing J storage. This document only verifies
storage-mapping identities and does NOT establish the requested relationship
between AMD(Ybus) and AMD(J). It must not be treated as authorization to reorder
J assembly. The remaining research concerns ordering correspondence only.

## Historical engineering interpretation

The existing fast numerical assembly can keep its [PQ|PV|Slack] CSC layout; KLU
already accepts a symbolic ordering separately. Physical AMD row/column storage
reordering is not needed merely to use AMD. If exploring physical reordered
assembly, column-only movement preserves local direct-write positions, while
symmetric reordered sorted CSC needs an extra local-rank/scatter representation.
No such solver integration or performance tradeoff has been tested here.

Reproduce from persistent worktree:

    docs/experiments/amd-ybus/run-mapping-experiment.sh

Source: examples/inspect_amd_mapping.rs. Results: mapping-results/summary.json
and case{39,118,9241}-{scalar,seeded}.json.
