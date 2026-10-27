# Specialized Rust structure API

`rustpower_amd::powerflow` owns the power-flow-specific types and logic. It has
no KLU types, C allocator calls, or dependency on RustPower's solver traits.

```rust
use rustpower_amd::{Control, powerflow::{YbusPattern, OrderingWorkspace}};
# fn example(p: &[usize], rows: &[usize], npv: usize, npq: usize) -> Result<(), rustpower_amd::Status> {
let pattern = YbusPattern::analyze(p, rows, npv, npq)?;
let j = pattern.reduced_jacobian();
let mut workspace = OrderingWorkspace::new();
let ordering = workspace.order(&pattern, &Control::default())?;
let permutation = ordering.permutation();
# let _ = (j, permutation);
# Ok(())
# }
```

The same immutable pattern supplies both operations. `columns()` exposes named
counts and diagonal positions for future numerical fill integration. The pattern
borrows the source CSC arrays immutably; Rust prevents mutation while it is used.
It is not a cache detached from the identity/lifetime of its source matrix.

For active bus adjacency B and PQ subset Q, the generated scalar structure is
`[[B, B[:,Q]], [B[Q,:], B[Q,Q]]]` with angle variables first and PQ magnitudes last.
For column j: active count a, PQ count q, weight w=2 for PQ or1 for PV:

- J column length: a+q; repeated for the second variable of a PQ bus.
- Initial AMD representative adjacency length: a-1.
- Weighted external degree: a+q-w.
- J nnz: sum_j w_j*(a_j+q_j).

`YbusPattern::analyze` validates and derives these boundaries once. Generating J
and initializing AMD do not reclassify neighbors or recount the structure.
`QuotientGraph::initialize` privately copies active off-diagonal row IDs to a
mutable arena, using the stored diagonal position to skip the diagonal. That
copy remains necessary because AMD overwrites and compacts adjacency while
representing fill; the read-only Ybus cannot be its mutable workspace.

The private `QuotientGraph` owns named offsets, list lengths, adjacency storage,
external degrees and live-entry count. `OrderingWorkspace` resets that graph for
every order call and reuses vector allocations, including across size changes.
Callers neither allocate Iw nor pass iwlen/pfree, encoded parents or parallel
mutable arrays. `Ordering` provides named permutation, inverse and Info access.
Legacy order_ybus APIs delegate to these types for compatibility.

This is a staged encapsulation: the imported elimination core still uses its
original internal index-array loops and allocates its own auxiliary vectors.
No claim of a complete idiomatic rewrite, a compact active-only core, complete
workspace reuse or a performance improvement is made. Those would be separate
changes with C-reference checks. Current unsafe mode is an explicit unsafe
method; checked order is the normal interface.

The typed Rust API feeds the optional `klu::KluOrdering` adapter. The root
application selects it through a distinct experimental AmdNewtonSolver/plugin.
Original KLU, Solve, native NewtonSolver and generic NR remain unchanged.

Verified on39/118/9241 and workspace reuse9241->39: scalar J CSC and all three
column metadata arrays equal the existing JacobianCache; checked/unsafe order
matches our seeded C permutation, inverse is consistent. Existing lifecycle
regressions and scalar C-oracle tests also pass. Earlier benchmark numbers refer
to the previous preparation implementation, not measured speed of this API.
