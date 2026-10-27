# Ybus-seeded AMD

`amd_super2.c` and `amd_internal.h` originate from SuiteSparse v7.14.1 AMD3.3.4 (BSD-3-Clause, original notices retained). The AMD2 function is renamed and initialized with PQ supervariables of size2, PV size1 and absorbed Vm parents. Weighted external degrees, dense variables and empty principal initialization account for these initial groups. Pivot selection, absorption, elimination, statistics, path compression and postorder are otherwise unchanged. Original source and exact experimental patch are preserved under docs/experiments/amd-ybus/restored-prototype.

`ybus_order.c` builds the quotient adjacency directly from validated Ybus, calls the seeded core and constructs KLU Symbolic through public klu_l_malloc/free APIs. There is no global Ybus state and no full J graph construction in this ordering path. The Rust caller validates the structural preconditions and provides original-path fallback. Unsupported AMD/KLU header major versions return unavailable. BTF remains user-controlled; optimization requires a connected active structural graph, whose full J is one strongly connected block with a structural diagonal/matching. This permits preserving BTF's one-block result without changing its configured default.


## Feature and preservation

This directory is owned by `rustpower-amd`. Its optional `c_amd` feature enables
`cc` and compiles the two C sources; default Rust ordering builds retain their
Rust-only dependency boundary. `klu::KluOrdering` provides cache/preparation
state externally to the unchanged `rustpower-sol-klu` owner. The root application
opts in through the new AmdNewtonSolver or AmdOrderingPlugin. Existing KLU,
DefaultSolver, generic Solve/NR and ordinary native/ECS construction retain the
original path.

Upstream AMD BSD license and attribution are retained in LICENSE.txt, C sources
and amd_internal.h. ybus_order.c is the one-block Symbolic adapter using public
KLU allocation/free APIs. Active original/measurement C is under the crate's
experiments directory with its existing notices and is excluded from packaging.
Historical hashes/results remain under docs/experiments/amd-ybus.
