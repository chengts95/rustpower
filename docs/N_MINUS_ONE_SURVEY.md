# N−1 correctness survey on main + the existing 0.5.5 load-reader fix

Base: local main `3eedbb1`. The existing `origin/0.5.5` delta was applied in an
isolated checkout: changelog and serde defaults for `Load.const_i_percent` and
`Load.const_z_percent`, including its existing CSV regression test. That test
passes. The original main checkout and AMD research checkout are preserved.
No remote was fetched or changed.

## Run

Build the Python extension with `python,klu_dyn` and load it from a separate
PYTHONPATH package, then run from this checkout:

```sh
python scripts/check_n_minus_one.py --output /tmp/n-minus-one-results.json
```

This is a correctness survey with explicit per-scenario outcomes. Its nonzero
exit status includes Rust panics observed at the Python boundary. Pandapower is
run with `lightsim2grid=False`, flat starts, max 50 iterations and 1e-10 tolerance.
The Rust build uses the original dynamic KLU solver. There is no AMD feature.
The installed virtualenv extension was preserved.

## Coverage and observed outcomes

Cases 14, 39 and 57: every initially in-service line and two-winding transformer
is independently taken out, then restored. There are 146 unique contingencies,
repeated with solver caching disabled and enabled: 292 records. Case 118 is
excluded.

133 contingencies preserve supply connectivity (266 records). The survey checks
fresh import and reimport into an existing grid, bus magnitude/angle, line and
transformer active/reactive flows, terminal currents and loading against
Pandapower. The 106 connected line contingencies additionally exercise the
public handle's `in_service` setter (212 records): outage, zero offline flow,
and restoration to base-case voltages. Connected comparisons pass in this run.
Baseline restoration is also checked after reimport for all records.

13 unique contingencies produce buses disconnected from all Slack buses (26
records). Of these, 10 trigger `PanicException: Ybus is missing a structural
diagonal entry` (20 records); the other 3 return an unconverged solve (6 records).
The panic occurs with either cache setting: the uncached NR path also constructs
a temporary V4 Jacobian cache.

### Minimal reproduction

```python
import pandapower.networks as pn
import rustpower as rp
net = pn.case14()
net.trafo.at[3, "in_service"] = False  # bus 7 loses supply
rp.PowerGrid.from_pandapower(net).solve(max_iter=50, tol=1e-10)
```

The missing structural diagonal is unwrapped with `expect` in
`src/basic/jacobian_cache.rs:40`, called from both cached/uncached preparation
in `src/basic/newtonpf.rs`. Topology validation currently checks references and
switch targets, while Slack reachability is not represented as an explicit
solve outcome. Island handling needs a deliberate API policy before a repair:
report an island status retaining the original case for restoration, or cut
unsupplied islands and solve the supplied remainder.

## Current user decision

Skip pre-solve island/reachability checks. Attempt the numerical solve and
record failures for later diagnosis. Focus on fixed-pattern symbolic reuse
through service-state changes; see [in_service investigation](IN_SERVICE_N_MINUS_ONE.md).
The policy discussion above records the earlier survey stage.

## Scope for a subsequent N−1 suite

- Define island and nonconvergence outcomes separately from convergence.
- Select outages by persistent equipment ID, including parallel branches.
  The current line lookup takes endpoint buses and returns the first match;
  imported transformer handles have no public lookup or in-service setter.
  Transformer contingencies here therefore use reimport.
- Add topology mutation/restoration tests for transformer handles once provided.
- Keep thermal/voltage limit violations distinct from numerical correctness.
  This survey compares loading values; it does not implement security verdicts.
- Expand to generator outages and large cases after the small-case outcomes
  and device-selection APIs are established.

Parallel-endpoint result rows are skipped by the endpoint-only comparison;
these three cases produced no ambiguous connected line-handle selections.
The survey records that limitation explicitly when it occurs.

Raw outputs and logs are stored at
`/home/cts/Documents/Codex/2026-10-26/task/main-n1-results.json`,
`main-n1-survey.log`, `main-n1-build.log` and `main-n1-default-test.log`.
