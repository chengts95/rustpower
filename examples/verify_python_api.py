"""Contract tests for the Python API v2 (docs/design/python_api_v2.md).

Covers: initialization, repeated solves, immediate-mode property writes,
transactional edit()/abort, in_service topology changes, None lookup
semantics, validation errors, and the reset/re-ingest path.
"""
import os
import sys
import numpy as np

import rustpower as rp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = os.path.join(ROOT, "cases", "IEEE118", "data.zip")

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        sys.exit(1)

print("rustpower", rp.version(), "features:", rp.features())

# --- 1. Batch solve (scenario A) ----------------------------------------
grid = rp.PowerGrid(case_path=CASE)
report = grid.solve()
check("initial solve converges", bool(report) and grid.converged,
      f"({report!r})")
v0 = grid.v.copy()

res = grid.res_bus
check("res_bus is a DataFrame with 118 rows",
      len(res) == 118 and "vm_pu" in res.columns and "va_degree" in res.columns)
check("res_line has flows", len(grid.res_line) > 0)
desc = grid.describe()
check("describe() reports 118 buses",
      int(desc[desc["element"] == "bus"]["count"].iloc[0]) == 118)

report2 = grid.solve()
check("repeated solve identical", np.linalg.norm(v0 - grid.v) < 1e-10,
      f"(rebuild={report2.rebuild})")

# --- 2. Parameter loop (scenario B): property writes, no init_pf ---------
load = grid.load(bus=0)
check("load(bus=0) found", load is not None)
orig_p = load.p_mw
check("p_mw getter reads back", abs(orig_p - 51.0) < 1.0, f"(p={orig_p:.2f})")

load.p_mw = orig_p * 1.10
report = grid.solve()
check("solve after property write converges", bool(report),
      f"(iterations={report.iterations}, rebuild={report.rebuild})")
check("property write took incremental path", report.rebuild == "incremental")
v2 = grid.v.copy()
d02 = np.linalg.norm(v0 - v2)
check("solution actually changed", 1e-7 < d02 < 0.1, f"(|dv|={d02:.2e})")

# Cross-check: full rebuild agrees with the incremental path
grid.init_pf()
grid.solve()
check("full rebuild matches incremental path",
      np.linalg.norm(v2 - grid.v) < 1e-8,
      f"(diff={np.linalg.norm(v2 - grid.v):.2e})")

load.p_mw = orig_p
grid.solve()
check("reverting load recovers original solution",
      np.linalg.norm(v0 - grid.v) < 1e-7,
      f"(diff={np.linalg.norm(v0 - grid.v):.2e})")

# --- 2b. Exactness under heavy repeated edits (re-aggregation, no drift) --
for k in range(2000):
    load.p_mw = orig_p * (1.0 + 0.001 * (k % 7))
load.p_mw = orig_p * 1.10
grid.solve()
v_hammered = grid.v.copy()
grid.init_pf()      # ground truth: full rebuild from case data
grid.solve()
# Both solutions are Newton-converged to tol=1e-8 from different warm starts,
# so they agree to solver tolerance; aggregation drift would blow far past it.
check("2000 repeated edits stay at solver tolerance",
      np.linalg.norm(v_hammered - grid.v) < 1e-8,
      f"(diff={np.linalg.norm(v_hammered - grid.v):.2e})")
load.p_mw = orig_p
grid.solve()

# --- 3. Generator setpoint (PV bus holds magnitude) -----------------------
g = grid.gen()
check("gen() found", g is not None, f"(bus {g.bus})")
vm_target = g.vm_pu * 1.01
g.vm_pu = vm_target
report = grid.solve()
check("solve after vm_pu write converges", bool(report),
      f"(rebuild={report.rebuild})")
vm_res = grid.bus(g.bus).vm_pu
check("PV bus holds new setpoint", abs(vm_res - vm_target) < 1e-6,
      f"(set {vm_target:.4f}, got {vm_res:.4f})")

# --- 3b. v ordering contract and warm-start API ----------------------------
# v is indexed by bus id: must agree with res_bus rows keyed by bus_id
res = grid.res_bus
vm_by_id = np.abs(grid.v)[res["bus_id"].to_numpy()]
check("v is bus-id ordered (matches res_bus)",
      float(np.max(np.abs(vm_by_id - res["vm_pu"].to_numpy()))) < 1e-12)
b7 = grid.bus(7)
check("v[7] equals bus(7).vm_pu", abs(abs(grid.v[7]) - b7.vm_pu) < 1e-12)

# warm start: passing the solved vector back converges immediately
v_prev = grid.v.copy()
rep = grid.solve(v_init=v_prev)
check("warm start from solution converges fast",
      bool(rep) and rep.iterations <= 2, f"(iterations={rep.iterations})")
check("warm start reproduces solution",
      np.linalg.norm(grid.v - v_prev) < 1e-8)

# v_init is a pure warm start: a flat (wrong) guess must not move PV setpoints
g0 = grid.gen()
vm_set = g0.vm_pu
rep = grid.solve(v_init=np.full(len(v_prev), 0.95 + 0.0j))
check("flat-start solve converges", bool(rep), f"(iterations={rep.iterations})")
vm_after = grid.bus(g0.bus).vm_pu
check("PV setpoint unaffected by v_init", abs(vm_after - vm_set) < 1e-6,
      f"(set {vm_set:.4f}, got {vm_after:.4f})")
check("flat start reaches the same solution",
      np.linalg.norm(grid.v - v_prev) < 1e-7,
      f"(diff={np.linalg.norm(grid.v - v_prev):.2e})")

# property-assignment form
grid.v = v_prev
rep = grid.solve()
check("v setter warm start works", bool(rep) and rep.iterations <= 2,
      f"(iterations={rep.iterations})")

# --- 4. Transactional editor (scenario C) ---------------------------------
g2 = rp.PowerGrid()
with g2.edit() as e:
    b0, _ = e.add_bus(110.0)
    b1, _ = e.add_bus(110.0)
    e.add_ext_grid(b0, vm_pu=1.02)
    # two parallel lines so we can switch one off later without islanding
    line_a = e.add_line(b0, b1, 10.0, r_ohm_per_km=0.06, x_ohm_per_km=0.4)
    line_b = e.add_line(b0, b1, 10.0, r_ohm_per_km=0.06, x_ohm_per_km=0.4)
    e.add_load(b1, 20.0, 5.0)
report = g2.solve()       # no init_pf anywhere
check("editor grid solves without init_pf", bool(report),
      f"(rebuild={report.rebuild})")
check("commit marked topology dirty -> full rebuild", report.rebuild == "full")
vm_both = g2.bus(b1).vm_pu
check("proxy result access works", 0.9 < vm_both < 1.02, f"(vm={vm_both:.4f})")

# --- 5. Transaction abort leaves the world untouched ----------------------
n_before = g2.n_bus
try:
    with g2.edit() as e:
        e.add_bus(110.0)
        e.add_bus(110.0)
        raise ValueError("boom")
except ValueError:
    pass
check("abort rolled back created buses", g2.n_bus == n_before,
      f"(n_bus={g2.n_bus})")
report = g2.solve()
check("grid still solves after abort", bool(report))

# --- 6. in_service toggle = topology class --------------------------------
line_b.in_service = False
report = g2.solve()
check("solve after in_service=False converges", bool(report),
      f"(rebuild={report.rebuild})")
check("in_service triggered full rebuild", report.rebuild == "full")
vm_single = g2.bus(b1).vm_pu
check("dropping one parallel line lowers voltage", vm_single < vm_both - 1e-6,
      f"({vm_both:.5f} -> {vm_single:.5f})")
line_b.in_service = True
g2.solve()
check("re-enabling line restores voltage",
      abs(g2.bus(b1).vm_pu - vm_both) < 1e-9)

# --- 7. None lookup semantics (D3) ----------------------------------------
check("load at absent bus is None (no exception)", grid.load(bus=99999) is None)
check("absent bus is None", grid.bus(99999) is None)
check("absent line is None", g2.line(0, 99999) is None)

# --- 8. Validation errors are readable, not crashes -----------------------
empty = rp.PowerGrid()
try:
    empty.solve()
    check("empty grid raises", False)
except RuntimeError as ex:
    check("empty grid raises RuntimeError", "empty" in str(ex).lower())

no_slack = rp.PowerGrid()
with no_slack.edit() as e:
    b, _ = e.add_bus(110.0)
    e.add_load(b, 1.0, 0.0)
try:
    no_slack.solve()
    check("no-slack grid raises", False)
except RuntimeError as ex:
    check("no-slack grid raises RuntimeError", "slack" in str(ex).lower())

# --- 9. Reset / re-ingest path (full reload, no residue) ------------------
net = rp.load_csv_zip(CASE)
g3 = rp.PowerGrid()
g3.load_network(net)
g3.solve()
v_first = g3.v.copy()
g3.load_network(net)      # reload onto a populated grid
report = g3.solve()
check("reload converges", bool(report))
check("reload equals fresh load (no residue)",
      np.linalg.norm(v_first - g3.v) < 1e-12 and len(g3.res_bus) == 118,
      f"(diff={np.linalg.norm(v_first - g3.v):.2e})")

print("\nAll checks passed.")
