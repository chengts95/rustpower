"""Self-judged verification of generator Q-limit enforcement (qlim=True).

Philosophy: validate against physics, not against another tool. pandapower is
used only as the input data source (case118) and for an informational
comparison — its transformer model differs (it cannot represent capacitive
branches), so cross-tool voltage agreement is NOT the acceptance criterion.

Physics criteria for a correct qlim solution:
  K1. The power flow converges.
  K2. Every generator bus's reactive output lies within its aggregate
      [sum(qmin), sum(qmax)] band.
  K3. Buses whose Q is strictly inside the band still hold their voltage
      setpoint (remained PV).
  K4. Buses that left their setpoint sit exactly at a Q limit (demoted to PQ
      with clamped Q) — no bus may both leave the setpoint AND be off-limit.
  K5. Without enforcement (qlim=False) at least one of those bands is
      violated, i.e. enforcement actually did something.
"""
import sys
import numpy as np
import pandapower as pp
import pandapower.networks as nw
import rustpower as rp

Q_TOL_MVAR = 1e-3   # clamping/aggregation tolerance
VM_TOL = 1e-6       # setpoint holding tolerance
MARGIN_MVAR = 1.0   # "strictly inside the band" margin

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        sys.exit(1)

def gen_bus_tables(net):
    """Per-bus aggregate gen setpoints/limits, load and shunt data (inputs)."""
    gens = net.gen[net.gen.in_service]
    by_bus = gens.groupby("bus").agg(
        qmin=("min_q_mvar", "sum"), qmax=("max_q_mvar", "sum"),
        vm=("vm_pu", "first"))
    loads = net.load[net.load.in_service].groupby("bus").agg(
        q=("q_mvar", "sum"))
    shunts = net.shunt[net.shunt.in_service].groupby("bus").agg(
        q=("q_mvar", "sum"))
    return by_bus, loads, shunts

def gen_q_by_bus(grid, gen_buses, load_q, shunt_q):
    """Reactive output of the generators at each gen bus, from OUR solution.
    res_bus q is consumption-positive and includes everything embedded in the
    Y-bus (lines, trafos, SHUNTS): q_res = Q_load + Q_shunt*vm^2 - Q_gen."""
    res = grid.res_bus
    q_res = dict(zip(res["bus_id"].astype(int), res["q_mvar"]))
    vm = dict(zip(res["bus_id"].astype(int), res["vm_pu"]))
    return {
        b: float(load_q.get(b, 0.0))
           + float(shunt_q.get(b, 0.0)) * vm[b] ** 2
           - q_res[b]
        for b in gen_buses
    }

net = nw.case118()
gen_lim, loads, shunts = gen_bus_tables(net)
load_q = loads["q"].to_dict()
shunt_q = shunts["q"].to_dict()
slack_buses = set(net.ext_grid.bus.astype(int))
gen_buses = [int(b) for b in gen_lim.index if int(b) not in slack_buses]

# --- K5 first: plain run must violate some band --------------------------
g_plain = rp.PowerGrid()
g_plain.from_pp_net(nw.case118())
report = g_plain.solve()
check("plain solve converges", bool(report), f"({report!r})")
q_plain = gen_q_by_bus(g_plain, gen_buses, load_q, shunt_q)
viol_plain = [b for b in gen_buses
              if q_plain[b] < gen_lim.loc[b, "qmin"] - Q_TOL_MVAR
              or q_plain[b] > gen_lim.loc[b, "qmax"] + Q_TOL_MVAR]
check("K5: limits bind without enforcement", len(viol_plain) > 0,
      f"({len(viol_plain)} gen buses out of band)")

# --- qlim run --------------------------------------------------------------
grid = rp.PowerGrid(qlim=True)
grid.from_pp_net(nw.case118())
report = grid.solve()
check("K1: qlim solve converges", bool(report), f"({report!r})")
vm = {int(b): float(v) for b, v in zip(grid.res_bus["bus_id"], grid.res_bus["vm_pu"])}
q_gen = gen_q_by_bus(grid, gen_buses, load_q, shunt_q)

# K2: all gen buses inside their aggregate band
out_of_band = {b: q_gen[b] for b in gen_buses
               if q_gen[b] < gen_lim.loc[b, "qmin"] - Q_TOL_MVAR
               or q_gen[b] > gen_lim.loc[b, "qmax"] + Q_TOL_MVAR}
check("K2: every gen bus within its Q band", not out_of_band,
      f"(violations: {out_of_band})")

held, demoted, inconsistent = [], [], []
for b in gen_buses:
    at_setpoint = abs(vm[b] - gen_lim.loc[b, "vm"]) < VM_TOL
    inside = (gen_lim.loc[b, "qmin"] + MARGIN_MVAR < q_gen[b]
              < gen_lim.loc[b, "qmax"] - MARGIN_MVAR)
    at_limit = (abs(q_gen[b] - gen_lim.loc[b, "qmin"]) < Q_TOL_MVAR
                or abs(q_gen[b] - gen_lim.loc[b, "qmax"]) < Q_TOL_MVAR)
    if at_setpoint:
        held.append(b)
    elif at_limit:
        demoted.append(b)
    else:
        inconsistent.append((b, vm[b], gen_lim.loc[b, "vm"], q_gen[b]))
    # K3: strictly-inside gens must hold the setpoint
    if inside and not at_setpoint:
        inconsistent.append((b, vm[b], gen_lim.loc[b, "vm"], q_gen[b]))

check("K3/K4: every gen bus is either PV-at-setpoint or PQ-at-limit",
      not inconsistent, f"(inconsistent: {inconsistent[:5]})")
check("qlim demoted at least one bus", len(demoted) > 0,
      f"({len(held)} held PV, {len(demoted)} demoted to Q-limit)")

# --- repeated qlim solves stay consistent ----------------------------------
v1 = grid.v.copy()
report2 = grid.solve()
check("repeated qlim solve converges", bool(report2))
check("repeated qlim solve consistent",
      np.linalg.norm(grid.v - v1) < 1e-7,
      f"(diff={np.linalg.norm(grid.v - v1):.2e})")

# --- informational only: cross-tool diff (NOT a criterion) -----------------
net_q = nw.case118()
pp.runpp(net_q, enforce_q_lims=True)
d = np.max(np.abs(np.abs(grid.v) - net_q.res_bus.vm_pu.values))
print(f"\n[INFO] max |vm - pandapower(enforce_q_lims)| = {d:.2e} "
      "(transformer models differ; informational only)")

print("\nAll qlim checks passed.")
