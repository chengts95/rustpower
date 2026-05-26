import pandapower as pp
import pandapower.networks as pn
import json

net39 = pn.case39()

print("=== case39 poly_cost ===")
print("columns:", list(net39.poly_cost.columns))
print(net39.poly_cost.to_string())

print("\n=== case39 gen (OPF-relevant columns) ===")
cols = [c for c in net39.gen.columns if any(k in c for k in ['p_mw','q_mvar','vm','cost'])]
print(net39.gen[cols].to_string())

print("\n=== case39 ext_grid ===")
print(net39.ext_grid.to_string())

# Run OPF
try:
    pp.runopp(net39, verbose=False)
    print("\n=== case39 OPF results ===")
    print("Objective (cost):", net39.res_cost)
    print("\nGenerator outputs (res_gen):")
    print(net39.res_gen[['p_mw','q_mvar','vm_pu']].to_string())
    print("\next_grid outputs (res_ext_grid):")
    print(net39.res_ext_grid[['p_mw','q_mvar','vm_pu']].to_string())
    print("\nBus voltages (sample):")
    print(net39.res_bus[['vm_pu','va_degree']].head(10).to_string())
except Exception as e:
    print("OPF failed:", e)

print("\n=== case118 poly_cost ===")
net118 = pn.case118()
print("columns:", list(net118.poly_cost.columns))
print("shape:", net118.poly_cost.shape)
print(net118.poly_cost.head(5).to_string())

try:
    pp.runopp(net118, verbose=False)
    print("\n=== case118 OPF results ===")
    print("Objective (cost):", net118.res_cost)
    print("\nGenerator outputs (res_gen) first 5:")
    print(net118.res_gen[['p_mw','q_mvar','vm_pu']].head(5).to_string())
except Exception as e:
    print("OPF failed:", e)
