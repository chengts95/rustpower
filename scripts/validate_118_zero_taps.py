import pandapower as pp
import pandapower.networks as nw
import pandas as pd

# Load IEEE 118 case
net = nw.case118()

# Zero out all transformer taps and shifts
net.trafo.tap_pos = 0.0
net.trafo.shift_degree = 0.0

# Run power flow
pp.runpp(net)

# Extract bus results
res_bus = net.res_bus[['vm_pu', 'va_degree']]

print("--- IEEE 118 BUS RESULTS (ZERO TAPS, First 10) ---")
print(res_bus.head(10))

# Save for comparison
res_bus.to_csv('ieee118_results_zero_taps_py.csv')
