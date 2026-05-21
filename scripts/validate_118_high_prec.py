import pandapower as pp
import pandapower.networks as nw
import pandas as pd
import numpy as np

# Load IEEE 118 case
net = nw.case118()

# Run power flow with high tolerance
pp.runpp(net, tolerance_mva=1e-10)

# Extract bus results with high precision
res_bus = net.res_bus[['vm_pu', 'va_degree']].copy()

print("--- IEEE 118 SLACK BUS INFO ---")
slack_bus = net.ext_grid.bus.values[0]
print(f"Slack Bus Index: {slack_bus}")
print(f"Slack Bus Angle: {net.ext_grid.va_degree.values[0]}")

print("\n--- IEEE 118 BUS RESULTS (First 10, High Precision) ---")
pd.options.display.float_format = '{:.10f}'.format
print(res_bus.head(10))
