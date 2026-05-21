import pandapower as pp
import pandapower.networks as nw
import pandas as pd

# Load IEEE 118 case
net = nw.case118()

# Run power flow
pp.runpp(net)

# Extract bus results
res_bus = net.res_bus[['vm_pu', 'va_degree']]

# List available columns to debug
print("Transformer columns:", net.trafo.columns.tolist())

# Extract available transformer data
cols = [c for c in ['hv_bus', 'lv_bus', 'tap_pos', 'shift_degree', 'tap_step_percent'] if c in net.trafo.columns]
trafo_data = net.trafo[cols]

print("\n--- IEEE 118 BUS RESULTS (First 10) ---")
print(res_bus.head(10))

print("\n--- IEEE 118 TRANSFORMER DATA ---")
print(trafo_data)

# Save to CSV for detailed comparison
res_bus.to_csv('ieee118_bus_results_py.csv')
trafo_data.to_csv('ieee118_trafo_data_py.csv')
