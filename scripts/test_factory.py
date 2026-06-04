import rustpower
import numpy as np
import pandas as pd

def test_factory():
    print("Testing Grid Factory (programmatic grid construction)...")
    
    grid = rustpower.PowerGrid()
    
    # 1. Set base
    grid.set_base(f_hz=50.0, sn_mva=100.0)
    
    # 2. Add elements
    grid.add_bus(id=0, vn_kv=110.0, name="Slack Bus")
    grid.add_bus(id=1, vn_kv=110.0, name="Load Bus")
    
    grid.add_ext_grid(bus=0, vm_pu=1.0, va_degree=0.0)
    grid.add_load(bus=1, p_mw=50.0, q_mvar=20.0)
    
    # Line: 0.1 + j0.2 Ohm/km, 10km
    grid.add_line(from_bus=0, to_bus=1, r_ohm_per_km=0.1, x_ohm_per_km=0.2, length_km=10.0)
    
    print(f"Grid constructed: {grid}")
    
    # 3. Solve
    print("Initializing and solving...")
    grid.init_pf()
    grid.solve()
    
    print(f"Converged: {grid.converged}")
    print(f"Iterations: {grid.iterations}")
    
    # 4. Check results
    res = grid.get_bus_results()
    df = pd.DataFrame(res)
    print("\nBus Results:")
    print(df)
    
    # Check if bus 1 voltage dropped
    vm_1 = df[df.bus_id == 1].vm_pu.values[0]
    print(f"\nBus 1 Voltage: {vm_1:.4f} pu")
    
    if vm_1 < 1.0:
        print("Success: Voltage drop observed as expected.")
    else:
        print("Warning: No voltage drop? Check parameters.")

if __name__ == "__main__":
    test_factory()
