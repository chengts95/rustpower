import rustpower
import numpy as np
import pandas as pd
import time

def test_wrapper():
    print(f"RustPower Version: {rustpower.version()}")
    print(f"Features: {rustpower.features()}")
    
    # 1. Load grid
    case_path = 'cases/IEEE118/data.zip'
    grid = rustpower.PowerGrid(case_path=case_path)
    print(f"\nInitial Grid: {grid}")
    
    # 2. Initialize and Solve
    print("Initializing and solving...")
    grid.init_pf()
    grid.solve()
    
    print(f"Converged: {grid.converged}")
    print(f"Iterations: {grid.iterations}")
    print(f"Buses: {grid.n_bus}, Lines: {grid.n_line}")
    
    # 3. Access Matrix and Vectors
    v = grid.v
    print(f"\nVoltage vector shape: {v.shape}")
    print(f"First 5 voltages (reordered): {v[:5]}")
    
    ybus_data = grid.y_bus
    print(f"Y-bus shape: {ybus_data['shape']}")
    print(f"Y-bus NNZ: {len(ybus_data['data'])}")
    
    # 4. Test Setters
    print("\nTesting single load update...")
    # IEEE 118, bus 1 (index 1) usually has a load
    grid.set_load(1, 60.0, 20.0) 
    grid.solve()
    print(f"Solved after single update. Iterations: {grid.iterations}")
    
    print("\nTesting batch load update...")
    bus_ids = [1, 2, 3, 4, 5]
    p_mws = [50.0, 60.0, 70.0, 80.0, 90.0]
    q_mvars = [20.0, 25.0, 30.0, 35.0, 40.0]
    
    start = time.perf_counter()
    grid.set_loads(bus_ids, p_mws, q_mvars)
    batch_time = time.perf_counter() - start
    print(f"Batch update (5 loads) took {batch_time*1000:.3f}ms")
    
    grid.solve()
    print(f"Solved after batch update. Converged: {grid.converged}")

    # 5. Extract Results to Pandas
    bus_res = grid.get_bus_results()
    df = pd.DataFrame(bus_res)
    print("\nBus Results (Head):")
    print(df.head())

if __name__ == "__main__":
    test_wrapper()
