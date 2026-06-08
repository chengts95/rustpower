"""
Tutorial 2: Loading External Cases and Inspecting Configuration ⚡️

In this tutorial, we will learn how to:
1. Load a grid model from a standard data file (IEEE 118-bus system).
2. Inspect the "factory settings" of the grid before running any simulation.
3. Solve the system and display the final results.
"""

import rustpower
import os

def main():
    print("--- Tutorial 2: Case Loading & Inspection ---")
    
    # 1. Loading an external case
    case_path = 'cases/IEEE118/data.zip'
    if not os.path.exists(case_path):
        print(f"Error: Case file not found at {case_path}")
        return

    print(f"Loading grid from: {case_path}...")
    # Loading automatically calls init_pf() for immediate inspection
    grid = rustpower.PowerGrid(case_path=case_path)
    
    print(f"Grid loaded! Total Buses: {grid.n_bus}, Total Lines: {grid.n_line}")

    # 2. Inspecting the Case Configuration (Input Parameters)
    # Let's see the physical parameters of the buses and lines we just loaded.
    print("\n--- Inspecting Input Parameters (First 5 records) ---")
    print(grid.display_case_buses())
    print(grid.display_case_lines())

    # 3. Simulation & Results
    print("\nSolving power flow...")
    grid.solve()
    
    if grid.converged:
        print(f"✅ Convergence reached in {grid.iterations} iterations.")
        
        # Quick display of the results
        print("\n--- Power Flow Results (Buses) ---")
        print(grid.display_buses())
        
        print("\n--- Power Flow Results (Lines) ---")
        print(grid.display_lines())
    else:
        print("❌ Power flow failed to converge.")

if __name__ == "__main__":
    main()
