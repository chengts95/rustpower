import rustpower as rp
import pandas as pd
import numpy as np

def main():
    print("--- Solver Stability Diagnosis ---")
    
    # 1. Load the 118-bus case
    case_path = "cases/IEEE118/data.zip"
    print(f"Loading case from: {case_path}")
    grid = rp.PowerGrid(case_path=case_path)
    
    # Initial solve
    grid.solve()
    v_initial = grid.v.copy()
    print(f"Initial Convergence: {grid.converged}, Iterations: {grid.iterations}")

    # 2. Find an existing load to modify
    load_df = grid.display_case_loads()
    # Let's pick the first load
    target_bus = int(load_df.iloc[0]['bus'])
    orig_p = load_df.iloc[0]['p_mw']
    print(f"\nTarget Bus: {target_bus}, Original P: {orig_p:.2f} MW")
    
    # Modify by ONLY 1%
    new_p = orig_p * 1.01
    print(f"Modifying load to: {new_p:.2f} MW (1% increase)")
    
    # Get load handle
    load_h = grid.load(target_bus)
    load_h.set_p(new_p)
    
    # 3. Rebuild and solve
    print("Running init_pf()...")
    grid.init_pf()
    print("Solving...")
    grid.solve()
    
    if grid.converged:
        v_new = grid.v
        diff = np.linalg.norm(v_initial - v_new)
        print(f"✅ SUCCESS: Converged in {grid.iterations} iterations. L2-diff: {diff:.8f}")
    else:
        print("❌ FAILURE: Diverged after 1% modification!")
        # Print some results to see the damage
        res_df = grid.res_bus
        print("\nFirst 5 bus results (after divergence):")
        print(res_df.head())

if __name__ == "__main__":
    main()
