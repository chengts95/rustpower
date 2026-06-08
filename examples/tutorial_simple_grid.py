"""
Tutorial 1: Building Your First Power System ⚡️

In this tutorial, we will learn how to:
1. Create a 110kV two-bus system.
2. Define a Slack Bus and a Load.
3. Connect them with a transmission line.
4. Solve and inspect results.
5. Modify parameters and re-solve.
"""

import rustpower

def main():
    # --- Step 1: Building the Grid ---
    grid = rustpower.PowerGrid()
    
    print("Building grid...")
    sub_id, sub_h = grid.add_bus(vn_kv=110.0, name="Substation")
    grid.add_ext_grid(bus=sub_id, vm_pu=1.0)
    
    fac_id, fac_h = grid.add_bus(vn_kv=110.0, name="Factory")
    grid.add_load(bus=fac_id, p_mw=30.0, q_mvar=10.0)
    
    grid.add_line(from_bus=sub_id, to_bus=fac_id, length_km=20.0, 
                  r_ohm_per_km=0.1, x_ohm_per_km=0.1, max_i_ka=0.2)
    
    print(f"Grid built! Nodes: {grid.n_bus}, Lines: {grid.n_line}")

    # --- Step 2: Initialize & Solve ---
    grid.init_pf()
    grid.solve()
    
    if grid.converged:
        print(f"✅ Initial convergence reached in {grid.iterations} iterations.")
    
    print("\n--- Initial Results (30MW Load) ---")
    print(grid.display_buses())
    print(grid.display_lines())

    # --- Step 3: Modify Parameters & Re-solve ---
    print("\n--- Factory expansion! Doubling the load to 60MW... ---")
    # NEW: Self-aware handles make updates easy and readable!
    # Option 1: Update the specific load element directly
    grid.load(fac_id).set_p(60.0)
    grid.load(fac_id).set_q(20.0)
    
    # After modifying source components (Load, Gen, etc.), 
    # we MUST call init_pf() to rebuild the mathematical matrices.
    grid.init_pf()
    grid.solve()

    if grid.converged:
        print(f"✅ Re-solve convergence reached in {grid.iterations} iterations.")
        
    print("\n--- New Results (60MW Load) ---")
    print(grid.display_buses())
    print(grid.display_lines())
    print("Notice the voltage dropped significantly and the line is severely overloaded!")

if __name__ == "__main__":
    main()
