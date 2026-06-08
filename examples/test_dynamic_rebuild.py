import rustpower as rp
import pandas as pd
import numpy as np

def main():
    print("--- Dynamic Rebuild Verification (Simple Grid) ---")
    
    # 1. Build a simple 2-bus system
    grid = rp.PowerGrid()
    with grid.defer() as b:
        b.add_bus(vn_kv=110.0, name="Source")
        b.add_bus(vn_kv=110.0, name="LoadBus")
    
    # Add Slack and Load
    grid.add_ext_grid(bus=0, vm_pu=1.02)
    load_h = grid.add_load(bus=1, p_mw=30.0, q_mvar=10.0)
    grid.add_line(from_bus=0, to_bus=1, length_km=10.0)
    
    # INITIALIZATION: Build the matrices and lookup tables once!
    print("Initializing power flow matrices...")
    grid.init_pf()
    
    # Initial solve
    grid.solve()
    v_initial = grid.v.copy()
    print(f"Initial State Converged. V at LoadBus: {np.abs(v_initial[1]):.4f} p.u.")

    # 2. Modify load by 10%
    print("\nModifying load: 30MW -> 33MW (10% increase)...")
    load_h.set_p(33.0)
    
    # 3. Solve (Reactive path)
    grid.solve()
    v_new = grid.v
    print(f"New State Converged. V at LoadBus: {np.abs(v_new[1]):.4f} p.u.")
    
    diff = np.linalg.norm(v_initial - v_new)
    print(f"L2-norm of voltage change: {diff:.8e}")
    
    if diff > 1e-6:
        print("✅ SUCCESS: Power flow results reacted correctly to dynamic modification.")
    else:
        print("❌ FAILURE: Results didn't change! Check synchronization logic.")

    # 4. Idempotency Check
    print("\nRunning Idempotency Check...")
    n_bus = grid.n_bus
    grid.init_pf()
    grid.solve()
    if grid.n_bus == n_bus:
        print("✅ SUCCESS: No duplication after init_pf().")

if __name__ == "__main__":
    main()
