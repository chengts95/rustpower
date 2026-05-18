import rustpower
import numpy as np
import pandas as pd

def test():
    print("Testing without plugin...")
    grid = rustpower.PowerGrid(case_path='cases/IEEE118/data.zip')
    grid.init_pf()
    grid.run_pf()
    try:
        inc = grid.get_incidence()
    except Exception as e:
        print(f"Caught expected error: {e}")

    print("\nTesting full result extraction with Pandas...")
    grid = rustpower.PowerGrid(case_path='cases/IEEE118/data.zip', branch_analysis=True)
    grid.init_pf()
    grid.run_pf()
    grid.post_process()
    
    bus_res = grid.get_bus_results()
    bus_df = pd.DataFrame(bus_res)
    print("\nBus Results (First 5 rows):")
    print(bus_df.head())
    
    line_res = grid.get_line_results()
    line_df = pd.DataFrame(line_res)
    print("\nLine Results (First 5 rows):")
    print(line_df.head())
    
    if len(line_df) > 0:
        print(f"\nSuccessfully extracted {len(line_df)} line results!")
    else:
        print("\nError: Line results are still empty.")

if __name__ == "__main__":
    test()
