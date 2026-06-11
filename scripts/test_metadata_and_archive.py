import rustpower
import pandas as pd
import os

def test_features():
    print("--- RustPower Metadata ---")
    print(f"Version: {rustpower.version()}")
    print(f"Enabled Features: {rustpower.features()}")
    
    if 'archive' in rustpower.features():
        print("\n--- Testing Parquet Archival ---")
        grid = rustpower.PowerGrid(case_path='cases/IEEE118/data.zip')
        grid.init_pf()
        grid.solve()
        grid.post_process()
        
        # Save results to parquet
        archive_path = "ieee118_results_from_py.zip"
        grid.save_results(archive_path)
        
        if os.path.exists(archive_path):
            print(f"Successfully saved results to {archive_path}")
            size = os.path.getsize(archive_path)
            print(f"Archive size: {size} bytes")
        else:
            print("Failed to save results.")
    else:
        print("\nArchive feature not enabled, skipping archival test.")

if __name__ == "__main__":
    test_features()
