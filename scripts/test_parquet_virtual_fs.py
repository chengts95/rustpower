import rustpower
import pandas as pd
import zipfile
import io
import os

def test_parquet_virtual_fs():
    print("--- Testing Python-side Virtual FS for Parquet ---")
    grid = rustpower.PowerGrid(case_path='cases/IEEE118/data.zip')
    grid.init_pf()
    grid.run_pf()
    grid.post_process()
    
    # Get the raw ZIP bytes
    zip_bytes = grid.get_parquet_results()
    
    # Use Python's zipfile to treat it as a virtual memory file system
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        print(f"Archive contains {len(z.namelist())} files.")
        
        # Find all parquet files in archetypes/
        archetype_files = [f for f in z.namelist() if f.startswith('archetypes/') and f.endswith('.parquet')]
        
        for file_path in archetype_files:
            print(f"\nProcessing: {file_path}")
            # Read directly from ZIP into Pandas without touching the disk
            with z.open(file_path) as f:
                try:
                    df = pd.read_parquet(f)
                    print(f"  Rows: {len(df)}")
                    print(f"  Columns: {df.columns.tolist()[:3]}...") # Show first 3 columns
                except Exception as e:
                    print(f"  Error loading parquet: {e}")

if __name__ == "__main__":
    test_parquet_virtual_fs()
