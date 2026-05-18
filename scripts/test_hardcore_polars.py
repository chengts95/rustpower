import rustpower
import pandas as pd
import io
import zipfile
import os

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

def test_full_workflow():
    print("--- RustPower 0.5.0: The Ultimate Workflow ---")
    grid = rustpower.PowerGrid(case_path='cases/IEEE118/data.zip', branch_analysis=True)
    grid.init_pf()
    grid.run_pf()
    grid.post_process()
    
    # 1. The "Fast Path" (Standard Numpy/Pandas)
    print("\n[Standard Path] Bus results via Numpy/Pandas:")
    bus_df = pd.DataFrame(grid.get_bus_results())
    print(bus_df[['bus_id', 'vm_pu', 'p_mw']].head())
    
    # 2. The "Hardcore Path" (Parquet/Virtual FS)
    print("\n[Hardcore Path] Archiving to memory-only Parquet...")
    zip_bytes = grid.get_parquet_results()
    
    if HAS_POLARS:
        print("Using Polars to explore the archive (Rust power on both sides!):")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            # Find the main result archetype
            for name in z.namelist():
                if name.startswith('archetypes/') and name.endswith('.parquet'):
                    with z.open(name) as f:
                        # Polars can read the file-like object directly
                        df = pl.read_parquet(f)
                        if 'VBusResult.0' in df.columns:
                            print(f"\nFound results in {name}:")
                            print(df.head(5))
                            break
    else:
        print("Polars not installed. Standard path works perfectly, but you're missing the 'Hardcore' mode!")
        print("Tip: pip install polars")

if __name__ == "__main__":
    test_full_workflow()
