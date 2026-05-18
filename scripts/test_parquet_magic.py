import rustpower
import pandas as pd
import io
import os

def test_parquet_results():
    print("--- Testing 'Tuple Magic' Parquet Results ---")
    grid = rustpower.PowerGrid(case_path='cases/IEEE118/data.zip')
    grid.init_pf()
    grid.run_pf()
    grid.post_process()
    
    results = grid.get_parquet_results()
    print(f"Found {len(results)} archetypes in the archive.")
    
    for columns, data in results:
        cols_tuple = tuple(columns)
        print(f"\nArchetype with columns: {cols_tuple}")
        try:
            # Load into pandas without intermediate file
            df = pd.read_parquet(io.BytesIO(data))
            print(f"  Rows: {len(df)}")
            print("  Preview:")
            print(df.head(2))
        except Exception as e:
            print(f"  Error loading parquet: {e}")

if __name__ == "__main__":
    test_parquet_results()
