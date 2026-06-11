import rustpower
import pandas as pd
import io
import zipfile

def test_parquet_results():
    print("--- Testing 'Tuple Magic' Parquet Results ---")
    grid = rustpower.PowerGrid(case_path='cases/IEEE118/data.zip')
    grid.init_pf()
    grid.solve()
    grid.post_process()
    
    zip_bytes = grid.get_parquet_results()
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        archetype_files = [f for f in z.namelist() if f.startswith('archetypes/') and f.endswith('.parquet')]
        print(f"Found {len(archetype_files)} archetypes in the archive.")
        
        for file_path in archetype_files:
            with z.open(file_path) as f:
                try:
                    df = pd.read_parquet(f)
                    print(f"\nArchetype: {file_path} with columns: {df.columns.tolist()}")
                    print(f"  Rows: {len(df)}")
                    print("  Preview:")
                    print(df.head(2))
                except Exception as e:
                    print(f"  Error loading parquet: {e}")

if __name__ == "__main__":
    test_parquet_results()
