import duckdb
import os
import zipfile
import io
import tempfile
import shutil
import rustpower

def inspect_columns():
    grid = rustpower.PowerGrid(case_path='cases/IEEE118/data.zip')
    grid.init_pf()
    grid.run_pf()
    grid.post_process()
    zip_bytes = grid.get_parquet_results()
    
    tmp = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            z.extractall(tmp)
        
        con = duckdb.connect()
        path = os.path.join(tmp, 'archetypes', '*.parquet')
        df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}', union_by_name=True)").df()
        print("Available columns in unified view:")
        print(df['column_name'].tolist())
    finally:
        shutil.rmtree(tmp)

if __name__ == "__main__":
    inspect_columns()
