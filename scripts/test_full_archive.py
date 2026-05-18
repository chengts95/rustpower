import rustpower
import pandas as pd
import zipfile
import io
import os
import tempfile
import shutil
import duckdb

def test_full_archive_integration():
    print("--- RustPower 0.5.0: Full Case + Results Integration via DuckDB ---")
    grid = rustpower.PowerGrid(case_path='cases/IEEE118/data.zip')
    grid.init_pf()
    grid.run_pf()
    grid.post_process()
    
    # 1. Get both archives
    case_zip = grid.get_parquet_case()
    res_zip = grid.get_parquet_results()
    
    tmp_dir = tempfile.mkdtemp()
    try:
        # Extract case data to 'case/' and results to 'res/'
        case_dir = os.path.join(tmp_dir, 'case')
        res_dir = os.path.join(tmp_dir, 'res')
        
        with zipfile.ZipFile(io.BytesIO(case_zip)) as z:
            z.extractall(case_dir)
        with zipfile.ZipFile(io.BytesIO(res_zip)) as z:
            z.extractall(res_dir)
            
        con = duckdb.connect()
        
        # Create views for both
        con.execute(f"CREATE VIEW case_world AS SELECT * FROM read_parquet('{case_dir}/archetypes/*.parquet', union_by_name=True)")
        con.execute(f"CREATE VIEW res_world AS SELECT * FROM read_parquet('{res_dir}/archetypes/*.parquet', union_by_name=True)")
        
        print("\n[DuckDB] Joining Case Parameters with Simulation Results...")
        # Join using the internal ECS 'id'
        joined_query = con.execute("""
            SELECT 
                c.id,
                c."FromBus.item" as from_bus,
                c."ToBus.item" as to_bus,
                c."LineParams.r_ohm_per_km" as r,
                r."LineResultData.loading_percent" as loading
            FROM case_world c
            JOIN res_world r ON c.id = r.id
            WHERE c."LineParams.r_ohm_per_km" IS NOT NULL
            ORDER BY loading DESC
            LIMIT 10
        """).df()
        
        print("\nJoined Line Data (First 10 rows):")
        print(joined_query)

    finally:
        shutil.rmtree(tmp_dir)
        print("\nCleanup complete.")

if __name__ == "__main__":
    test_full_archive_integration()
