import rustpower
import pandas as pd
import zipfile
import io
import os
import tempfile
import shutil

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

def test_duckdb_integration():
    print("--- RustPower 0.5.0: DuckDB Integration (Zero Annoyance Mode) ---")
    grid = rustpower.PowerGrid(case_path='cases/IEEE118/data.zip', branch_analysis=True)
    grid.init_pf()
    grid.run_pf()
    grid.post_process()
    
    # 1. Get the raw archive bytes
    zip_bytes = grid.get_parquet_results()
    
    if not HAS_DUCKDB:
        print("\nDuckDB not installed. Please 'pip install duckdb' to see the magic.")
        return

    # 2. Extract to a temporary directory (DuckDB loves directories)
    tmp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            z.extractall(tmp_dir)
        
        # 3. The Magic SQL: union_by_name=True
        # This makes the "arch_X" names completely IRRELEVANT.
        # It joins everything into a single unified view based on column names.
        con = duckdb.connect()
        
        # We point DuckDB to the glob pattern of all archetypes
        parquet_path = os.path.join(tmp_dir, 'archetypes', '*.parquet')
        
        print(f"\n[DuckDB] Unifying {len(os.listdir(os.path.join(tmp_dir, 'archetypes')))} archetypes...")
        
        # Create a unified view
        con.execute(f"""
            CREATE VIEW unified_world AS 
            SELECT * FROM read_parquet('{parquet_path}', union_by_name=True)
        """)
        
        # 4. Now the user just queries what they want! 
        # They don't care which arch file it came from.
        print("\n[Query] Getting consolidated Bus Results (Vm, P, Q):")
        bus_query = con.execute("""
            SELECT 
                "BusID.item" as id, 
                "VBusResult.0" as vm_pu, 
                "SBusResult.0" as p_mw, 
                "SBusResult.1" as q_mvar 
            FROM unified_world 
            WHERE "BusID.item" IS NOT NULL
            ORDER BY id ASC
            LIMIT 5
        """).df()
        print(bus_query)

        print("\n[Query] Getting consolidate Line Loading:")
        line_query = con.execute("""
            SELECT 
                "LineResultData.p_from_mw" as p_mw, 
                "LineResultData.q_from_mvar" as q_mvar, 
                "LineResultData.loading_percent" as loading
            FROM unified_world 
            WHERE "LineResultData.loading_percent" IS NOT NULL
            LIMIT 5
        """).df()
        print(line_query)

    finally:
        shutil.rmtree(tmp_dir)
        print("\nCleanup complete.")

if __name__ == "__main__":
    test_duckdb_integration()
