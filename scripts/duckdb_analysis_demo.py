import rustpower
import pandas as pd
import duckdb
import zipfile
import io
import os
import tempfile
import shutil

"""
RustPower 0.5.0: The Ultimate DuckDB Integration Demo
---------------------------------------------------
This script demonstrates how to:
1. Run a high-performance power flow in Rust.
2. Export the entire ECS world state (Case + Results) as Parquet archives.
3. Use DuckDB to unify and analyze the fragmented ECS data using SQL.
"""

def run_hardcore_analysis():
    print("🚀 Initializing RustPower with IEEE 118 case...")
    # branch_analysis=True adds the incidence matrix plugin for extra detail
    grid = rustpower.PowerGrid(case_path='cases/IEEE118/data.zip', branch_analysis=True)
    
    print("⚡ Running Newton-Raphson Power Flow (KLU)...")
    grid.init_pf()
    grid.solve()
    grid.post_process()
    print(f"✅ Converged in {grid.iterations} iterations.\n")

    # 1. Capture the "Case" (Static params) and "Results" (Dynamic data)
    print("📦 Archiving ECS state to memory-only Parquet...")
    case_zip_bytes = grid.get_parquet_case()
    res_zip_bytes = grid.get_parquet_results()

    # 2. Setup DuckDB for analysis
    # We'll extract to a temporary virtual FS (temp folder)
    tmp_dir = tempfile.mkdtemp()
    try:
        case_path = os.path.join(tmp_dir, "case")
        res_path = os.path.join(tmp_dir, "res")
        
        with zipfile.ZipFile(io.BytesIO(case_zip_bytes)) as z:
            z.extractall(case_path)
        with zipfile.ZipFile(io.BytesIO(res_zip_bytes)) as z:
            z.extractall(res_path)

        # 3. Use DuckDB Magic to unify the fragments
        con = duckdb.connect()
        
        # 'union_by_name' is the key here: it merges different archetypes into a single table
        con.execute(f"CREATE VIEW case_world AS SELECT * FROM read_parquet('{case_path}/archetypes/*.parquet', union_by_name=True)")
        con.execute(f"CREATE VIEW res_world AS SELECT * FROM read_parquet('{res_path}/archetypes/*.parquet', union_by_name=True)")

        print("📊 [SQL] Identifying top 5 most loaded transmission lines...")
        # A single SQL query joins topology, parameters, and results seamlessly
        top_loaded = con.execute("""
            SELECT 
                c."FromBus.item" as from_bus,
                c."ToBus.item" as to_bus,
                c."LineParams.r_ohm_per_km" as resistance,
                r."LineResultData.p_from_mw" as p_mw,
                r."LineResultData.loading_percent" as loading_pct
            FROM case_world c
            JOIN res_world r ON c.id = r.id
            WHERE c."LineParams.r_ohm_per_km" IS NOT NULL
            ORDER BY loading_pct DESC
            LIMIT 5
        """).df()
        print(top_loaded)

        print("\n📊 [SQL] Aggregating average voltage per Zone...")
        # Notice how we join Bus results with their original Zone ID
        zone_analysis = con.execute("""
            SELECT 
                c."Zone.item" as zone,
                AVG(r."VBusResult.0") as avg_vm_pu,
                MIN(r."VBusResult.0") as min_vm_pu,
                MAX(r."VBusResult.0") as max_vm_pu
            FROM case_world c
            JOIN res_world r ON c.id = r.id
            WHERE c."BusID.item" IS NOT NULL
            GROUP BY zone
            ORDER BY avg_vm_pu ASC
        """).df()
        print(zone_analysis)

    finally:
        shutil.rmtree(tmp_dir)
        print("\n🧹 Cleanup complete. Memory-only analysis finished.")

if __name__ == "__main__":
    if 'archive' not in rustpower.features():
        print("❌ Error: Current build of rustpower does not have the 'archive' feature enabled.")
    else:
        run_hardcore_analysis()
