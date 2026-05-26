"""Dump poly_cost.csv for test cases and add to IEEE118 zip."""
import pandapower.networks as pn
import zipfile
import os

# ── case39 ──────────────────────────────────────────────────────────────────
net39 = pn.case39()
print("case39 poly_cost columns:", list(net39.poly_cost.columns))
print(net39.poly_cost.to_string())
csv39 = net39.poly_cost.to_csv(index=False)
os.makedirs("cases/IEEE39", exist_ok=True)
with open("cases/IEEE39/poly_cost.csv", "w") as f:
    f.write(csv39)
print("\nWrote cases/IEEE39/poly_cost.csv")

# ── case118 ─────────────────────────────────────────────────────────────────
net118 = pn.case118()
print("\ncase118 poly_cost shape:", net118.poly_cost.shape)
print("first 5 rows:")
print(net118.poly_cost.head().to_string())
csv118 = net118.poly_cost.to_csv(index=False)

# Write standalone CSV
with open("cases/IEEE118/poly_cost.csv", "w") as f:
    f.write(csv118)
print("\nWrote cases/IEEE118/poly_cost.csv")

# Add to zip (append mode, replace if exists)
zip_path = "cases/IEEE118/data.zip"
# Read existing zip, rebuild with poly_cost.csv added/replaced
import io
with zipfile.ZipFile(zip_path, 'r') as zin:
    existing = {name: zin.read(name) for name in zin.namelist()}

existing['poly_cost.csv'] = csv118.encode('utf-8')

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
    for name, data in existing.items():
        zout.writestr(name, data)
print("Updated cases/IEEE118/data.zip with poly_cost.csv")
print("Files in zip:", list(existing.keys()))
