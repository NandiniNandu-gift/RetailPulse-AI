
import pandas as pd
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INVENTORY_FILE = BASE_DIR / "Data" / "Raw" / "inventory_snapshot.csv"
PRODUCTS_FILE = BASE_DIR / "Data" / "Raw" / "products.csv"

OUTPUT_DIR = BASE_DIR / "Data" / "Processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATASETS
# ==========================================

inventory = pd.read_csv(INVENTORY_FILE)
products = pd.read_csv(PRODUCTS_FILE)

# Remove product reorder_point before merging.
# Inventory reorder_point will be used for stock analysis.
products = products.drop(columns=["reorder_point"])

df = inventory.merge(
    products,
    on="product_id",
    how="left",
    validate="many_to_one"
)


# ==========================================
# 3. INVENTORY STATUS
# ==========================================

df["stock_status"] = "Healthy Stock"

df.loc[
    df["stock_on_hand"] <= df["reorder_point"],
    "stock_status"
] = "Low Stock"


# ==========================================
# 4. INVENTORY ANALYSIS
# ==========================================

print("\n======================================")
print(" RETAILPULSE AI - INVENTORY INTELLIGENCE")
print("======================================")

# Overall inventory
print("\n1. OVERALL INVENTORY")

print("Total Inventory Records:", len(df))

print("Total Units in Stock:",
      df["stock_on_hand"].sum())

total_inventory_cost = (
    df["stock_on_hand"] * df["unit_cost"]
).sum()

print("Total Inventory Cost: Rs.",
      round(total_inventory_cost, 2))


# Stock status
print("\n2. STOCK STATUS")

print(df["stock_status"].value_counts().to_string())


# Low stock products
low_stock = df[df["stock_status"] == "Low Stock"].copy()

low_stock["reorder_quantity"] = (
    low_stock["reorder_point"] -
    low_stock["stock_on_hand"]
)

print("\n3. LOW STOCK PRODUCTS")

if len(low_stock) > 0:
    print(
        low_stock[
            [
                "store_id",
                "product_id",
                "product_name",
                "stock_on_hand",
                "reorder_point",
                "reorder_quantity"
            ]
        ].to_string(index=False)
    )
else:
    print("No low-stock products found.")


# Category-wise inventory
df["inventory_cost"] = (
    df["stock_on_hand"] * df["unit_cost"]
)

category_analysis = df.groupby("category").agg(
    total_stock=("stock_on_hand", "sum"),
    average_stock=("stock_on_hand", "mean"),
    total_inventory_cost=("inventory_cost", "sum")
).reset_index()

print("\n4. CATEGORY-WISE INVENTORY")

print(category_analysis.to_string(index=False))


# Store-wise inventory
store_analysis = df.groupby("store_id").agg(
    total_stock=("stock_on_hand", "sum"),
    low_stock_items=(
        "stock_status",
        lambda x: (x == "Low Stock").sum()
    )
).reset_index()

print("\n5. STORE-WISE INVENTORY")

print(store_analysis.to_string(index=False))


# ==========================================
# 5. SAVE REPORTS
# ==========================================

df.to_csv(
    OUTPUT_DIR / "inventory_analysis.csv",
    index=False
)

low_stock.to_csv(
    OUTPUT_DIR / "low_stock_report.csv",
    index=False
)

category_analysis.to_csv(
    OUTPUT_DIR / "category_inventory_analysis.csv",
    index=False
)

store_analysis.to_csv(
    OUTPUT_DIR / "store_inventory_analysis.csv",
    index=False
)


# ==========================================
# 6. COMPLETION
# ==========================================

print("\n======================================")
print(" INVENTORY ANALYSIS COMPLETED!")
print("======================================")

print("\n4 Reports saved successfully in:")
print(OUTPUT_DIR)
