
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "Data" / "Processed" / "inventory_analysis.csv"

OUTPUT_DIR = BASE_DIR / "Reports" / "Inventory_Charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. LOAD DATA
# =====================================

df = pd.read_csv(DATA_FILE)

print("\n====================================")
print(" RETAILPULSE AI - INVENTORY CHARTS")
print("====================================")


# =====================================
# 3. STOCK STATUS CHART
# =====================================

stock_status = df["stock_status"].value_counts()

plt.figure(figsize=(8, 6))
stock_status.plot(kind="bar")

plt.title("Inventory Stock Status")
plt.xlabel("Stock Status")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "stock_status.png", dpi=300)
plt.close()


# =====================================
# 4. STORE-WISE LOW STOCK CHART
# =====================================

store_low = df[df["stock_status"] == "Low Stock"].groupby(
    "store_id"
).size()

plt.figure(figsize=(9, 5))
store_low.plot(kind="bar")

plt.title("Store-wise Low Stock Items")
plt.xlabel("Store")
plt.ylabel("Number of Low Stock Items")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "store_low_stock.png", dpi=300)
plt.close()


# =====================================
# 5. CATEGORY-WISE INVENTORY CHART
# =====================================

category_stock = df.groupby("category")["stock_on_hand"].sum()

plt.figure(figsize=(10, 6))
category_stock.sort_values().plot(kind="barh")

plt.title("Category-wise Inventory Stock")
plt.xlabel("Total Stock Units")
plt.ylabel("Category")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "category_inventory.png", dpi=300)
plt.close()


# =====================================
# 6. INVENTORY COST CHART
# =====================================

category_cost = df.groupby("category")["inventory_cost"].sum()

plt.figure(figsize=(9, 6))
category_cost.plot(kind="bar")

plt.title("Inventory Cost by Category")
plt.xlabel("Category")
plt.ylabel("Inventory Cost (Rs.)")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "inventory_cost.png", dpi=300)
plt.close()


# =====================================
# 7. COMPLETION
# =====================================

print("\n4 INVENTORY CHARTS CREATED SUCCESSFULLY!")

print("\nCharts saved in:")
print(OUTPUT_DIR)

print("\n====================================")
print(" INVENTORY VISUALIZATION COMPLETED")
print("====================================")
