import pandas as pd
import numpy as np
from pathlib import Path

# Project folders
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "Data"

for folder in ["Raw", "Cleaned", "Processed"]:
    (DATA / folder).mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# 1. Products dataset
products = pd.DataFrame({
    "product_id": [f"P{i:03}" for i in range(1, 21)],
    "product_name": [f"Product_{i}" for i in range(1, 21)],
    "category": np.random.choice(
        ["Grocery", "Beverages", "Snacks", "Personal Care", "Household"], 20
    ),
    "unit_cost": np.random.randint(20, 200, 20),
    "unit_price": np.random.randint(50, 300, 20),
    "reorder_point": np.random.randint(20, 100, 20),
    "lead_time_days": np.random.randint(2, 12, 20)
})

# 2. Stores dataset
stores = pd.DataFrame({
    "store_id": ["S001", "S002", "S003", "S004", "S005"],
    "store_name": ["Bengaluru", "Mysuru", "Chennai", "Hyderabad", "Pune"],
    "region": ["South", "South", "South", "South", "West"]
})

# 3. Suppliers dataset
suppliers = pd.DataFrame({
    "supplier_id": [f"SUP{i:03}" for i in range(1, 6)],
    "supplier_name": [
        "FreshRoute", "Metro Wholesale", "PrimeGoods",
        "SwiftCart", "Apex Supply"
    ],
    "region": ["South", "West", "North", "South", "East"]
})

# 4. Sales transactions dataset
n = 10000
sales = pd.DataFrame({
    "transaction_id": [f"TX{i:06}" for i in range(1, n + 1)],
    "transaction_date": pd.to_datetime(
        np.random.choice(
            pd.date_range("2024-01-01", "2025-12-31"), n
        )
    ),
    "store_id": np.random.choice(stores["store_id"], n),
    "product_id": np.random.choice(products["product_id"], n),
    "quantity": np.random.randint(1, 10, n),
    "payment_method": np.random.choice(
        ["UPI", "Cash", "Card", "Wallet"], n
    )
})

sales = sales.merge(products, on="product_id")
sales["discount_pct"] = np.random.choice(
    [0, 0.05, 0.10, 0.15], n
)
sales["unit_price"] = sales["unit_price"].astype(float)
sales["revenue"] = (
    sales["quantity"] * sales["unit_price"] *
    (1 - sales["discount_pct"])
)
sales["profit"] = (
    sales["revenue"] -
    sales["quantity"] * sales["unit_cost"]
)

sales = sales[[
    "transaction_id", "transaction_date", "store_id",
    "product_id", "quantity", "unit_price",
    "discount_pct", "revenue", "profit", "payment_method"
]]

# 5. Purchase orders dataset
purchase_orders = pd.DataFrame({
    "po_id": [f"PO{i:05}" for i in range(1, 501)],
    "order_date": pd.to_datetime(
        np.random.choice(
            pd.date_range("2024-01-01", "2025-12-01"), 500
        )
    ),
    "store_id": np.random.choice(stores["store_id"], 500),
    "product_id": np.random.choice(products["product_id"], 500),
    "supplier_id": np.random.choice(suppliers["supplier_id"], 500),
    "ordered_quantity": np.random.randint(30, 300, 500),
    "received_quantity": np.random.randint(20, 300, 500),
    "delivery_delay_days": np.random.randint(0, 10, 500)
})

purchase_orders["received_quantity"] = np.minimum(
    purchase_orders["received_quantity"],
    purchase_orders["ordered_quantity"]
)

# 6. Inventory snapshot dataset
inventory = pd.DataFrame([
    {
        "snapshot_date": "2025-12-31",
        "store_id": store,
        "product_id": product,
        "stock_on_hand": np.random.randint(0, 200),
        "reorder_point": int(
            products.loc[
                products["product_id"] == product,
                "reorder_point"
            ].iloc[0]
        )
    }
    for store in stores["store_id"]
    for product in products["product_id"]
])

# Save all datasets
datasets = {
    "sales_transactions.csv": sales,
    "products.csv": products,
    "stores.csv": stores,
    "suppliers.csv": suppliers,
    "purchase_orders.csv": purchase_orders,
    "inventory_snapshot.csv": inventory
}

for filename, df in datasets.items():
    df.to_csv(DATA / "Raw" / filename, index=False)

print("\nRETAILPULSE AI DATASETS CREATED SUCCESSFULLY!")
for filename, df in datasets.items():
    print(filename, ":", len(df), "rows")

print("\nDataset location:", DATA / "Raw")
