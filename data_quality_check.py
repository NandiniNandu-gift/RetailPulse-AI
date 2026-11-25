
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "Data" / "Raw"
OUTPUT_DIR = BASE_DIR / "Data" / "Processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
files = {
    "Sales": "sales_transactions.csv",
    "Products": "products.csv",
    "Stores": "stores.csv",
    "Suppliers": "suppliers.csv",
    "Purchase Orders": "purchase_orders.csv",
    "Inventory": "inventory_snapshot.csv"
}

datasets = {}

print("\n===================================")
print(" RETAILPULSE AI - DATA QUALITY AUDIT")
print("===================================")

for name, filename in files.items():
    path = RAW_DIR / filename

    if not path.exists():
        print(f"\nERROR: File not found: {filename}")
        raise FileNotFoundError(path)

    datasets[name] = pd.read_csv(path)

# Audit summary
report = []

for name, df in datasets.items():

    print(f"\n--- {name} DATASET ---")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    print("Missing values:", missing)
    print("Duplicate rows:", duplicates)

    report.append({
        "dataset": name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": missing,
        "duplicate_rows": duplicates
    })

# Business rule checks
sales = datasets["Sales"]
products = datasets["Products"]
inventory = datasets["Inventory"]
purchase_orders = datasets["Purchase Orders"]

print("\n===================================")
print(" BUSINESS RULE VALIDATION")
print("===================================")

checks = {
    "Negative sales quantity": int((sales["quantity"] < 0).sum()),
    "Zero sales quantity": int((sales["quantity"] == 0).sum()),
    "Negative revenue": int((sales["revenue"] < 0).sum()),
    "Negative profit": int((sales["profit"] < 0).sum()),
    "Invalid product price": int((products["unit_price"] <= 0).sum()),
    "Invalid product cost": int((products["unit_cost"] <= 0).sum()),
    "Negative inventory": int((inventory["stock_on_hand"] < 0).sum()),
    "Invalid received quantity": int(
        (purchase_orders["received_quantity"] >
         purchase_orders["ordered_quantity"]).sum()
    )
}

for check, count in checks.items():
    print(f"{check}: {count}")

# Check invalid product/store references
product_ids = set(products["product_id"])
store_ids = set(datasets["Stores"]["store_id"])

invalid_sales_products = int(
    (~sales["product_id"].isin(product_ids)).sum()
)

invalid_sales_stores = int(
    (~sales["store_id"].isin(store_ids)).sum()
)

print("\nInvalid product references in sales:",
      invalid_sales_products)

print("Invalid store references in sales:",
      invalid_sales_stores)

# Save quality report
report_df = pd.DataFrame(report)
report_df.to_csv(
    OUTPUT_DIR / "data_quality_report.csv",
    index=False
)

# Save business validation report
checks_df = pd.DataFrame(
    list(checks.items()),
    columns=["validation_check", "issue_count"]
)

checks_df.to_csv(
    OUTPUT_DIR / "business_validation_report.csv",
    index=False
)

print("\n===================================")
print(" DATA QUALITY AUDIT COMPLETED!")
print("===================================")

print("\nReports saved in:")
print(OUTPUT_DIR)
