import pandas as pd
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
SALES_FILE = BASE_DIR / "Data" / "Raw" / "sales_transactions.csv"
OUTPUT_DIR = BASE_DIR / "Data" / "Processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
sales = pd.read_csv(SALES_FILE)

# Convert transaction date
sales["transaction_date"] = pd.to_datetime(sales["transaction_date"])
sales["month"] = sales["transaction_date"].dt.to_period("M").astype(str)

print("\n======================================")
print(" RETAILPULSE AI - SALES ANALYSIS")
print("======================================")

# 1. Overall business performance
print("\n1. OVERALL SALES PERFORMANCE")
print("Total Transactions:", len(sales))
print("Total Quantity Sold:", sales["quantity"].sum())
print("Total Revenue: Rs.", round(sales["revenue"].sum(), 2))
print("Total Profit: Rs.", round(sales["profit"].sum(), 2))
print("Average Transaction Revenue: Rs.", round(sales["revenue"].mean(), 2))
print("Loss-making Transactions:", (sales["profit"] < 0).sum())

# 2. Monthly sales analysis
monthly_sales = sales.groupby("month").agg(
    total_transactions=("transaction_id", "count"),
    total_quantity=("quantity", "sum"),
    total_revenue=("revenue", "sum"),
    total_profit=("profit", "sum")
).reset_index()

monthly_sales = monthly_sales.sort_values("month")

print("\n2. MONTHLY SALES PERFORMANCE")
print(monthly_sales.to_string(index=False))

# 3. Product performance
product_performance = sales.groupby("product_id").agg(
    total_quantity=("quantity", "sum"),
    total_revenue=("revenue", "sum"),
    total_profit=("profit", "sum")
).reset_index()

product_performance = product_performance.sort_values(
    "total_revenue", ascending=False
)

print("\n3. TOP 5 PRODUCTS BY REVENUE")
print(product_performance.head(5).to_string(index=False))

# 4. Store performance
store_performance = sales.groupby("store_id").agg(
    total_transactions=("transaction_id", "count"),
    total_revenue=("revenue", "sum"),
    total_profit=("profit", "sum")
).reset_index()

store_performance = store_performance.sort_values(
    "total_revenue", ascending=False
)

print("\n4. STORE PERFORMANCE")
print(store_performance.to_string(index=False))

# 5. Payment method analysis
payment_analysis = sales.groupby("payment_method").agg(
    total_transactions=("transaction_id", "count"),
    total_revenue=("revenue", "sum")
).reset_index()

payment_analysis = payment_analysis.sort_values(
    "total_revenue", ascending=False
)

print("\n5. PAYMENT METHOD ANALYSIS")
print(payment_analysis.to_string(index=False))

# Save analysis reports
monthly_sales.to_csv(OUTPUT_DIR / "monthly_sales_analysis.csv", index=False)
product_performance.to_csv(OUTPUT_DIR / "product_performance.csv", index=False)
store_performance.to_csv(OUTPUT_DIR / "store_performance.csv", index=False)
payment_analysis.to_csv(OUTPUT_DIR / "payment_analysis.csv", index=False)

print("\n======================================")
print(" SALES ANALYSIS COMPLETED SUCCESSFULLY")
print("======================================")
print("\nReports saved in:", OUTPUT_DIR)