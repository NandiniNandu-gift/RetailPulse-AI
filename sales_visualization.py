import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "Data" / "Raw" / "sales_transactions.csv"
OUTPUT_DIR = BASE_DIR / "Reports" / "Sales_Charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load sales data
sales = pd.read_csv(DATA_FILE)
sales["transaction_date"] = pd.to_datetime(sales["transaction_date"])

# 1. Monthly Revenue Trend
monthly = sales.groupby(
    sales["transaction_date"].dt.to_period("M")
)["revenue"].sum()

plt.figure(figsize=(12, 6))
monthly.index = monthly.index.astype(str)
monthly.plot(marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (Rs.)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_revenue_trend.png", dpi=300)
plt.close()

# 2. Top 5 Products
products = sales.groupby("product_id")["revenue"].sum()
top_products = products.sort_values(ascending=False).head(5)

plt.figure(figsize=(9, 5))
top_products.sort_values().plot(kind="barh")
plt.title("Top 5 Products by Revenue")
plt.xlabel("Revenue (Rs.)")
plt.ylabel("Product ID")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_5_products.png", dpi=300)
plt.close()

# 3. Store-wise Revenue
stores = sales.groupby("store_id")["revenue"].sum().sort_values(ascending=False)

plt.figure(figsize=(9, 5))
stores.plot(kind="bar")
plt.title("Store-wise Revenue Performance")
plt.xlabel("Store")
plt.ylabel("Revenue (Rs.)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "store_revenue.png", dpi=300)
plt.close()

# 4. Payment Method Distribution
payments = sales["payment_method"].value_counts()

plt.figure(figsize=(8, 6))
payments.plot(kind="pie", autopct="%1.1f%%", startangle=90)
plt.title("Payment Method Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "payment_distribution.png", dpi=300)
plt.close()

print("\n===================================")
print(" SALES VISUALIZATION COMPLETED!")
print("===================================")
print("4 charts saved successfully in:")
print(OUTPUT_DIR)