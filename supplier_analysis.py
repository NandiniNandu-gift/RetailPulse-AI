
import pandas as pd
from pathlib import Path

# ======================================
# 1. PROJECT PATHS
# ======================================

BASE_DIR = Path(__file__).resolve().parent.parent

SUPPLIER_FILE = BASE_DIR / "Data" / "Raw" / "suppliers.csv"
ORDERS_FILE = BASE_DIR / "Data" / "Raw" / "purchase_orders.csv"

OUTPUT_DIR = BASE_DIR / "Data" / "Processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================
# 2. LOAD DATA
# ======================================

suppliers = pd.read_csv(SUPPLIER_FILE)
orders = pd.read_csv(ORDERS_FILE)

# Merge supplier details with purchase orders
df = orders.merge(
    suppliers,
    on="supplier_id",
    how="left",
    validate="many_to_one"
)

# ======================================
# 3. DATA PREPARATION
# ======================================

df["order_date"] = pd.to_datetime(df["order_date"])

df["delivery_status"] = df["delivery_delay_days"].apply(
    lambda x: "On Time" if x <= 0 else "Delayed"
)

df["fulfilment_pct"] = (
    df["received_quantity"] / df["ordered_quantity"]
) * 100

df["short_quantity"] = (
    df["ordered_quantity"] - df["received_quantity"]
)


# ======================================
# 4. OVERALL SUPPLIER PERFORMANCE
# ======================================

print("\n======================================")
print(" RETAILPULSE AI - SUPPLIER INTELLIGENCE")
print("======================================")

print("\n1. OVERALL PURCHASE ORDER SUMMARY")

print("Total Purchase Orders:", len(df))
print("Total Suppliers:", df["supplier_id"].nunique())

print("Total Ordered Quantity:",
      df["ordered_quantity"].sum())

print("Total Received Quantity:",
      df["received_quantity"].sum())

print("Total Short Quantity:",
      df["short_quantity"].sum())

print("Average Delivery Delay:",
      round(df["delivery_delay_days"].mean(), 2), "days")


# ======================================
# 5. SUPPLIER-WISE PERFORMANCE
# ======================================

supplier_performance = df.groupby(
    ["supplier_id", "supplier_name", "region"]
).agg(
    total_orders=("po_id", "count"),
    total_ordered=("ordered_quantity", "sum"),
    total_received=("received_quantity", "sum"),
    average_delay=("delivery_delay_days", "mean"),
    on_time_orders=("delivery_status",
                    lambda x: (x == "On Time").sum()),
    delayed_orders=("delivery_status",
                    lambda x: (x == "Delayed").sum())
).reset_index()

supplier_performance["on_time_delivery_pct"] = (
    supplier_performance["on_time_orders"] /
    supplier_performance["total_orders"]
) * 100

supplier_performance["fulfilment_pct"] = (
    supplier_performance["total_received"] /
    supplier_performance["total_ordered"]
) * 100

supplier_performance["average_delay"] = (
    supplier_performance["average_delay"].round(2)
)

supplier_performance["on_time_delivery_pct"] = (
    supplier_performance["on_time_delivery_pct"].round(2)
)

supplier_performance["fulfilment_pct"] = (
    supplier_performance["fulfilment_pct"].round(2)
)

supplier_performance = supplier_performance.sort_values(
    "on_time_delivery_pct", ascending=False
)

print("\n2. SUPPLIER-WISE PERFORMANCE")

print(supplier_performance.to_string(index=False))


# ======================================
# 6. DELIVERY DELAY ANALYSIS
# ======================================

print("\n3. DELIVERY DELAY ANALYSIS")

print("On-time Orders:",
      (df["delivery_status"] == "On Time").sum())

print("Delayed Orders:",
      (df["delivery_status"] == "Delayed").sum())

print("Negative Delay Records:",
      (df["delivery_delay_days"] < 0).sum())


# ======================================
# 7. INCOMPLETE PURCHASE ORDERS
# ======================================

incomplete_orders = df[df["short_quantity"] > 0].copy()

print("\n4. INCOMPLETE PURCHASE ORDERS")

print("Incomplete Orders:", len(incomplete_orders))

print(
    incomplete_orders[
        [
            "po_id",
            "supplier_name",
            "product_id",
            "ordered_quantity",
            "received_quantity",
            "short_quantity"
        ]
    ].head(20).to_string(index=False)
)


# ======================================
# 8. SAVE REPORTS
# ======================================

df.to_csv(
    OUTPUT_DIR / "supplier_order_analysis.csv",
    index=False
)

supplier_performance.to_csv(
    OUTPUT_DIR / "supplier_performance_report.csv",
    index=False
)

incomplete_orders.to_csv(
    OUTPUT_DIR / "incomplete_purchase_orders.csv",
    index=False
)


# ======================================
# 9. COMPLETION
# ======================================

print("\n======================================")
print(" SUPPLIER ANALYSIS COMPLETED!")
print("======================================")

print("\n3 Reports saved successfully in:")
print(OUTPUT_DIR)
