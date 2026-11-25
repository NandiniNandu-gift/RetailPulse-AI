
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# RETAILPULSE AI - SUPPLIER VISUALIZATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Data" / "Processed"
REPORT_DIR = BASE_DIR / "Reports" / "Supplier_Charts"

INPUT_FILE = DATA_DIR / "supplier_order_analysis.csv"
PERFORMANCE_FILE = DATA_DIR / "supplier_performance_report.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load supplier analysis reports."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required file not found: {INPUT_FILE}\n"
            "Run supplier_analysis.py first."
        )

    if not PERFORMANCE_FILE.exists():
        raise FileNotFoundError(
            f"Required file not found: {PERFORMANCE_FILE}\n"
            "Run supplier_analysis.py first."
        )

    orders = pd.read_csv(INPUT_FILE)
    suppliers = pd.read_csv(PERFORMANCE_FILE)

    required_order_columns = [
        "supplier_name",
        "delivery_status",
        "delivery_delay_days",
        "short_quantity",
        "ordered_quantity",
        "received_quantity"
    ]

    required_supplier_columns = [
        "supplier_name",
        "on_time_delivery_pct",
        "fulfilment_pct",
        "average_delay",
        "total_orders"
    ]

    for column in required_order_columns:
        if column not in orders.columns:
            raise ValueError(f"Missing column in order report: {column}")

    for column in required_supplier_columns:
        if column not in suppliers.columns:
            raise ValueError(
                f"Missing column in supplier report: {column}"
            )

    return orders, suppliers


def save_chart(filename):
    """Save chart and close the figure."""

    plt.tight_layout()
    plt.savefig(REPORT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def create_supplier_charts(orders, suppliers):

    # 1. Supplier-wise Purchase Orders
    plt.figure(figsize=(10, 6))

    plt.bar(
        suppliers["supplier_name"],
        suppliers["total_orders"]
    )

    plt.title("Supplier-wise Purchase Orders")
    plt.xlabel("Supplier")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=20)

    save_chart("supplier_orders.png")

    # 2. Supplier-wise Delivery Performance
    plt.figure(figsize=(10, 6))

    plt.bar(
        suppliers["supplier_name"],
        suppliers["on_time_delivery_pct"]
    )

    plt.title("Supplier-wise On-time Delivery Percentage")
    plt.xlabel("Supplier")
    plt.ylabel("On-time Delivery (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=20)

    save_chart("supplier_delivery_performance.png")

    # 3. Supplier-wise Fulfilment Percentage
    plt.figure(figsize=(10, 6))

    plt.bar(
        suppliers["supplier_name"],
        suppliers["fulfilment_pct"]
    )

    plt.title("Supplier-wise Order Fulfilment")
    plt.xlabel("Supplier")
    plt.ylabel("Fulfilment (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=20)

    save_chart("supplier_fulfilment.png")

    # 4. Supplier-wise Average Delivery Delay
    plt.figure(figsize=(10, 6))

    plt.bar(
        suppliers["supplier_name"],
        suppliers["average_delay"]
    )

    plt.title("Supplier-wise Average Delivery Delay")
    plt.xlabel("Supplier")
    plt.ylabel("Average Delay (Days)")
    plt.xticks(rotation=20)

    save_chart("supplier_delivery_delay.png")

    # 5. On-time vs Delayed Orders
    delivery_counts = (
        orders["delivery_status"]
        .value_counts()
        .reindex(["On Time", "Delayed"], fill_value=0)
    )

    plt.figure(figsize=(8, 6))

    plt.pie(
        delivery_counts.values,
        labels=delivery_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("On-time vs Delayed Purchase Orders")

    save_chart("delivery_status_distribution.png")

    # 6. Supplier-wise Short Quantity
    supplier_shortage = (
        orders.groupby("supplier_name")["short_quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        supplier_shortage.index,
        supplier_shortage.values
    )

    plt.title("Supplier-wise Short Quantity")
    plt.xlabel("Supplier")
    plt.ylabel("Short Quantity (Units)")
    plt.xticks(rotation=20)

    save_chart("supplier_short_quantity.png")


def main():

    print("=" * 55)
    print("RETAILPULSE AI - SUPPLIER VISUALIZATION")
    print("=" * 55)

    orders, suppliers = load_data()

    print("\nData loaded successfully.")
    print(f"Purchase Orders: {len(orders)}")
    print(f"Suppliers: {len(suppliers)}")

    create_supplier_charts(orders, suppliers)

    print("\n6 supplier charts created successfully!")
    print(f"\nCharts saved in:\n{REPORT_DIR}")

    print("\nGenerated files:")

    for file in sorted(REPORT_DIR.glob("*.png")):
        print(f"- {file.name}")

    print("\n" + "=" * 55)
    print("SUPPLIER VISUALIZATION COMPLETED!")
    print("=" * 55)


if __name__ == "__main__":
    main()
