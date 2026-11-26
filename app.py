
# RETAILPULSE AI - RETAIL ANALYTICS DASHBOARD

from pathlib import Path
import re
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# PAGE CONFIGURATION
st.set_page_config(
    page_title="RetailPulse AI | Retail Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# DASHBOARD DESIGN
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg, #0b1220 0%, #101a2d 55%, #0c1525 100%
    );
    color: #edf3ff;
}

[data-testid="stSidebar"] {
    background: #0a1120;
    border-right: 1px solid #26344d;
}

[data-testid="stSidebar"] * {
    color: #e6edfa !important;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f5f8ff !important;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg, rgba(27,43,68,.96), rgba(17,29,49,.96)
    );
    border: 1px solid #2c4161;
    padding: 16px 18px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,.18);
}

[data-testid="stMetricLabel"] {
    color: #a9bddb !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

[data-testid="stMetricDelta"] {
    font-weight: 600;
}

div[data-testid="stTabs"] button {
    color: #c9d8ee;
    font-weight: 600;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #7dd3fc !important;
    border-bottom-color: #38bdf8 !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #2b3d59;
    border-radius: 12px;
    overflow: hidden;
}

.stDownloadButton button, .stButton button {
    border-radius: 10px;
    border: 1px solid #3b82f6;
    background: #15345b;
    color: white;
}

.rp-hero {
    background: linear-gradient(120deg, #152b49, #13213a);
    border: 1px solid #2f4a6d;
    border-radius: 22px;
    padding: 26px 30px;
    margin-bottom: 18px;
}

.rp-eyebrow {
    color: #7dd3fc;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 700;
}

.rp-sub {
    color: #b8c9e2;
    margin-top: 5px;
}

.section-note {
    color: #9fb2cc;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# PART 2 - DATASET LOADING
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "Data" / "Raw"
PROCESSED_DIR = PROJECT_DIR / "Data" / "Processed"

DATASET_FILES = {
    "sales": ["sales_transactions.csv"],
    "products": ["products.csv"],
    "stores": ["stores.csv"],
    "suppliers": ["suppliers.csv"],
    "inventory": [
        "inventory_analysis.csv",
        "raw_inventory_snapshots.csv"
    ],
    "low_stock": ["low_stock_report.csv"],
    "purchase_orders": ["purchase_orders.csv"],
    "supplier_performance": ["supplier_performance_reports.csv"],
    "category_inventory": ["category_inventory_analysis.csv"],
    "store_inventory": ["store_inventory_analysis.csv"],
    "monthly_sales": ["monthly_sales_analysis.csv"],
    "product_performance": ["product_performance.csv"],
    "store_performance": ["store_performance.csv"],
    "payment_analysis": ["payment_analysis.csv"],
    "data_quality": ["data_quality_report.csv"],
    "business_validation": ["business_validation_report.csv"],
    "incomplete_purchase_orders": ["incomplete_purchase_orders.csv"],
}


def normalize_filename(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


@st.cache_data(show_spinner=False)
def read_csv_file(file_path):
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin1")
    except Exception:
        return pd.DataFrame()

def find_dataset_file(file_names):
    base_dir = Path(__file__).resolve().parent

    for file_name in file_names:
        target = normalize_filename(Path(file_name).stem)

        for file_path in base_dir.rglob("*.csv"):
            if normalize_filename(file_path.stem) == target:
                return file_path
    return None

@st.cache_data(show_spinner=False)
def load_all_datasets():
    loaded_data = {}

    for dataset_name, file_names in DATASET_FILES.items():
        file_path = find_dataset_file(file_names)

        if file_path is not None:
            loaded_data[dataset_name] = read_csv_file(file_path)
        else:
            loaded_data[dataset_name] = pd.DataFrame()

    return loaded_data


# LOAD DATASETS
data = load_all_datasets()

sales = data["sales"]
products = data["products"]
stores = data["stores"]
suppliers = data["suppliers"]
inventory = data["inventory"]
low_stock = data["low_stock"]
purchase_orders = data["purchase_orders"]
supplier_performance = data["supplier_performance"]
category_inventory = data["category_inventory"]
store_inventory = data["store_inventory"]
monthly_sales = data["monthly_sales"]
product_performance = data["product_performance"]
store_performance = data["store_performance"]
payment_analysis = data["payment_analysis"]
data_quality = data["data_quality"]
business_validation = data["business_validation"]
incomplete_purchase_orders = data["incomplete_purchase_orders"]


# CLEAN COLUMN NAMES
for dataset in data.values():
    if not dataset.empty:
        dataset.columns = [
            str(column).strip().lower().replace(" ", "_")
            for column in dataset.columns
        ]


# DATA QUALITY REPORT
data_quality_report = data_quality.copy()


# ==========================================
# PART 3 - DATA CLEANING & HELPER FUNCTIONS
# ==========================================

def clean_numeric_columns(df, columns):
    if df.empty:
        return df

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    return df


# CLEAN SALES DATA
if not sales.empty:

    sales.columns = [
        str(column).strip().lower()
        for column in sales.columns
    ]

    if "transaction_date" in sales.columns:
        sales["transaction_date"] = pd.to_datetime(
            sales["transaction_date"],
            errors="coerce"
        )

    clean_numeric_columns(
        sales,
        [
            "quantity",
            "unit_price",
            "discount_pct",
            "revenue",
            "profit",
        ],
    )

    for column in ["store_id", "product_id", "payment_method"]:
        if column in sales.columns:
            sales[column] = sales[column].astype("string").str.strip()


# CLEAN OTHER DATASETS
clean_numeric_columns(
    products,
    ["unit_cost", "unit_price", "reorder_point", "lead_time_days"],
)

clean_numeric_columns(
    inventory,
    ["stock_on_hand", "reorder_point", "unit_cost", "unit_price",
     "lead_time_days", "inventory_cost"],
)

clean_numeric_columns(
    low_stock,
    ["stock_on_hand", "reorder_point", "unit_cost",
     "unit_price", "lead_time_days", "reorder_quantity"],
)

clean_numeric_columns(
    purchase_orders,
    ["ordered_quantity", "received_quantity",
     "delivery_delay_days", "fulfilment_pct", "short_quantity"],
)

clean_numeric_columns(
    supplier_performance,
    ["total_orders", "total_ordered", "total_received",
     "average_delay", "on_time_orders", "delayed_orders",
     "on_time_delivery_pct", "fulfilment_pct"],
)


# GENERAL FORMATTING FUNCTIONS
def format_number(value):
    try:
        return f"{float(value):,.0f}"
    except (ValueError, TypeError):
        return "0"


def format_currency(value):
    try:
        return f"₹{float(value):,.2f}"
    except (ValueError, TypeError):
        return "₹0.00"


def format_percentage(value):
    try:
        return f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return "0.00%"


def safe_sum(df, column):
    if df.empty or column not in df.columns:
        return 0

    return pd.to_numeric(
        df[column], errors="coerce"
    ).fillna(0).sum()


def safe_mean(df, column):
    if df.empty or column not in df.columns:
        return 0

    values = pd.to_numeric(
        df[column], errors="coerce"
    ).dropna()

    return values.mean() if not values.empty else 0


def show_metric(label, value, help_text=None):
    st.metric(
        label=label,
        value=value,
        help=help_text,
    )


# COMMON CHART DESIGN
CHART_TEMPLATE = "plotly_dark"


def style_chart(fig, height=380):
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans",
            color="#dce8f8",
            size=12,
        ),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        gridcolor="rgba(150,170,200,0.12)",
        zeroline=False,
    )

    return fig


def show_empty_message(message="No data available for this section."):
    st.info(message)


def show_section_heading(title, description=""):
    st.markdown(f"## {title}")

    if description:
        st.markdown(
            f'<p class="section-note">{description}</p>',
            unsafe_allow_html=True,
        )


# DATASET STATUS
def dataset_status(df):
    return "Available" if not df.empty else "Not found"


st.session_state["retailpulse_data_loaded"] = True

# ==========================================
# PART 4 - SIDEBAR NAVIGATION & FILTERS
# ==========================================

st.sidebar.markdown(
    """
    <div style="padding:10px 0 20px 0;">
        <div style="font-size:30px;">📊</div>
        <div style="font-size:22px;font-weight:700;">
            RetailPulse <span style="color:#38bdf8;">AI</span>
        </div>
        <div style="font-size:11px;color:#91a8c8;letter-spacing:2px;">
            RETAIL INTELLIGENCE PLATFORM
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

dashboard_pages = [
    "Executive Overview",
    "Sales Analytics",
    "Product Analytics",
    "Store Analytics",
    "Inventory Analytics",
    "Low Stock & Reorder",
    "Supplier Analytics",
    "Purchase Order Analytics",
    "Payment Analytics",
    "Category Analytics",
    "Data Quality",
    "Business Validation",
    "Data Explorer",
]

selected_page = st.sidebar.radio(
    "DASHBOARD NAVIGATION",
    dashboard_pages,
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Global Filters")

# STORE FILTER
store_options = ["All Stores"]

if not stores.empty and "store_id" in stores.columns:
    store_options += sorted(
        stores["store_id"].dropna().astype(str).unique().tolist()
    )
elif not sales.empty and "store_id" in sales.columns:
    store_options += sorted(
        sales["store_id"].dropna().astype(str).unique().tolist()
    )

selected_store = st.sidebar.selectbox(
    "Select Store",
    store_options,
)

# PAYMENT FILTER
payment_options = ["All Payment Methods"]

if not sales.empty and "payment_method" in sales.columns:
    payment_options += sorted(
        sales["payment_method"].dropna().astype(str).unique().tolist()
    )

selected_payment = st.sidebar.selectbox(
    "Payment Method",
    payment_options,
)

# DATE FILTER
date_filtered_sales = sales.copy()

if (
    not sales.empty
    and "transaction_date" in sales.columns
    and sales["transaction_date"].notna().any()
):
    min_date = sales["transaction_date"].min().date()
    max_date = sales["transaction_date"].max().date()

    selected_dates = st.sidebar.date_input(
        "Transaction Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates

        date_filtered_sales = date_filtered_sales[
            date_filtered_sales["transaction_date"].dt.date.between(
                start_date, end_date
            )
        ]

# APPLY STORE FILTER
if (
    selected_store != "All Stores"
    and "store_id" in date_filtered_sales.columns
):
    date_filtered_sales = date_filtered_sales[
        date_filtered_sales["store_id"].astype(str) == selected_store
    ]

# APPLY PAYMENT FILTER
if (
    selected_payment != "All Payment Methods"
    and "payment_method" in date_filtered_sales.columns
):
    date_filtered_sales = date_filtered_sales[
        date_filtered_sales["payment_method"].astype(str)
        == selected_payment
    ]

filtered_sales = date_filtered_sales.copy()

st.sidebar.markdown("---")

st.sidebar.caption(
    "RetailPulse AI | Interactive Retail Analytics"
)

st.sidebar.caption(
    f"Selected page: {selected_page}"
)

# MAIN PAGE HEADER
st.markdown(
    f"""
    <div class="rp-hero">
        <div class="rp-eyebrow">RETAILPULSE AI / ANALYTICS</div>
        <h1 style="margin:8px 0 4px 0;">{selected_page}</h1>
        <div class="rp-sub">
            Explore retail performance through interactive analytics
            and business intelligence.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if sales.empty:
    st.warning(
        "Sales Transactions CSV was not found. "
        "Please check Data/Raw/sales_transactions.csv."
    )
else:
    st.caption(
        f"Showing {len(filtered_sales):,} sales transactions "
        f"after applying the selected filters."
    )

# ==========================================
# PART 5 - EXECUTIVE OVERVIEW DASHBOARD
# ==========================================

if selected_page == "Executive Overview":

    show_section_heading(
        "📊 Executive Overview",
        "A complete snapshot of retail business performance."
    )

    if filtered_sales.empty:
        show_empty_message("No sales records available for the selected filters.")

    else:
        total_revenue = safe_sum(filtered_sales, "revenue")
        total_profit = safe_sum(filtered_sales, "profit")
        total_transactions = (
            filtered_sales["transaction_id"].nunique()
            if "transaction_id" in filtered_sales.columns
            else len(filtered_sales)
        )
        total_quantity = safe_sum(filtered_sales, "quantity")

        profit_margin = (
            (total_profit / total_revenue) * 100
            if total_revenue else 0
        )

        # KPI CARDS
        st.markdown("### Business Performance")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            show_metric("Total Revenue", format_currency(total_revenue))

        with c2:
            show_metric("Total Profit", format_currency(total_profit))

        with c3:
            show_metric("Transactions", format_number(total_transactions))

        with c4:
            show_metric("Quantity Sold", format_number(total_quantity))

        c5, c6 = st.columns(2)

        with c5:
            show_metric("Profit Margin", format_percentage(profit_margin))

        with c6:
            average_order_value = (
                total_revenue / total_transactions
                if total_transactions else 0
            )
            show_metric("Average Transaction Value", format_currency(average_order_value))

        st.markdown("---")

        # MONTHLY SALES TREND
        st.markdown("### 📈 Monthly Sales Trend")

        if "transaction_date" in filtered_sales.columns:
            trend_data = filtered_sales.dropna(
                subset=["transaction_date"]
            ).copy()

            if not trend_data.empty:
                trend_data["month"] = (
                    trend_data["transaction_date"].dt.to_period("M").astype(str)
                )

                monthly_data = (
                    trend_data.groupby("month", as_index=False)
                    .agg(
                        Revenue=("revenue", "sum"),
                        Profit=("profit", "sum"),
                    )
                )

                fig = px.line(
                    monthly_data,
                    x="month",
                    y=["Revenue", "Profit"],
                    markers=True,
                    title="Revenue & Profit Over Time",
                )

                fig = style_chart(fig, 420)
                st.plotly_chart(fig, use_container_width=True)

        # STORE AND PAYMENT ANALYSIS
        left, right = st.columns(2)

        with left:
            st.markdown("### 🏬 Store-wise Revenue")

            if "store_id" in filtered_sales.columns:
                store_data = (
                    filtered_sales.groupby("store_id", as_index=False)
                    .agg(Revenue=("revenue", "sum"))
                    .sort_values("Revenue", ascending=False)
                )

                if not stores.empty and {"store_id", "store_name"}.issubset(stores.columns):
                    store_data = store_data.merge(
                        stores[["store_id", "store_name"]],
                        on="store_id",
                        how="left",
                    )
                    store_data["Store"] = store_data["store_name"].fillna(
                        store_data["store_id"]
                    )
                else:
                    store_data["Store"] = store_data["store_id"]

                fig = px.bar(
                    store_data,
                    x="Store",
                    y="Revenue",
                    text_auto=".2s",
                    title="Revenue by Store",
                )
                st.plotly_chart(
                    style_chart(fig, 360),
                    use_container_width=True,
                )

        with right:
            st.markdown("### 💳 Payment Method Analysis")

            if "payment_method" in filtered_sales.columns:
                payment_data = (
                    filtered_sales.groupby("payment_method", as_index=False)
                    .agg(Revenue=("revenue", "sum"))
                )

                fig = px.pie(
                    payment_data,
                    names="payment_method",
                    values="Revenue",
                    hole=0.55,
                    title="Revenue Share by Payment Method",
                )
                st.plotly_chart(
                    style_chart(fig, 360),
                    use_container_width=True,
                )

        # TOP PRODUCTS
        st.markdown("### 🏆 Top 10 Products by Revenue")

        if "product_id" in filtered_sales.columns:
            top_products = (
                filtered_sales.groupby("product_id", as_index=False)
                .agg(
                    Revenue=("revenue", "sum"),
                    Quantity=("quantity", "sum"),
                    Profit=("profit", "sum"),
                )
                .sort_values("Revenue", ascending=False)
                .head(10)
            )

            if not products.empty and {"product_id", "product_name"}.issubset(products.columns):
                top_products = top_products.merge(
                    products[["product_id", "product_name"]],
                    on="product_id",
                    how="left",
                )
                top_products["Product"] = top_products["product_name"].fillna(
                    top_products["product_id"]
                )
            else:
                top_products["Product"] = top_products["product_id"]

            fig = px.bar(
                top_products.sort_values("Revenue"),
                x="Revenue",
                y="Product",
                orientation="h",
                text_auto=".2s",
                title="Top 10 Products",
            )
            st.plotly_chart(
                style_chart(fig, 450),
                use_container_width=True,
            )

            st.dataframe(
                top_products[
                    ["Product", "Revenue", "Quantity", "Profit"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        # EXPORT
        st.markdown("---")
        st.markdown("### 📥 Export Executive Overview Data")

        csv_data = filtered_sales.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Filtered Sales CSV",
            data=csv_data,
            file_name="retailpulse_executive_sales.csv",
            mime="text/csv",
        )

# ==========================================
# PART 6 - SALES ANALYTICS DASHBOARD
# ==========================================

if selected_page == "Sales Analytics":

    show_section_heading(
        "📈 Sales Analytics",
        "Detailed analysis of revenue, profit, transactions and discounts."
    )

    if filtered_sales.empty:
        show_empty_message("No sales data available for the selected filters.")

    else:
        # SALES KPI METRICS
        revenue = safe_sum(filtered_sales, "revenue")
        profit = safe_sum(filtered_sales, "profit")
        quantity = safe_sum(filtered_sales, "quantity")
        transactions = (
            filtered_sales["transaction_id"].nunique()
            if "transaction_id" in filtered_sales.columns
            else len(filtered_sales)
        )

        avg_price = safe_mean(filtered_sales, "unit_price")
        avg_discount = safe_mean(filtered_sales, "discount_pct")

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            show_metric("Sales Revenue", format_currency(revenue))

        with k2:
            show_metric("Sales Profit", format_currency(profit))

        with k3:
            show_metric("Units Sold", format_number(quantity))

        with k4:
            show_metric("Transactions", format_number(transactions))

        k5, k6 = st.columns(2)

        with k5:
            show_metric("Average Unit Price", format_currency(avg_price))

        with k6:
            show_metric("Average Discount", format_percentage(avg_discount))

        st.markdown("---")

        # DAILY SALES TREND
        st.markdown("### 📅 Daily Sales Trend")

        if "transaction_date" in filtered_sales.columns:
            daily_data = filtered_sales.dropna(
                subset=["transaction_date"]
            ).copy()

            if not daily_data.empty:
                daily_data = (
                    daily_data.groupby(
                        daily_data["transaction_date"].dt.date
                    )
                    .agg(
                        Revenue=("revenue", "sum"),
                        Profit=("profit", "sum"),
                    )
                    .reset_index()
                    .rename(columns={"transaction_date": "Date"})
                )

                fig = px.area(
                    daily_data,
                    x="Date",
                    y="Revenue",
                    title="Daily Revenue Trend",
                    markers=True,
                )

                st.plotly_chart(
                    style_chart(fig, 380),
                    use_container_width=True,
                )

        # MONTHLY REVENUE AND PROFIT
        st.markdown("### 📊 Monthly Revenue & Profit")

        if "transaction_date" in filtered_sales.columns:
            monthly = filtered_sales.dropna(
                subset=["transaction_date"]
            ).copy()

            if not monthly.empty:
                monthly["Month"] = (
                    monthly["transaction_date"]
                    .dt.to_period("M")
                    .astype(str)
                )

                monthly = (
                    monthly.groupby("Month", as_index=False)
                    .agg(
                        Revenue=("revenue", "sum"),
                        Profit=("profit", "sum"),
                    )
                )

                fig = px.bar(
                    monthly,
                    x="Month",
                    y=["Revenue", "Profit"],
                    barmode="group",
                    title="Monthly Revenue vs Profit",
                )

                st.plotly_chart(
                    style_chart(fig, 420),
                    use_container_width=True,
                )

        # STORE AND DISCOUNT ANALYSIS
        left, right = st.columns(2)

        with left:
            st.markdown("### 🏬 Store Sales Comparison")

            if "store_id" in filtered_sales.columns:
                store_sales = (
                    filtered_sales.groupby("store_id", as_index=False)
                    .agg(
                        Revenue=("revenue", "sum"),
                        Profit=("profit", "sum"),
                    )
                    .sort_values("Revenue", ascending=False)
                )

                fig = px.bar(
                    store_sales,
                    x="store_id",
                    y="Revenue",
                    color="Profit",
                    title="Revenue by Store",
                    text_auto=".2s",
                )

                st.plotly_chart(
                    style_chart(fig, 350),
                    use_container_width=True,
                )

        with right:
            st.markdown("### 🏷️ Discount Analysis")

            if "discount_pct" in filtered_sales.columns:
                discount_data = filtered_sales.copy()

                discount_data["Discount Group"] = pd.cut(
                    discount_data["discount_pct"],
                    bins=[-0.001, 0, 5, 10, 20, 100],
                    labels=[
                        "No Discount",
                        "0–5%",
                        "5–10%",
                        "10–20%",
                        "Above 20%",
                    ],
                    include_lowest=True,
                )

                discount_summary = (
                    discount_data.groupby(
                        "Discount Group",
                        observed=False,
                    )
                    .agg(
                        Revenue=("revenue", "sum"),
                        Transactions=("transaction_id", "nunique"),
                    )
                    .reset_index()
                )

                fig = px.bar(
                    discount_summary,
                    x="Discount Group",
                    y="Revenue",
                    title="Revenue by Discount Range",
                    text_auto=".2s",
                )

                st.plotly_chart(
                    style_chart(fig, 350),
                    use_container_width=True,
                )

        # QUANTITY VS REVENUE
        st.markdown("### 🔍 Quantity vs Revenue Relationship")

        if {"quantity", "revenue"}.issubset(filtered_sales.columns):
            fig = px.scatter(
                filtered_sales,
                x="quantity",
                y="revenue",
                color="profit",
                hover_data=["product_id", "store_id"]
                if {"product_id", "store_id"}.issubset(filtered_sales.columns)
                else None,
                title="Transaction Quantity vs Revenue",
                opacity=0.65,
            )

            st.plotly_chart(
                style_chart(fig, 420),
                use_container_width=True,
            )

        # BEST SALES DAYS
        st.markdown("### 🏆 Highest Revenue Days")

        if "transaction_date" in filtered_sales.columns:
            best_days = filtered_sales.dropna(
                subset=["transaction_date"]
            ).copy()

            if not best_days.empty:
                best_days = (
                    best_days.groupby(
                        best_days["transaction_date"].dt.date
                    )
                    .agg(
                        Revenue=("revenue", "sum"),
                        Profit=("profit", "sum"),
                    )
                    .reset_index()
                    .rename(columns={"transaction_date": "Date"})
                    .sort_values("Revenue", ascending=False)
                    .head(10)
                )

                st.dataframe(
                    best_days,
                    use_container_width=True,
                    hide_index=True,
                )

        # TRANSACTION DETAILS
        st.markdown("---")
        st.markdown("### 📋 Detailed Sales Transactions")

        st.dataframe(
            filtered_sales,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "📥 Download Sales Analytics CSV",
            data=filtered_sales.to_csv(index=False).encode("utf-8"),
            file_name="retailpulse_sales_analytics.csv",
            mime="text/csv",
        )

# ==========================================
# PART 7 - PRODUCT ANALYTICS DASHBOARD
# ==========================================

if selected_page == "Product Analytics":

    show_section_heading(
        "🏆 Product Analytics",
        "Analyze product revenue, profitability, demand and pricing."
    )

    if filtered_sales.empty or "product_id" not in filtered_sales.columns:
        show_empty_message("Product sales data is not available.")

    else:
        # PRODUCT PERFORMANCE SUMMARY
        product_summary = (
            filtered_sales.groupby("product_id", as_index=False)
            .agg(
                Total_Revenue=("revenue", "sum"),
                Total_Profit=("profit", "sum"),
                Total_Quantity=("quantity", "sum"),
                Transactions=("transaction_id", "nunique")
                if "transaction_id" in filtered_sales.columns
                else ("product_id", "size"),
            )
        )

        product_summary["Profit_Margin"] = np.where(
            product_summary["Total_Revenue"] != 0,
            product_summary["Total_Profit"]
            / product_summary["Total_Revenue"] * 100,
            0,
        )

        # ADD PRODUCT MASTER DETAILS
        if not products.empty and "product_id" in products.columns:
            product_columns = [
                column for column in [
                    "product_id",
                    "product_name",
                    "category",
                    "unit_cost",
                    "unit_price",
                    "reorder_point",
                    "lead_time_days",
                ]
                if column in products.columns
            ]

            product_summary = product_summary.merge(
                products[product_columns].drop_duplicates("product_id"),
                on="product_id",
                how="left",
            )

        if "product_name" in product_summary.columns:
            product_summary["Product"] = (
                product_summary["product_name"]
                .fillna(product_summary["product_id"])
            )
        else:
            product_summary["Product"] = product_summary["product_id"]

        # KPI CARDS
        st.markdown("### Product Performance KPIs")

        best_product = product_summary.sort_values(
            "Total_Revenue", ascending=False
        ).iloc[0]

        loss_products = int(
            (product_summary["Total_Profit"] < 0).sum()
        )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            show_metric(
                "Products Sold",
                format_number(product_summary["product_id"].nunique()),
            )

        with k2:
            show_metric(
                "Product Revenue",
                format_currency(product_summary["Total_Revenue"].sum()),
            )

        with k3:
            show_metric(
                "Product Profit",
                format_currency(product_summary["Total_Profit"].sum()),
            )

        with k4:
            show_metric(
                "Loss-making Products",
                format_number(loss_products),
            )

        st.markdown(
            f"**Top Revenue Product:** {best_product['Product']} "
            f"({format_currency(best_product['Total_Revenue'])})"
        )

        st.markdown("---")

        # TOP PRODUCTS BY REVENUE AND PROFIT
        left, right = st.columns(2)

        with left:
            st.markdown("### 💰 Top 10 Products by Revenue")

            top_revenue = product_summary.nlargest(
                10, "Total_Revenue"
            ).sort_values("Total_Revenue")

            fig = px.bar(
                top_revenue,
                x="Total_Revenue",
                y="Product",
                orientation="h",
                text_auto=".2s",
                title="Top Products by Revenue",
            )

            st.plotly_chart(
                style_chart(fig, 430),
                use_container_width=True,
            )

        with right:
            st.markdown("### 📈 Top 10 Products by Profit")

            top_profit = product_summary.nlargest(
                10, "Total_Profit"
            ).sort_values("Total_Profit")

            fig = px.bar(
                top_profit,
                x="Total_Profit",
                y="Product",
                orientation="h",
                text_auto=".2s",
                title="Top Products by Profit",
            )

            st.plotly_chart(
                style_chart(fig, 430),
                use_container_width=True,
            )

        # QUANTITY PERFORMANCE
        st.markdown("### 📦 Best-selling Products by Quantity")

        top_quantity = product_summary.nlargest(
            10, "Total_Quantity"
        ).sort_values("Total_Quantity")

        fig = px.bar(
            top_quantity,
            x="Total_Quantity",
            y="Product",
            orientation="h",
            color="Total_Revenue",
            text_auto=".2s",
            title="Top 10 Products by Quantity Sold",
        )

        st.plotly_chart(
            style_chart(fig, 420),
            use_container_width=True,
        )

        # CATEGORY ANALYSIS
        if "category" in product_summary.columns:
            st.markdown("### 🗂️ Category-wise Product Performance")

            category_summary = (
                product_summary.groupby("category", as_index=False)
                .agg(
                    Revenue=("Total_Revenue", "sum"),
                    Profit=("Total_Profit", "sum"),
                    Quantity=("Total_Quantity", "sum"),
                )
                .sort_values("Revenue", ascending=False)
            )

            left, right = st.columns(2)

            with left:
                fig = px.pie(
                    category_summary,
                    names="category",
                    values="Revenue",
                    hole=0.5,
                    title="Revenue Share by Category",
                )

                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )

            with right:
                fig = px.bar(
                    category_summary,
                    x="category",
                    y=["Revenue", "Profit"],
                    barmode="group",
                    title="Category Revenue vs Profit",
                )

                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )

            st.dataframe(
                category_summary,
                use_container_width=True,
                hide_index=True,
            )

        # PRODUCT PRICING ANALYSIS
        if {"unit_cost", "unit_price"}.issubset(product_summary.columns):
            st.markdown("### 🏷️ Product Cost vs Selling Price")

            pricing_data = product_summary.dropna(
                subset=["unit_cost", "unit_price"]
            ).copy()

            if not pricing_data.empty:
                pricing_data["Pricing Status"] = np.where(
                    pricing_data["unit_cost"] > pricing_data["unit_price"],
                    "Cost Above Price",
                    "Price Above Cost",
                )

                fig = px.scatter(
                    pricing_data,
                    x="unit_cost",
                    y="unit_price",
                    color="Pricing Status",
                    hover_name="Product",
                    title="Product Cost vs Selling Price",
                )

                st.plotly_chart(
                    style_chart(fig, 420),
                    use_container_width=True,
                )

                pricing_risks = pricing_data[
                    pricing_data["unit_cost"] > pricing_data["unit_price"]
                ]

                if not pricing_risks.empty:
                    st.warning(
                        f"{len(pricing_risks)} products have unit cost "
                        "higher than unit selling price. Review their pricing."
                    )

                    st.dataframe(
                        pricing_risks[
                            ["Product", "unit_cost", "unit_price"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

        # LOSS-MAKING PRODUCTS
        st.markdown("### ⚠️ Products with Negative Profit")

        loss_data = product_summary[
            product_summary["Total_Profit"] < 0
        ].sort_values("Total_Profit")

        if loss_data.empty:
            st.success("No negative-profit products in the selected sales data.")
        else:
            st.dataframe(
                loss_data[
                    ["Product", "Total_Revenue", "Total_Profit",
                     "Total_Quantity", "Profit_Margin"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        # PRODUCT DETAILS
        st.markdown("---")
        st.markdown("### 📋 Complete Product Performance")

        st.dataframe(
            product_summary.sort_values(
                "Total_Revenue", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "📥 Download Product Analytics",
            data=product_summary.to_csv(index=False).encode("utf-8"),
            file_name="retailpulse_product_analytics.csv",
            mime="text/csv",
        )

# ==========================================
# PART 8 - STORE ANALYTICS DASHBOARD
# ==========================================

if selected_page == "Store Analytics":

    show_section_heading(
        "🏬 Store Analytics",
        "Compare store performance, revenue contribution and profitability."
    )

    if filtered_sales.empty or "store_id" not in filtered_sales.columns:
        show_empty_message("Store sales data is not available.")

    else:
        # STORE PERFORMANCE SUMMARY
        store_summary = (
            filtered_sales.groupby("store_id", as_index=False)
            .agg(
                Revenue=("revenue", "sum"),
                Profit=("profit", "sum"),
                Quantity=("quantity", "sum"),
                Transactions=("transaction_id", "nunique")
                if "transaction_id" in filtered_sales.columns
                else ("store_id", "size"),
            )
        )

        store_summary["Profit_Margin"] = np.where(
            store_summary["Revenue"] != 0,
            store_summary["Profit"] / store_summary["Revenue"] * 100,
            0,
        )

        store_summary["Average_Transaction_Value"] = np.where(
            store_summary["Transactions"] != 0,
            store_summary["Revenue"] / store_summary["Transactions"],
            0,
        )

        # STORE MASTER DETAILS
        if not stores.empty and {
            "store_id", "store_name"
        }.issubset(stores.columns):

            store_summary = store_summary.merge(
                stores[["store_id", "store_name", "region"]]
                .drop_duplicates("store_id"),
                on="store_id",
                how="left",
            )

            store_summary["Store"] = (
                store_summary["store_name"].fillna(
                    store_summary["store_id"]
                )
            )
        else:
            store_summary["Store"] = store_summary["store_id"]

        store_summary = store_summary.sort_values(
            "Revenue", ascending=False
        )

        # KPI CARDS
        st.markdown("### Store Performance KPIs")

        top_store = store_summary.iloc[0]

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            show_metric(
                "Active Stores",
                format_number(store_summary["store_id"].nunique()),
            )

        with k2:
            show_metric(
                "Total Revenue",
                format_currency(store_summary["Revenue"].sum()),
            )

        with k3:
            show_metric(
                "Total Profit",
                format_currency(store_summary["Profit"].sum()),
            )

        with k4:
            show_metric(
                "Top Store",
                str(top_store["Store"]),
            )

        st.markdown(
            f"**Highest Revenue Store:** {top_store['Store']} — "
            f"{format_currency(top_store['Revenue'])}"
        )

        st.markdown("---")

        # REVENUE COMPARISON
        left, right = st.columns(2)

        with left:
            st.markdown("### 💰 Store-wise Revenue")

            fig = px.bar(
                store_summary.sort_values("Revenue"),
                x="Revenue",
                y="Store",
                orientation="h",
                color="Revenue",
                text_auto=".2s",
                title="Revenue by Store",
            )

            st.plotly_chart(
                style_chart(fig, 390),
                use_container_width=True,
            )

        with right:
            st.markdown("### 📈 Store-wise Profit")

            fig = px.bar(
                store_summary.sort_values("Profit"),
                x="Profit",
                y="Store",
                orientation="h",
                color="Profit",
                text_auto=".2s",
                title="Profit by Store",
            )

            st.plotly_chart(
                style_chart(fig, 390),
                use_container_width=True,
            )

        # TRANSACTION COMPARISON
        st.markdown("### 🧾 Store Transaction Analysis")

        fig = px.bar(
            store_summary,
            x="Store",
            y="Transactions",
            color="Quantity",
            text_auto=".2s",
            title="Transactions by Store",
        )

        st.plotly_chart(
            style_chart(fig, 380),
            use_container_width=True,
        )

        # REVENUE CONTRIBUTION
        left, right = st.columns(2)

        with left:
            st.markdown("### 🥧 Revenue Contribution")

            fig = px.pie(
                store_summary,
                names="Store",
                values="Revenue",
                hole=0.55,
                title="Store Revenue Share",
            )

            st.plotly_chart(
                style_chart(fig, 370),
                use_container_width=True,
            )

        with right:
            st.markdown("### 🎯 Profit Margin Comparison")

            fig = px.bar(
                store_summary.sort_values("Profit_Margin"),
                x="Store",
                y="Profit_Margin",
                text_auto=".2f",
                title="Profit Margin by Store",
            )

            st.plotly_chart(
                style_chart(fig, 370),
                use_container_width=True,
            )

        # STORE RANKING
        st.markdown("### 🏆 Store Performance Ranking")

        ranking = store_summary.copy()
        ranking.insert(
            0,
            "Revenue Rank",
            range(1, len(ranking) + 1),
        )

        ranking["Revenue"] = ranking["Revenue"].round(2)
        ranking["Profit"] = ranking["Profit"].round(2)
        ranking["Profit_Margin"] = ranking["Profit_Margin"].round(2)
        ranking["Average_Transaction_Value"] = (
            ranking["Average_Transaction_Value"].round(2)
        )

        st.dataframe(
            ranking,
            use_container_width=True,
            hide_index=True,
        )

        # STORE DETAIL EXPORT
        st.download_button(
            "📥 Download Store Analytics",
            data=ranking.to_csv(index=False).encode("utf-8"),
            file_name="retailpulse_store_analytics.csv",
            mime="text/csv",
        )

# ==========================================
# PART 9 - INVENTORY ANALYTICS DASHBOARD
# ==========================================

if selected_page == "Inventory Analytics":

    show_section_heading(
        "📦 Inventory Analytics",
        "Monitor stock levels, inventory value and product availability."
    )

    if inventory.empty:
        show_empty_message(
            "Inventory Analysis CSV was not found. "
            "Check Data/Processed or Data/Raw."
        )

    else:
        inv = inventory.copy()

        # INVENTORY KPI METRICS
        total_stock = safe_sum(inv, "stock_on_hand")
        inventory_value = safe_sum(inv, "inventory_cost")
        unique_products = (
            inv["product_id"].nunique()
            if "product_id" in inv.columns else 0
        )
        unique_stores = (
            inv["store_id"].nunique()
            if "store_id" in inv.columns else 0
        )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            show_metric("Total Stock Units", format_number(total_stock))

        with k2:
            show_metric("Inventory Value", format_currency(inventory_value))

        with k3:
            show_metric("Products", format_number(unique_products))

        with k4:
            show_metric("Stores", format_number(unique_stores))

        st.markdown("---")

        # STOCK STATUS
        st.markdown("### 🚦 Stock Status Analysis")

        if {"stock_on_hand", "reorder_point"}.issubset(inv.columns):

            inv["Calculated Status"] = np.where(
                inv["stock_on_hand"] <= inv["reorder_point"],
                "Low Stock",
                "Healthy Stock",
            )

            status_data = (
                inv.groupby("Calculated Status")
                .size()
                .reset_index(name="Items")
            )

            left, right = st.columns(2)

            with left:
                fig = px.pie(
                    status_data,
                    names="Calculated Status",
                    values="Items",
                    hole=0.55,
                    title="Stock Status Distribution",
                )

                st.plotly_chart(
                    style_chart(fig, 360),
                    use_container_width=True,
                )

            with right:
                fig = px.bar(
                    status_data,
                    x="Calculated Status",
                    y="Items",
                    color="Calculated Status",
                    text_auto=True,
                    title="Low Stock vs Healthy Stock",
                )

                st.plotly_chart(
                    style_chart(fig, 360),
                    use_container_width=True,
                )

        # STORE-WISE STOCK
        left, right = st.columns(2)

        with left:
            st.markdown("### 🏬 Store-wise Inventory")

            if "store_id" in inv.columns:
                store_stock = (
                    inv.groupby("store_id", as_index=False)
                    .agg(
                        Total_Stock=("stock_on_hand", "sum"),
                        Inventory_Cost=("inventory_cost", "sum")
                        if "inventory_cost" in inv.columns
                        else ("stock_on_hand", "sum"),
                    )
                )

                fig = px.bar(
                    store_stock,
                    x="store_id",
                    y="Total_Stock",
                    color="Inventory_Cost",
                    text_auto=".2s",
                    title="Stock Available by Store",
                )

                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )

        with right:
            st.markdown("### 🗂️ Category-wise Inventory")

            if "category" in inv.columns:
                category_stock = (
                    inv.groupby("category", as_index=False)
                    .agg(
                        Total_Stock=("stock_on_hand", "sum"),
                        Inventory_Cost=("inventory_cost", "sum")
                        if "inventory_cost" in inv.columns
                        else ("stock_on_hand", "sum"),
                    )
                )

                fig = px.pie(
                    category_stock,
                    names="category",
                    values="Total_Stock",
                    hole=0.5,
                    title="Stock Share by Category",
                )

                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )

        # INVENTORY COST
        if "category" in inv.columns and "inventory_cost" in inv.columns:
            st.markdown("### 💰 Inventory Cost by Category")

            cost_data = (
                inv.groupby("category", as_index=False)
                .agg(Inventory_Cost=("inventory_cost", "sum"))
                .sort_values("Inventory_Cost", ascending=False)
            )

            fig = px.bar(
                cost_data,
                x="category",
                y="Inventory_Cost",
                text_auto=".2s",
                title="Inventory Value by Category",
            )

            st.plotly_chart(
                style_chart(fig, 380),
                use_container_width=True,
            )

        # PRODUCT STOCK ANALYSIS
        st.markdown("### 📊 Product-wise Stock Levels")

        if "product_id" in inv.columns:
            product_stock = (
                inv.groupby("product_id", as_index=False)
                .agg(
                    Stock=("stock_on_hand", "sum"),
                    Reorder_Point=("reorder_point", "sum"),
                )
            )

            if not products.empty and {
                "product_id", "product_name"
            }.issubset(products.columns):

                product_stock = product_stock.merge(
                    products[["product_id", "product_name"]],
                    on="product_id",
                    how="left",
                )

                product_stock["Product"] = (
                    product_stock["product_name"].fillna(
                        product_stock["product_id"]
                    )
                )
            else:
                product_stock["Product"] = product_stock["product_id"]

            fig = px.bar(
                product_stock.sort_values("Stock"),
                x="Stock",
                y="Product",
                orientation="h",
                color="Reorder_Point",
                title="Stock vs Reorder Point",
            )

            st.plotly_chart(
                style_chart(fig, 480),
                use_container_width=True,
            )

            st.dataframe(
                product_stock[
                    ["Product", "Stock", "Reorder_Point"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        # INVENTORY DATA TABLE
        st.markdown("---")
        st.markdown("### 📋 Detailed Inventory Records")

        st.dataframe(
            inv,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "📥 Download Inventory Analytics",
            data=inv.to_csv(index=False).encode("utf-8"),
            file_name="retailpulse_inventory_analytics.csv",
            mime="text/csv",
        )

# ==========================================
# PART 10 - LOW STOCK & REORDER DASHBOARD
# ==========================================

if selected_page == "Low Stock & Reorder":

    show_section_heading(
        "🚨 Low Stock & Reorder",
        "Identify inventory shortages and prioritize replenishment."
    )

    if low_stock.empty:
        st.warning(
            "Low Stock Report CSV was not found. "
            "Check Data/Processed or Data/Raw."
        )

    else:
        low = low_stock.copy()

        # NUMERIC CLEANING
        clean_numeric_columns(
            low,
            [
                "stock_on_hand",
                "reorder_point",
                "reorder_quantity",
                "unit_cost",
                "lead_time_days",
            ],
        )

        # LOW STOCK KPI
        total_low_items = len(low)

        total_reorder_quantity = safe_sum(
            low, "reorder_quantity"
        )

        estimated_reorder_cost = (
            safe_sum(low, "reorder_quantity")
            * safe_mean(low, "unit_cost")
        )

        affected_stores = (
            low["store_id"].nunique()
            if "store_id" in low.columns else 0
        )

        affected_products = (
            low["product_id"].nunique()
            if "product_id" in low.columns else 0
        )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            show_metric("Low Stock Records", format_number(total_low_items))

        with k2:
            show_metric(
                "Reorder Quantity",
                format_number(total_reorder_quantity),
            )

        with k3:
            show_metric("Affected Stores", format_number(affected_stores))

        with k4:
            show_metric(
                "Affected Products",
                format_number(affected_products),
            )

        st.markdown(
            f"**Estimated Reorder Cost:** "
            f"{format_currency(estimated_reorder_cost)}"
        )

        st.caption(
            "Estimated cost uses total reorder quantity multiplied by "
            "average unit cost; it is an approximate planning figure."
        )

        st.markdown("---")

        # PRIORITY CLASSIFICATION
        if {"stock_on_hand", "reorder_point"}.issubset(low.columns):

            low["Stock Gap"] = (
                low["reorder_point"] - low["stock_on_hand"]
            )

            low["Priority"] = np.select(
                [
                    low["stock_on_hand"] <= 0,
                    low["stock_on_hand"] < low["reorder_point"] * 0.5,
                    low["stock_on_hand"] <= low["reorder_point"],
                ],
                [
                    "Critical",
                    "High",
                    "Medium",
                ],
                default="Normal",
            )

        else:
            low["Priority"] = "Review"

        st.markdown("### ⚠️ Reorder Priority Overview")

        priority_data = (
            low.groupby("Priority")
            .size()
            .reset_index(name="Items")
        )

        fig = px.bar(
            priority_data,
            x="Priority",
            y="Items",
            color="Priority",
            text_auto=True,
            title="Inventory Reorder Priority",
        )

        st.plotly_chart(
            style_chart(fig, 350),
            use_container_width=True,
        )

        # STORE-WISE LOW STOCK
        left, right = st.columns(2)

        with left:
            st.markdown("### 🏬 Store-wise Low Stock")

            if "store_id" in low.columns:
                store_low = (
                    low.groupby("store_id", as_index=False)
                    .agg(
                        Low_Stock_Items=("product_id", "count")
                        if "product_id" in low.columns
                        else ("store_id", "size"),
                        Reorder_Quantity=("reorder_quantity", "sum")
                        if "reorder_quantity" in low.columns
                        else ("store_id", "size"),
                    )
                )

                fig = px.bar(
                    store_low,
                    x="store_id",
                    y="Low_Stock_Items",
                    color="Reorder_Quantity",
                    text_auto=True,
                    title="Low Stock Items by Store",
                )

                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )

        with right:
            st.markdown("### 🗂️ Category-wise Stock Risk")

            if "category" in low.columns:
                category_low = (
                    low.groupby("category", as_index=False)
                    .agg(
                        Low_Stock_Items=("product_id", "count")
                        if "product_id" in low.columns
                        else ("category", "size"),
                    )
                )

                fig = px.pie(
                    category_low,
                    names="category",
                    values="Low_Stock_Items",
                    hole=0.5,
                    title="Low Stock Distribution by Category",
                )

                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )

        # LOW STOCK PRODUCT DETAILS
        st.markdown("### 📦 Products Requiring Reorder")

        display_columns = [
            column for column in [
                "store_id",
                "product_id",
                "product_name",
                "category",
                "stock_on_hand",
                "reorder_point",
                "Stock Gap",
                "reorder_quantity",
                "lead_time_days",
                "Priority",
            ]
            if column in low.columns
        ]

        if "Priority" in low.columns:
            priority_order = {
                "Critical": 0,
                "High": 1,
                "Medium": 2,
                "Normal": 3,
                "Review": 4,
            }

            low["_priority_order"] = low["Priority"].map(
                priority_order
            ).fillna(5)

            low = low.sort_values(
                ["_priority_order", "Stock Gap"],
                ascending=[True, False],
            )

        st.dataframe(
            low[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        # REORDER EXPORT
        st.markdown("---")

        st.download_button(
            "📥 Download Reorder Report",
            data=low[display_columns].to_csv(
                index=False
            ).encode("utf-8"),
            file_name="retailpulse_reorder_report.csv",
            mime="text/csv",
        )

# ==========================================
# PART 11 - SUPPLIER ANALYTICS DASHBOARD
# ==========================================

if selected_page == "Supplier Analytics":

    show_section_heading(
        "🚚 Supplier Analytics",
        "Evaluate supplier reliability, delivery performance and fulfilment."
    )

    # USE SUPPLIER PERFORMANCE REPORT
    if not supplier_performance.empty:
        supplier_summary = supplier_performance.copy()

    # OTHERWISE DERIVE FROM PURCHASE ORDERS
    elif not purchase_orders.empty and "supplier_id" in purchase_orders.columns:

        po = purchase_orders.copy()

        supplier_summary = (
            po.groupby("supplier_id", as_index=False)
            .agg(
                Total_Orders=("po_id", "nunique")
                if "po_id" in po.columns
                else ("supplier_id", "size"),
                Total_Ordered=("ordered_quantity", "sum"),
                Total_Received=("received_quantity", "sum"),
                Average_Delay=("delivery_delay_days", "mean")
                if "delivery_delay_days" in po.columns
                else ("ordered_quantity", "mean"),
            )
        )

        supplier_summary["Fulfilment_Pct"] = np.where(
            supplier_summary["Total_Ordered"] > 0,
            supplier_summary["Total_Received"]
            / supplier_summary["Total_Ordered"] * 100,
            0,
        )

    else:
        supplier_summary = pd.DataFrame()

    if supplier_summary.empty:
        show_empty_message(
            "Supplier Performance Reports or Purchase Orders data not found."
        )

    else:
        # NORMALIZE IMPORTANT COLUMN NAMES
        supplier_summary.columns = [
            str(c).strip().lower() for c in supplier_summary.columns
        ]

        # KPI METRICS
        total_suppliers = (
            supplier_summary["supplier_id"].nunique()
            if "supplier_id" in supplier_summary.columns
            else len(supplier_summary)
        )

        total_orders = safe_sum(
            supplier_summary,
            "total_orders" if "total_orders" in supplier_summary.columns
            else "total_ordered",
        )

        total_ordered = safe_sum(
            supplier_summary, "total_ordered"
        )

        total_received = safe_sum(
            supplier_summary, "total_received"
        )

        average_delay = safe_mean(
            supplier_summary, "average_delay"
        )

        overall_fulfilment = (
            total_received / total_ordered * 100
            if total_ordered else 0
        )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            show_metric("Suppliers", format_number(total_suppliers))

        with k2:
            show_metric("Total Orders", format_number(total_orders))

        with k3:
            show_metric("Overall Fulfilment", format_percentage(overall_fulfilment))

        with k4:
            show_metric("Average Delay", f"{average_delay:.2f} days")

        st.markdown("---")

        # SUPPLIER NAME
        if (
            not suppliers.empty
            and {"supplier_id", "supplier_name"}.issubset(suppliers.columns)
            and "supplier_id" in supplier_summary.columns
        ):
            supplier_summary = supplier_summary.merge(
                suppliers[["supplier_id", "supplier_name"]]
                .drop_duplicates("supplier_id"),
                on="supplier_id",
                how="left",
                suffixes=("", "_master"),
            )

        if "supplier_name" in supplier_summary.columns:
            supplier_summary["Supplier"] = (
                supplier_summary["supplier_name"]
                .fillna(supplier_summary["supplier_id"])
            )
        elif "supplier_id" in supplier_summary.columns:
            supplier_summary["Supplier"] = supplier_summary["supplier_id"]
        else:
            supplier_summary["Supplier"] = "Supplier"

        # ON-TIME DELIVERY
        if "on_time_delivery_pct" in supplier_summary.columns:
            supplier_summary["On_Time_Pct"] = pd.to_numeric(
                supplier_summary["on_time_delivery_pct"],
                errors="coerce",
            )
        elif {"on_time_orders", "total_orders"}.issubset(
            supplier_summary.columns
        ):
            supplier_summary["On_Time_Pct"] = np.where(
                supplier_summary["total_orders"] > 0,
                supplier_summary["on_time_orders"]
                / supplier_summary["total_orders"] * 100,
                0,
            )
        else:
            supplier_summary["On_Time_Pct"] = np.nan

        # FULFILMENT PERCENTAGE
        if "fulfilment_pct" in supplier_summary.columns:
            supplier_summary["Fulfilment_Pct"] = pd.to_numeric(
                supplier_summary["fulfilment_pct"],
                errors="coerce",
            )
        elif {"total_ordered", "total_received"}.issubset(
            supplier_summary.columns
        ):
            supplier_summary["Fulfilment_Pct"] = np.where(
                supplier_summary["total_ordered"] > 0,
                supplier_summary["total_received"]
                / supplier_summary["total_ordered"] * 100,
                0,
            )

        # DELIVERY PERFORMANCE CHART
        st.markdown("### 🚛 Supplier Delivery Performance")

        if "On_Time_Pct" in supplier_summary.columns:
            delivery_data = supplier_summary.sort_values(
                "On_Time_Pct", ascending=True
            )

            fig = px.bar(
                delivery_data,
                x="On_Time_Pct",
                y="Supplier",
                orientation="h",
                color="On_Time_Pct",
                text_auto=".2f",
                title="On-time Delivery Percentage by Supplier",
            )

            st.plotly_chart(
                style_chart(fig, 400),
                use_container_width=True,
            )

        # ORDER FULFILMENT
        left, right = st.columns(2)

        with left:
            st.markdown("### 📦 Order Fulfilment")

            if "Fulfilment_Pct" in supplier_summary.columns:
                fig = px.bar(
                    supplier_summary.sort_values("Fulfilment_Pct"),
                    x="Supplier",
                    y="Fulfilment_Pct",
                    text_auto=".2f",
                    title="Fulfilment Percentage",
                )

                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )

        with right:
            st.markdown("### ⏱️ Average Delivery Delay")

            if "average_delay" in supplier_summary.columns:
                fig = px.bar(
                    supplier_summary.sort_values("average_delay"),
                    x="Supplier",
                    y="average_delay",
                    text_auto=".2f",
                    title="Average Delay in Days",
                )

                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )

        # ORDER QUANTITY ANALYSIS
        st.markdown("### 📊 Ordered vs Received Quantity")

        if {"total_ordered", "total_received"}.issubset(
            supplier_summary.columns
        ):
            quantity_data = supplier_summary.melt(
                id_vars=["Supplier"],
                value_vars=["total_ordered", "total_received"],
                var_name="Quantity Type",
                value_name="Quantity",
            )

            fig = px.bar(
                quantity_data,
                x="Supplier",
                y="Quantity",
                color="Quantity Type",
                barmode="group",
                title="Ordered Quantity vs Received Quantity",
            )

            st.plotly_chart(
                style_chart(fig, 400),
                use_container_width=True,
            )

        # SUPPLIER RANKING
        st.markdown("### 🏆 Supplier Performance Ranking")

        if "Fulfilment_Pct" in supplier_summary.columns:
            ranking = supplier_summary.sort_values(
                ["Fulfilment_Pct", "On_Time_Pct"]
                if "On_Time_Pct" in supplier_summary.columns
                else ["Fulfilment_Pct"],
                ascending=False,
            ).copy()

            ranking.insert(
                0,
                "Rank",
                range(1, len(ranking) + 1),
            )

            st.dataframe(
                ranking,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(
                supplier_summary,
                use_container_width=True,
                hide_index=True,
            )

        # SUPPLIER EXPORT
        st.download_button(
            "📥 Download Supplier Analytics",
            data=supplier_summary.to_csv(index=False).encode("utf-8"),
            file_name="retailpulse_supplier_analytics.csv",
            mime="text/csv",
        )

# ==========================================
# PART 12 - PURCHASE ORDER ANALYTICS
# ==========================================

if selected_page == "Purchase Order Analytics":

    show_section_heading(
        "📦 Purchase Order Analytics",
        "Track purchase orders, received quantities, pending items and delivery delays."
    )

    if purchase_orders.empty:
        show_empty_message(
            "Purchase Orders dataset not found. Check the purchase_orders.csv file in Data/Raw."
        )

    else:
        po = purchase_orders.copy()

        # CLEAN DATA
        for col in [
            "ordered_quantity",
            "received_quantity",
            "delivery_delay_days",
            "fulfilment_pct",
            "short_quantity",
        ]:
            if col in po.columns:
                po[col] = pd.to_numeric(po[col], errors="coerce").fillna(0)

        if "order_date" in po.columns:
            po["order_date"] = pd.to_datetime(
                po["order_date"], errors="coerce"
            )

        # CALCULATE PENDING QUANTITY
        if {"ordered_quantity", "received_quantity"}.issubset(po.columns):
            po["pending_quantity"] = (
                po["ordered_quantity"] - po["received_quantity"]
            ).clip(lower=0)
        else:
            po["pending_quantity"] = 0

        # DELIVERY STATUS
        if "delivery_status" in po.columns:
            po["delivery_status"] = (
                po["delivery_status"]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )
        else:
            po["delivery_status"] = "Unknown"

        # KPI CALCULATIONS
        total_po = (
            po["po_id"].nunique()
            if "po_id" in po.columns
            else len(po)
        )

        total_ordered = po["ordered_quantity"].sum() \
            if "ordered_quantity" in po.columns else 0

        total_received = po["received_quantity"].sum() \
            if "received_quantity" in po.columns else 0

        total_pending = po["pending_quantity"].sum()

        fulfilment = (
            total_received / total_ordered * 100
            if total_ordered > 0 else 0
        )

        delayed_orders = (
            po["delivery_status"].str.lower().str.contains(
                "delay|late|pending"
            ).sum()
        )

        # KPI CARDS
        k1, k2, k3, k4, k5 = st.columns(5)

        with k1:
            show_metric("Total POs", format_number(total_po))

        with k2:
            show_metric("Ordered Qty", format_number(total_ordered))

        with k3:
            show_metric("Received Qty", format_number(total_received))

        with k4:
            show_metric("Pending Qty", format_number(total_pending))

        with k5:
            show_metric("Fulfilment", format_percentage(fulfilment))

        st.markdown("---")

        # ORDERED VS RECEIVED
        st.markdown("### 📊 Ordered vs Received Quantity")

        quantity_comparison = pd.DataFrame({
            "Quantity Type": ["Ordered", "Received", "Pending"],
            "Quantity": [
                total_ordered,
                total_received,
                total_pending,
            ],
        })

        fig = px.bar(
            quantity_comparison,
            x="Quantity Type",
            y="Quantity",
            color="Quantity Type",
            text_auto=True,
            title="Purchase Order Quantity Overview",
        )

        st.plotly_chart(
            style_chart(fig, 350),
            use_container_width=True,
        )

        # DELIVERY STATUS
        left, right = st.columns(2)

        with left:
            st.markdown("### 🚚 Delivery Status Distribution")

            status_data = (
                po["delivery_status"]
                .value_counts()
                .reset_index()
            )

            status_data.columns = ["Delivery Status", "Orders"]

            fig = px.pie(
                status_data,
                names="Delivery Status",
                values="Orders",
                hole=0.55,
                title="Orders by Delivery Status",
            )

            st.plotly_chart(
                style_chart(fig, 370),
                use_container_width=True,
            )

        with right:
            st.markdown("### ⏱️ Delivery Delay Analysis")

            if "delivery_delay_days" in po.columns:
                delay_data = po.groupby(
                    "delivery_status",
                    as_index=False,
                )["delivery_delay_days"].mean()

                fig = px.bar(
                    delay_data,
                    x="delivery_status",
                    y="delivery_delay_days",
                    text_auto=".2f",
                    title="Average Delay by Delivery Status",
                )

                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )

        # MONTHLY PO TREND
        st.markdown("### 📅 Monthly Purchase Order Trend")

        if "order_date" in po.columns:
            monthly_po = (
                po.dropna(subset=["order_date"])
                .assign(
                    Month=lambda x: x["order_date"].dt.to_period("M").astype(str)
                )
                .groupby("Month", as_index=False)
                .agg(
                    Total_Orders=(
                        "po_id", "nunique"
                    ) if "po_id" in po.columns
                    else ("order_date", "size"),
                    Ordered_Quantity=("ordered_quantity", "sum"),
                    Received_Quantity=("received_quantity", "sum"),
                )
            )

            if not monthly_po.empty:
                fig = px.line(
                    monthly_po,
                    x="Month",
                    y="Total_Orders",
                    markers=True,
                    title="Monthly Purchase Order Count",
                )

                st.plotly_chart(
                    style_chart(fig, 380),
                    use_container_width=True,
                )

        # STORE-WISE ANALYSIS
        left, right = st.columns(2)

        with left:
            st.markdown("### 🏬 Store-wise Purchase Orders")

            if "store_id" in po.columns:
                store_po = (
                    po.groupby("store_id", as_index=False)
                    .agg(
                        Total_Orders=(
                            "po_id", "nunique"
                        ) if "po_id" in po.columns
                        else ("store_id", "size"),
                        Ordered_Quantity=("ordered_quantity", "sum"),
                        Received_Quantity=("received_quantity", "sum"),
                    )
                )

                fig = px.bar(
                    store_po,
                    x="store_id",
                    y="Total_Orders",
                    color="store_id",
                    text_auto=True,
                    title="Purchase Orders by Store",
                )

                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )

        with right:
            st.markdown("### 🏢 Supplier-wise Purchase Orders")

            if "supplier_name" in po.columns:
                supplier_po = (
                    po.groupby("supplier_name", as_index=False)
                    .agg(
                        Total_Orders=(
                            "po_id", "nunique"
                        ) if "po_id" in po.columns
                        else ("supplier_name", "size"),
                        Ordered_Quantity=("ordered_quantity", "sum"),
                        Received_Quantity=("received_quantity", "sum"),
                    )
                    .sort_values("Total_Orders", ascending=False)
                )

                fig = px.bar(
                    supplier_po,
                    x="supplier_name",
                    y="Total_Orders",
                    color="supplier_name",
                    text_auto=True,
                    title="Purchase Orders by Supplier",
                )

                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )

        # DELAYED / PENDING ORDER DETAILS
        st.markdown("### ⚠️ Delayed and Pending Purchase Orders")

        delayed_data = po[
            po["delivery_status"].str.lower().str.contains(
                "delay|late|pending"
            )
        ].copy()

        if not delayed_data.empty:
            st.dataframe(
                delayed_data,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No delayed or pending orders identified by delivery status.")

        # COMPLETE PURCHASE ORDER TABLE
        with st.expander("📋 View Complete Purchase Order Details"):
            st.dataframe(
                po,
                use_container_width=True,
                hide_index=True,
            )

        # DOWNLOAD
        st.download_button(
            "📥 Download Purchase Order Analytics",
            data=po.to_csv(index=False).encode("utf-8"),
            file_name="retailpulse_purchase_order_analytics.csv",
            mime="text/csv",
        )

# ==========================================
# PART 13 - PAYMENT ANALYTICS DASHBOARD
# ==========================================

if selected_page == "Payment Analytics":

    show_section_heading(
        "💳 Payment Analytics",
        "Understand customer payment preferences, revenue contribution and transaction patterns."
    )

    # USE FILTERED SALES DATA
    if filtered_sales.empty:
        show_empty_message("No sales transactions available for the selected filters.")

    elif "payment_method" not in filtered_sales.columns:
        show_empty_message("Payment method column is missing from the sales dataset.")

    else:
        payment_data = filtered_sales.copy()

        payment_data["payment_method"] = (
            payment_data["payment_method"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        # PAYMENT SUMMARY
        payment_summary = (
            payment_data.groupby("payment_method", as_index=False)
            .agg(
                Transactions=(
                    "transaction_id", "nunique"
                ) if "transaction_id" in payment_data.columns
                else ("payment_method", "size"),
                Total_Revenue=("revenue", "sum")
                if "revenue" in payment_data.columns
                else ("payment_method", "size"),
                Total_Quantity=("quantity", "sum")
                if "quantity" in payment_data.columns
                else ("payment_method", "size"),
            )
        )

        payment_summary["Average_Transaction_Value"] = (
            payment_summary["Total_Revenue"]
            / payment_summary["Transactions"].replace(0, np.nan)
        ).fillna(0)

        total_transactions = payment_summary["Transactions"].sum()
        total_revenue = payment_summary["Total_Revenue"].sum()

        payment_summary["Revenue_Share_Pct"] = (
            payment_summary["Total_Revenue"]
            / total_revenue * 100
            if total_revenue > 0 else 0
        )

        # KPI CARDS
        k1, k2, k3, k4 = st.columns(4)

        with k1:
            show_metric(
                "Payment Methods",
                format_number(payment_summary["payment_method"].nunique())
            )

        with k2:
            show_metric(
                "Transactions",
                format_number(total_transactions)
            )

        with k3:
            show_metric(
                "Total Revenue",
                format_currency(total_revenue)
            )

        with k4:
            avg_value = (
                total_revenue / total_transactions
                if total_transactions > 0 else 0
            )
            show_metric(
                "Avg Transaction Value",
                format_currency(avg_value)
            )

        st.markdown("---")

        # REVENUE BY PAYMENT METHOD
        st.markdown("### 💰 Revenue by Payment Method")

        revenue_chart = payment_summary.sort_values(
            "Total_Revenue", ascending=False
        )

        fig = px.bar(
            revenue_chart,
            x="payment_method",
            y="Total_Revenue",
            color="payment_method",
            text_auto=".2s",
            title="Revenue Contribution by Payment Method",
        )

        st.plotly_chart(
            style_chart(fig, 400),
            use_container_width=True,
        )

        # PAYMENT SHARE AND TRANSACTION COUNT
        left, right = st.columns(2)

        with left:
            st.markdown("### 🥧 Revenue Share")

            fig = px.pie(
                payment_summary,
                names="payment_method",
                values="Total_Revenue",
                hole=0.55,
                title="Payment Method Revenue Share",
            )

            st.plotly_chart(
                style_chart(fig, 370),
                use_container_width=True,
            )

        with right:
            st.markdown("### 🔢 Transaction Count")

            fig = px.bar(
                payment_summary.sort_values("Transactions"),
                x="payment_method",
                y="Transactions",
                color="payment_method",
                text_auto=True,
                title="Transactions by Payment Method",
            )

            st.plotly_chart(
                style_chart(fig, 370),
                use_container_width=True,
            )

        # AVERAGE TRANSACTION VALUE
        st.markdown("### 📊 Average Transaction Value")

        fig = px.bar(
            payment_summary.sort_values(
                "Average_Transaction_Value", ascending=False
            ),
            x="payment_method",
            y="Average_Transaction_Value",
            color="payment_method",
            text_auto=".2f",
            title="Average Revenue per Transaction",
        )

        st.plotly_chart(
            style_chart(fig, 370),
            use_container_width=True,
        )

        # STORE-WISE PAYMENT ANALYSIS
        st.markdown("### 🏬 Store-wise Payment Analysis")

        if "store_id" in payment_data.columns:
            store_payment = (
                payment_data.groupby(
                    ["store_id", "payment_method"],
                    as_index=False,
                )
                .agg(
                    Transactions=(
                        "transaction_id", "nunique"
                    ) if "transaction_id" in payment_data.columns
                    else ("payment_method", "size"),
                    Revenue=("revenue", "sum")
                    if "revenue" in payment_data.columns
                    else ("payment_method", "size"),
                )
            )

            fig = px.bar(
                store_payment,
                x="store_id",
                y="Revenue",
                color="payment_method",
                barmode="group",
                title="Payment Revenue Across Stores",
            )

            st.plotly_chart(
                style_chart(fig, 420),
                use_container_width=True,
            )

        # DAILY PAYMENT TREND
        st.markdown("### 📅 Daily Payment Trend")

        if (
            "transaction_date" in payment_data.columns
            and "revenue" in payment_data.columns
        ):
            payment_data["transaction_date"] = pd.to_datetime(
                payment_data["transaction_date"],
                errors="coerce",
            )

            daily_payment = (
                payment_data.dropna(subset=["transaction_date"])
                .groupby(
                    [
                        payment_data["transaction_date"].dt.date,
                        "payment_method",
                    ],
                    as_index=False,
                )
                .agg(Revenue=("revenue", "sum"))
            )

            daily_payment = daily_payment.rename(
                columns={"transaction_date": "Date"}
            )

            if not daily_payment.empty:
                fig = px.line(
                    daily_payment,
                    x="Date",
                    y="Revenue",
                    color="payment_method",
                    markers=True,
                    title="Daily Revenue by Payment Method",
                )

                st.plotly_chart(
                    style_chart(fig, 420),
                    use_container_width=True,
                )

        # PAYMENT SUMMARY TABLE
        st.markdown("### 📋 Payment Performance Summary")

        st.dataframe(
            payment_summary.sort_values(
                "Total_Revenue", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )

        # TRANSACTION DETAILS
        with st.expander("🔍 View Payment Transaction Details"):
            st.dataframe(
                payment_data,
                use_container_width=True,
                hide_index=True,
            )

        # DOWNLOAD
        st.download_button(
            "📥 Download Payment Analytics",
            data=payment_summary.to_csv(index=False).encode("utf-8"),
            file_name="retailpulse_payment_analytics.csv",
            mime="text/csv",
        )

# ==========================================
# PART 14 - CATEGORY ANALYTICS DASHBOARD
# ==========================================

if selected_page == "Category Analytics":

    show_section_heading(
        "🛍️ Category Analytics",
        "Compare product categories by revenue, profit, sales volume and inventory."
    )

    # CHECK REQUIRED DATA
    if filtered_sales.empty or products.empty:
        show_empty_message(
            "Sales transactions or Products dataset is not available."
        )

    elif not {"product_id", "category"}.issubset(products.columns):
        show_empty_message(
            "Product ID or Category column is missing from Products dataset."
        )

    else:
        # MERGE SALES WITH PRODUCT CATEGORY
        category_sales = filtered_sales.merge(
            products[["product_id", "category"]].drop_duplicates("product_id"),
            on="product_id",
            how="left",
        )

        category_sales["category"] = (
            category_sales["category"]
            .fillna("Uncategorized")
            .astype(str)
        )

        # CATEGORY SALES SUMMARY
        category_summary = (
            category_sales.groupby("category", as_index=False)
            .agg(
                Transactions=(
                    "transaction_id", "nunique"
                ) if "transaction_id" in category_sales.columns
                else ("product_id", "size"),
                Total_Quantity=("quantity", "sum")
                if "quantity" in category_sales.columns
                else ("product_id", "size"),
                Total_Revenue=("revenue", "sum")
                if "revenue" in category_sales.columns
                else ("product_id", "size"),
                Total_Profit=("profit", "sum")
                if "profit" in category_sales.columns
                else ("product_id", "size"),
            )
        )

        category_summary["Average_Revenue_Per_Transaction"] = (
            category_summary["Total_Revenue"]
            / category_summary["Transactions"].replace(0, np.nan)
        ).fillna(0)

        total_category_revenue = category_summary["Total_Revenue"].sum()

        category_summary["Revenue_Share_Pct"] = (
            category_summary["Total_Revenue"]
            / total_category_revenue * 100
            if total_category_revenue > 0 else 0
        )

        # KPI METRICS
        total_categories = category_summary["category"].nunique()
        total_revenue = category_summary["Total_Revenue"].sum()
        total_profit = category_summary["Total_Profit"].sum()
        total_quantity = category_summary["Total_Quantity"].sum()

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            show_metric("Categories", format_number(total_categories))

        with k2:
            show_metric("Total Revenue", format_currency(total_revenue))

        with k3:
            show_metric("Total Profit", format_currency(total_profit))

        with k4:
            show_metric("Units Sold", format_number(total_quantity))

        st.markdown("---")

        # REVENUE BY CATEGORY
        st.markdown("### 💰 Category-wise Revenue")

        revenue_data = category_summary.sort_values(
            "Total_Revenue", ascending=False
        )

        fig = px.bar(
            revenue_data,
            x="category",
            y="Total_Revenue",
            color="category",
            text_auto=".2s",
            title="Revenue Generated by Each Category",
        )

        st.plotly_chart(
            style_chart(fig, 400),
            use_container_width=True,
        )

        # PROFIT AND QUANTITY
        left, right = st.columns(2)

        with left:
            st.markdown("### 📈 Category-wise Profit")

            fig = px.bar(
                category_summary.sort_values("Total_Profit"),
                x="category",
                y="Total_Profit",
                color="category",
                text_auto=".2s",
                title="Profit Contribution by Category",
            )

            st.plotly_chart(
                style_chart(fig, 370),
                use_container_width=True,
            )

        with right:
            st.markdown("### 📦 Category-wise Quantity")

            fig = px.bar(
                category_summary.sort_values("Total_Quantity"),
                x="category",
                y="Total_Quantity",
                color="category",
                text_auto=True,
                title="Units Sold by Category",
            )

            st.plotly_chart(
                style_chart(fig, 370),
                use_container_width=True,
            )

        # REVENUE SHARE
        st.markdown("### 🥧 Category Revenue Contribution")

        fig = px.pie(
            category_summary,
            names="category",
            values="Total_Revenue",
            hole=0.55,
            title="Revenue Share by Category",
        )

        st.plotly_chart(
            style_chart(fig, 400),
            use_container_width=True,
        )

        # INVENTORY ANALYSIS
        st.markdown("### 🏬 Category-wise Inventory Analysis")

        if (
            not inventory.empty
            and "category" in inventory.columns
            and "stock_on_hand" in inventory.columns
        ):
            inventory_data = inventory.copy()

            inventory_data["stock_on_hand"] = pd.to_numeric(
                inventory_data["stock_on_hand"],
                errors="coerce",
            ).fillna(0)

            inventory_data["inventory_cost"] = (
                pd.to_numeric(
                    inventory_data["inventory_cost"],
                    errors="coerce",
                ).fillna(0)
                if "inventory_cost" in inventory_data.columns
                else 0
            )

            inventory_summary = (
                inventory_data.groupby("category", as_index=False)
                .agg(
                    Total_Stock=("stock_on_hand", "sum"),
                    Inventory_Value=("inventory_cost", "sum"),
                )
            )

            left, right = st.columns(2)

            with left:
                fig = px.bar(
                    inventory_summary.sort_values("Total_Stock"),
                    x="category",
                    y="Total_Stock",
                    color="category",
                    text_auto=True,
                    title="Available Stock by Category",
                )

                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )

            with right:
                fig = px.bar(
                    inventory_summary.sort_values("Inventory_Value"),
                    x="category",
                    y="Inventory_Value",
                    color="category",
                    text_auto=".2s",
                    title="Inventory Value by Category",
                )

                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )

            with st.expander("📋 View Category Inventory Details"):
                st.dataframe(
                    inventory_summary,
                    use_container_width=True,
                    hide_index=True,
                )

        else:
            st.info(
                "Category inventory analysis is unavailable because the required inventory columns are missing."
            )

        # BEST AND LOWEST PERFORMERS
        st.markdown("### 🏆 Category Performance Insights")

        best_category = category_summary.loc[
            category_summary["Total_Revenue"].idxmax()
        ]

        lowest_category = category_summary.loc[
            category_summary["Total_Revenue"].idxmin()
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Highest Revenue Category",
                str(best_category["category"]),
                format_currency(best_category["Total_Revenue"]),
            )

        with col2:
            st.metric(
                "Lowest Revenue Category",
                str(lowest_category["category"]),
                format_currency(lowest_category["Total_Revenue"]),
            )

        # COMPLETE CATEGORY TABLE
        st.markdown("### 📋 Category Performance Summary")

        st.dataframe(
            category_summary.sort_values(
                "Total_Revenue", ascending=False
            ),
            use_container_width=True,
            hide_index=True,
        )

        # EXPORT
        st.download_button(
            "📥 Download Category Analytics",
            data=category_summary.to_csv(index=False).encode("utf-8"),
            file_name="retailpulse_category_analytics.csv",
            mime="text/csv",
        )

# PART 15 - DATA QUALITY DASHBOARD

# LOAD DATA QUALITY REPORT DIRECTLY

data_quality_report = data_quality.copy()

data_quality.columns = [
    str(c).strip().lower().replace(" ", "_")
    for c in data_quality.columns
]

if selected_page == "Data Quality":

    show_section_heading(
        "🔍 Data Quality Dashboard",
        "Check dataset completeness and reliability."
    )
    st.write("Data Quality Rows:", len(data_quality))
    st.write("Data Quality File:", find_dataset_file(DATASET_FILES["data_quality"]))
    
    if data_quality.empty:
        show_empty_message("Data Quality Report not found.")

    else:
        dq = data_quality.copy()

        k1, k2, k3 = st.columns(3)

        with k1:
            show_metric("Datasets", format_number(len(dq)))

        with k2:
            show_metric(
                "Missing Values",
                format_number(safe_sum(dq, "missing_values"))
            )

        with k3:
            show_metric(
                "Duplicate Rows",
                format_number(safe_sum(dq, "duplicate_rows"))
            )

        st.markdown("### Dataset Quality Overview")

        fig = px.bar(
            dq,
            x="dataset",
            y=["missing_values", "duplicate_rows"],
            barmode="group",
            title="Missing Values vs Duplicate Rows"
        )
        st.plotly_chart(style_chart(fig, 380), use_container_width=True)

        dq["Quality Status"] = np.where(
            (dq["missing_values"] == 0) &
            (dq["duplicate_rows"] == 0),
            "Clean", "Needs Review"
        )

        st.dataframe(dq, use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download Quality Report",
            dq.to_csv(index=False).encode("utf-8"),
            "data_quality_report.csv",
            "text/csv"
        )

# PART 16 - BUSINESS VALIDATION DASHBOARD

if selected_page == "Business Validation":

    show_section_heading(
        "✅ Business Validation",
        "Identify business rule violations and unusual records."
    )

    if business_validation.empty:
        show_empty_message("Business Validation Report not found.")

    else:
        bv = business_validation.copy()

        total_issues = safe_sum(bv, "issue_count")
        failed_checks = (bv["issue_count"] > 0).sum()

        c1, c2, c3 = st.columns(3)

        with c1:
            show_metric("Validation Checks", format_number(len(bv)))

        with c2:
            show_metric("Total Flagged Records", format_number(total_issues))

        with c3:
            show_metric("Checks with Issues", format_number(failed_checks))

        bv["Status"] = np.where(
            bv["issue_count"] > 0, "Review Required", "Passed"
        )

        st.markdown("### Validation Results")

        fig = px.bar(
            bv,
            x="validation_check",
            y="issue_count",
            color="Status",
            title="Issues by Validation Check"
        )
        st.plotly_chart(style_chart(fig, 380), use_container_width=True)

        st.dataframe(bv, use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download Validation Report",
            bv.to_csv(index=False).encode("utf-8"),
            "business_validation_report.csv",
            "text/csv"
        )

# PART 17 - DATA EXPLORER

if selected_page == "Data Explorer":

    show_section_heading(
        "🗂️ Data Explorer",
        "Explore and download project datasets."
    )

    datasets = {
        "Sales": sales,
        "Products": products,
        "Stores": stores,
        "Suppliers": suppliers,
        "Inventory": inventory,
        "Low Stock": low_stock,
        "Purchase Orders": purchase_orders,
        "Supplier Performance": supplier_performance,
        "Data Quality": data_quality,
        "Business Validation": business_validation,
    }

    dataset_name = st.selectbox(
        "Select Dataset",
        list(datasets.keys())
    )

    df = datasets[dataset_name].copy()

    if df.empty:
        show_empty_message("Selected dataset is empty.")

    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            show_metric("Rows", format_number(len(df)))

        with c2:
            show_metric("Columns", format_number(len(df.columns)))

        with c3:
            show_metric("Missing Values", format_number(df.isna().sum().sum()))

        search = st.text_input("🔎 Search in Dataset")

        if search:
            mask = df.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            df = df[mask]

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Download Dataset",
            df.to_csv(index=False).encode("utf-8"),
            f"{dataset_name.lower().replace(' ', '_')}.csv",
            "text/csv"
        )
