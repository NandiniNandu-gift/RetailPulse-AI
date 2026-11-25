
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "Data" / "Raw"
OUTPUT_DIR = PROJECT_ROOT / "Machine_Learning" / "Demand_Forecasting"

INPUT_FILE = RAW_DIR / "sales_transactions.csv"
OUTPUT_FILE = OUTPUT_DIR / "demand_forecast_30_days.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD SALES DATA
# --------------------------------------------------

print("Loading sales data...")

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "transaction_date",
    "product_id",
    "store_id",
    "quantity"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in sales_transactions.csv: {missing_columns}"
    )

# Use the actual date column from the CSV
df["date"] = pd.to_datetime(df["transaction_date"])

df = df[
    ["date", "product_id", "store_id", "quantity"]
].copy()

df["quantity"] = pd.to_numeric(
    df["quantity"], errors="coerce"
).fillna(0)

df = df.dropna(subset=["date", "product_id", "store_id"])

df["quantity"] = df["quantity"].clip(lower=0)

print(f"Sales records loaded: {len(df)}")


# --------------------------------------------------
# 3. PREPARE PRODUCT AND STORE CODES
# --------------------------------------------------

products = sorted(df["product_id"].unique())
stores = sorted(df["store_id"].unique())

product_codes = {
    product: code
    for code, product in enumerate(products)
}

store_codes = {
    store: code
    for code, store in enumerate(stores)
}

start_date = df["date"].min()
end_date = df["date"].max()

all_dates = pd.date_range(
    start=start_date,
    end=end_date,
    freq="D"
)

print(f"Products: {len(products)}")
print(f"Stores: {len(stores)}")
print(f"Historical dates: {len(all_dates)}")


# --------------------------------------------------
# 4. CREATE DAILY PRODUCT-STORE DEMAND
# --------------------------------------------------

print("Preparing daily demand data...")

daily_sales = (
    df.groupby(
        ["date", "product_id", "store_id"],
        as_index=False
    )["quantity"]
    .sum()
)

training_records = []

# Keep complete daily history for every product-store pair
for product in products:

    for store in stores:

        group = daily_sales[
            (daily_sales["product_id"] == product) &
            (daily_sales["store_id"] == store)
        ]

        daily_series = (
            group.set_index("date")["quantity"]
            .reindex(all_dates, fill_value=0)
            .astype(float)
        )

        values = daily_series.to_numpy()

        product_code = product_codes[product]
        store_code = store_codes[store]

        # First 28 days are needed for lag features
        for i in range(28, len(values)):

            current_date = all_dates[i]

            features = {
                "product_code": product_code,
                "store_code": store_code,

                "day": current_date.day,
                "month": current_date.month,
                "day_of_week": current_date.dayofweek,
                "day_of_year": current_date.dayofyear,
                "is_weekend": int(current_date.dayofweek >= 5),

                "lag_1": values[i - 1],
                "lag_7": values[i - 7],
                "lag_14": values[i - 14],
                "lag_28": values[i - 28],

                "rolling_mean_7": np.mean(values[i - 7:i]),
                "rolling_mean_28": np.mean(values[i - 28:i])
            }

            features["quantity"] = values[i]

            training_records.append(features)


training_data = pd.DataFrame(training_records)

if training_data.empty:
    raise ValueError(
        "Not enough historical data to create training features."
    )

feature_columns = [
    "product_code",
    "store_code",
    "day",
    "month",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]

X = training_data[feature_columns]
y = training_data["quantity"]

print(f"Training records: {len(training_data)}")


# --------------------------------------------------
# 5. TRAIN RANDOM FOREST MODEL
# --------------------------------------------------

print("Training Demand Forecasting Model...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

model.fit(X, y)

print("Model training completed!")


# --------------------------------------------------
# 6. GENERATE NEXT 30 DAYS FORECAST
# --------------------------------------------------

print("Generating 30-day forecast...")

forecast_dates = pd.date_range(
    start=end_date + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

forecast_records = []

for product in products:

    for store in stores:

        group = daily_sales[
            (daily_sales["product_id"] == product) &
            (daily_sales["store_id"] == store)
        ]

        daily_series = (
            group.set_index("date")["quantity"]
            .reindex(all_dates, fill_value=0)
            .astype(float)
        )

        history = list(daily_series.to_numpy())

        product_code = product_codes[product]
        store_code = store_codes[store]

        for forecast_date in forecast_dates:

            features = {
                "product_code": product_code,
                "store_code": store_code,

                "day": forecast_date.day,
                "month": forecast_date.month,
                "day_of_week": forecast_date.dayofweek,
                "day_of_year": forecast_date.dayofyear,
                "is_weekend": int(forecast_date.dayofweek >= 5),

                "lag_1": history[-1],
                "lag_7": history[-7],
                "lag_14": history[-14],
                "lag_28": history[-28],

                "rolling_mean_7": np.mean(history[-7:]),
                "rolling_mean_28": np.mean(history[-28:])
            }

            input_data = pd.DataFrame(
                [features],
                columns=feature_columns
            )

            predicted_quantity = model.predict(input_data)[0]

            predicted_quantity = max(0, predicted_quantity)

            forecast_records.append({
                "forecast_date": forecast_date,
                "product_id": product,
                "store_id": store,
                "predicted_quantity": round(
                    predicted_quantity, 2
                )
            })

            # Use prediction for the next day's forecast
            history.append(predicted_quantity)


# --------------------------------------------------
# 7. SAVE FORECAST RESULTS
# --------------------------------------------------

forecast_df = pd.DataFrame(forecast_records)

forecast_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nForecast completed successfully!")

print(f"Total forecast records: {len(forecast_df)}")

print(f"Forecast period: {forecast_dates[0].date()} to {forecast_dates[-1].date()}")

print(f"Output saved at: {OUTPUT_FILE}")

print("\nForecast preview:")
print(forecast_df.head(10).to_string(index=False))
