from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "Data" / "Raw" / "sales_transactions.csv"

OUTPUT_DIR = Path(__file__).resolve().parent

DETAIL_FILE = OUTPUT_DIR / "demand_forecast_validation.csv"
METRICS_FILE = OUTPUT_DIR / "demand_forecast_metrics.csv"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading sales data...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["transaction_date"])

df["quantity"] = pd.to_numeric(
    df["quantity"], errors="coerce"
).fillna(0).clip(lower=0)

df = df[
    ["date", "product_id", "store_id", "quantity"]
].dropna()

products = sorted(df["product_id"].unique())
stores = sorted(df["store_id"].unique())

daily_sales = (
    df.groupby(
        ["date", "product_id", "store_id"]
    )["quantity"]
    .sum()
)

start_date = df["date"].min()
end_date = df["date"].max()

all_dates = pd.date_range(
    start=start_date,
    end=end_date,
    freq="D"
)

# Use the final 30 historical days for testing
forecast_dates = all_dates[-30:]

train_end_date = forecast_dates[0] - pd.Timedelta(days=1)

print("Historical period:", start_date.date(), "to", end_date.date())
print("Training ends:", train_end_date.date())
print("Testing period:", forecast_dates[0].date(), "to", forecast_dates[-1].date())


# --------------------------------------------------
# 3. CREATE PRODUCT AND STORE CODES
# --------------------------------------------------

product_codes = {
    product: code
    for code, product in enumerate(products)
}

store_codes = {
    store: code
    for code, store in enumerate(stores)
}


# --------------------------------------------------
# 4. FEATURE CREATION
# --------------------------------------------------

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


def create_features(current_date, history, product_code, store_code):

    return {
        "product_code": product_code,
        "store_code": store_code,

        "day": current_date.day,
        "month": current_date.month,
        "day_of_week": current_date.day_of_week,
        "day_of_year": current_date.day_of_year,
        "is_weekend": int(current_date.day_of_week >= 5),

        "lag_1": history[-1],
        "lag_7": history[-7],
        "lag_14": history[-14],
        "lag_28": history[-28],

        "rolling_mean_7": np.mean(history[-7:]),
        "rolling_mean_28": np.mean(history[-28:])
    }


# --------------------------------------------------
# 5. PREPARE TRAINING DATA
# --------------------------------------------------

print("\nPreparing training data...")

training_records = []

for product in products:

    for store in stores:

        series = (
            daily_sales
            .xs((product, store), level=("product_id", "store_id"))
            .reindex(all_dates, fill_value=0)
            if (product, store) in daily_sales.index.droplevel("date")
            else pd.Series(0, index=all_dates)
        )

        values = series.to_numpy(dtype=float)

        product_code = product_codes[product]
        store_code = store_codes[store]

        for i in range(28, len(all_dates)):

            current_date = all_dates[i]

            if current_date > train_end_date:
                break

            features = create_features(
                current_date,
                values[:i],
                product_code,
                store_code
            )

            features["quantity"] = values[i]

            training_records.append(features)


training_data = pd.DataFrame(training_records)

if training_data.empty:
    raise ValueError("Training data is empty. Check historical data.")

X_train = training_data[feature_columns]
y_train = training_data["quantity"]

print("Training records:", len(training_data))


# --------------------------------------------------
# 6. TRAIN RANDOM FOREST
# --------------------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

model.fit(X_train, y_train)

print("Model training completed!")


# --------------------------------------------------
# 7. GENERATE 30-DAY VALIDATION PREDICTIONS
# --------------------------------------------------

print("\nGenerating validation predictions...")

results = []

for product in products:

    for store in stores:

        series = (
            daily_sales
            .xs((product, store), level=("product_id", "store_id"))
            .reindex(all_dates, fill_value=0)
            if (product, store) in daily_sales.index.droplevel("date")
            else pd.Series(0, index=all_dates)
        )

        values = series.to_numpy(dtype=float)

        history = list(
            values[:len(all_dates) - 30]
        )

        product_code = product_codes[product]
        store_code = store_codes[store]

        for forecast_date in forecast_dates:

            features = create_features(
                forecast_date,
                history,
                product_code,
                store_code
            )

            input_data = pd.DataFrame(
                [features],
                columns=feature_columns
            )

            predicted = max(
                0,
                model.predict(input_data)[0]
            )

            # Seasonal naive baseline: repeat demand from 7 days ago
            baseline = history[-7]

            actual_index = all_dates.get_loc(forecast_date)
            actual = values[actual_index]

            results.append({
                "date": forecast_date,
                "product_id": product,
                "store_id": store,
                "actual_quantity": actual,
                "predicted_quantity": predicted,
                "baseline_quantity": baseline
            })

            # Recursive forecasting
            history.append(predicted)


# --------------------------------------------------
# 8. CALCULATE ACCURACY METRICS
# --------------------------------------------------

result_df = pd.DataFrame(results)

actual = result_df["actual_quantity"]
predicted = result_df["predicted_quantity"]
baseline = result_df["baseline_quantity"]


def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    total_actual = actual.sum()

    wape = (
        np.abs(actual - predicted).sum() / total_actual * 100
        if total_actual > 0 else np.nan
    )

    return mae, rmse, wape


model_mae, model_rmse, model_wape = calculate_metrics(
    actual, predicted
)

base_mae, base_rmse, base_wape = calculate_metrics(
    actual, baseline
)


# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

metrics_df = pd.DataFrame([
    {
        "Model": "Random Forest",
        "MAE": model_mae,
        "RMSE": model_rmse,
        "WAPE_Percent": model_wape
    },
    {
        "Model": "7-Day Seasonal Baseline",
        "MAE": base_mae,
        "RMSE": base_rmse,
        "WAPE_Percent": base_wape
    }
])

result_df.to_csv(DETAIL_FILE, index=False)
metrics_df.to_csv(METRICS_FILE, index=False)


# --------------------------------------------------
# 10. DISPLAY RESULTS
# --------------------------------------------------

print("\n========== MODEL ACCURACY RESULTS ==========")

print(metrics_df.to_string(index=False))

print("\nValidation records:", len(result_df))

print("Expected records:", len(products) * len(stores) * 30)

print("\nDetailed predictions saved:", DETAIL_FILE)

print("Accuracy metrics saved:", METRICS_FILE)

print("\nValidation completed successfully!")