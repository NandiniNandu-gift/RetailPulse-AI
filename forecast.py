
from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# RETAILPULSE AI - INTERMITTENT DEMAND FORECASTING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SALES_FILE = PROJECT_ROOT / "Data" / "Raw" / "sales_transactions.csv"
PRODUCTS_FILE = PROJECT_ROOT / "Data" / "Raw" / "products.csv"
STORES_FILE = PROJECT_ROOT / "Data" / "Raw" / "stores.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_START = pd.Timestamp("2025-10-02")
VALIDATION_END = pd.Timestamp("2025-10-31")

TEST_START = pd.Timestamp("2025-12-02")
TEST_END = pd.Timestamp("2025-12-31")

VALIDATION_TRAIN_END = pd.Timestamp("2025-10-01")
FINAL_TRAIN_END = pd.Timestamp("2025-12-01")

ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.40]
BETAS = [0.05, 0.10, 0.20, 0.30, 0.40]


# ============================================================
# 1. LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    print("\nLoading sales data...")

    if not SALES_FILE.exists():
        raise FileNotFoundError(
            f"Sales file not found: {SALES_FILE}"
        )

    sales = pd.read_csv(SALES_FILE)

    sales.columns = (
        sales.columns.astype(str).str.strip().str.lower()
    )

    required = [
        "transaction_date",
        "product_id",
        "store_id",
        "quantity"
    ]

    missing = [c for c in required if c not in sales.columns]

    if missing:
        raise ValueError(
            f"Missing required sales columns: {missing}"
        )

    sales["transaction_date"] = pd.to_datetime(
        sales["transaction_date"],
        errors="coerce"
    ).dt.normalize()

    sales["product_id"] = (
        sales["product_id"].astype(str).str.strip()
    )

    sales["store_id"] = (
        sales["store_id"].astype(str).str.strip()
    )

    sales["quantity"] = pd.to_numeric(
        sales["quantity"],
        errors="coerce"
    )

    sales = sales.dropna(
        subset=[
            "transaction_date",
            "product_id",
            "store_id",
            "quantity"
        ]
    )

    sales = sales[sales["quantity"] >= 0].copy()

    if sales.empty:
        raise ValueError("No valid sales records found.")

    # Use dimension tables when available.
    if PRODUCTS_FILE.exists():
        products_df = pd.read_csv(PRODUCTS_FILE)
        products_df.columns = (
            products_df.columns.astype(str).str.strip().str.lower()
        )

        if "product_id" in products_df.columns:
            products = (
                products_df["product_id"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
        else:
            products = sorted(sales["product_id"].unique())
    else:
        products = sorted(sales["product_id"].unique())

    if STORES_FILE.exists():
        stores_df = pd.read_csv(STORES_FILE)
        stores_df.columns = (
            stores_df.columns.astype(str).str.strip().str.lower()
        )

        if "store_id" in stores_df.columns:
            stores = (
                stores_df["store_id"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
        else:
            stores = sorted(sales["store_id"].unique())
    else:
        stores = sorted(sales["store_id"].unique())

    # Include all valid combinations.
    products = sorted(set(products) | set(sales["product_id"].unique()))
    stores = sorted(set(stores) | set(sales["store_id"].unique()))

    print("Sales records:", len(sales))
    print("Products:", len(products))
    print("Stores:", len(stores))

    return sales, products, stores


# ============================================================
# 2. CREATE COMPLETE DAILY PRODUCT-STORE PANEL
# ============================================================

def create_daily_panel(sales, products, stores):

    print("\nCreating daily demand panel...")

    start_date = sales["transaction_date"].min()
    end_date = max(
        sales["transaction_date"].max(),
        TEST_END
    )

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    combinations = pd.MultiIndex.from_product(
        [products, stores],
        names=["product_id", "store_id"]
    )

    daily_sales = (
        sales.groupby(
            ["product_id", "store_id", "transaction_date"]
        )["quantity"]
        .sum()
    )

    panel = (
        daily_sales.unstack("transaction_date")
        .reindex(
            index=combinations,
            columns=dates,
            fill_value=0
        )
        .fillna(0)
    )

    panel.index.names = ["product_id", "store_id"]
    panel.columns.name = "date"

    panel = panel.astype(float)

    if VALIDATION_START not in panel.columns:
        raise ValueError("October validation dates are missing.")

    if TEST_START not in panel.columns:
        raise ValueError("December test dates are missing.")

    print("Daily records:", panel.size)
    print("Product-store combinations:", len(panel))
    print("Date range:", dates.min().date(), "to", dates.max().date())

    zero_percentage = (
        (panel.to_numpy() == 0).sum()
        / panel.size
        * 100
    )

    print(f"Zero-demand percentage: {zero_percentage:.2f}%")

    return panel


# ============================================================
# 3. CROSTON METHOD
# ============================================================

def croston_forecast(history, alpha, horizon, sba=False):

    y = np.asarray(history, dtype=float)

    positive_positions = np.flatnonzero(y > 0)

    if len(positive_positions) == 0:
        return np.zeros(horizon)

    first = positive_positions[0]

    demand_size = y[first]
    interval = first + 1

    gap = 1

    for t in range(first + 1, len(y)):

        if y[t] > 0:

            demand_size = (
                alpha * y[t]
                + (1 - alpha) * demand_size
            )

            interval = (
                alpha * gap
                + (1 - alpha) * interval
            )

            gap = 1

        else:
            gap += 1

    if interval <= 0:
        forecast_value = 0.0
    else:
        forecast_value = demand_size / interval

    if sba:
        forecast_value *= (1 - alpha / 2)

    forecast_value = max(0.0, forecast_value)

    return np.full(horizon, forecast_value)


# ============================================================
# 4. TEUNTER-SYNTETOS-BABAI METHOD
# ============================================================

def tsb_forecast(history, alpha, beta, horizon):

    y = np.asarray(history, dtype=float)

    positive = y[y > 0]

    if len(positive) == 0:
        return np.zeros(horizon)

    probability = np.mean(y > 0)
    demand_size = np.mean(positive)

    for value in y:

        occurrence = 1.0 if value > 0 else 0.0

        probability = (
            beta * occurrence
            + (1 - beta) * probability
        )

        if value > 0:
            demand_size = (
                alpha * value
                + (1 - alpha) * demand_size
            )

    forecast_value = max(
        0.0,
        probability * demand_size
    )

    return np.full(horizon, forecast_value)


# ============================================================
# 5. SEASONAL NAIVE METHOD
# ============================================================

def seasonal_naive_forecast(history, horizon):

    values = list(np.asarray(history, dtype=float))

    predictions = []

    for _ in range(horizon):

        if len(values) >= 7:
            prediction = values[-7]
        else:
            prediction = np.mean(values) if values else 0.0

        prediction = max(0.0, float(prediction))

        predictions.append(prediction)
        values.append(prediction)

    return np.asarray(predictions)


# ============================================================
# 6. OTHER BASELINES
# ============================================================

def moving_average_forecast(history, horizon):

    y = np.asarray(history, dtype=float)

    value = np.mean(y[-28:]) if len(y) else 0.0

    return np.full(horizon, max(0.0, value))


def weekday_average_forecast(
    history,
    history_end_date,
    horizon
):

    y = np.asarray(history, dtype=float)

    lookback = min(56, len(y))
    recent = y[-lookback:]

    recent_dates = pd.date_range(
        end=history_end_date,
        periods=lookback,
        freq="D"
    )

    weekday_values = {}

    for weekday in range(7):

        values = recent[
            np.array(recent_dates.dayofweek) == weekday
        ]

        weekday_values[weekday] = (
            float(np.mean(values))
            if len(values) > 0
            else float(np.mean(recent))
        )

    predictions = []

    for day in range(1, horizon + 1):

        future_date = (
            history_end_date
            + pd.Timedelta(days=day)
        )

        predictions.append(
            max(0.0, weekday_values[future_date.dayofweek])
        )

    return np.asarray(predictions)


# ============================================================
# 7. MODEL CANDIDATES
# ============================================================

def create_candidates():

    candidates = {}

    candidates["Seasonal_Naive_7Day"] = {
        "method": "seasonal"
    }

    candidates["Moving_Average_28Day"] = {
        "method": "moving_average"
    }

    candidates["Weekday_Average_8Week"] = {
        "method": "weekday_average"
    }

    candidates["All_Zero_Baseline"] = {
        "method": "zero"
    }

    for alpha in ALPHAS:

        label = f"Croston_alpha_{alpha:.2f}"

        candidates[label] = {
            "method": "croston",
            "alpha": alpha
        }

        label = f"Croston_SBA_alpha_{alpha:.2f}"

        candidates[label] = {
            "method": "sba",
            "alpha": alpha
        }

    for alpha, beta in itertools.product(ALPHAS, BETAS):

        label = f"TSB_alpha_{alpha:.2f}_beta_{beta:.2f}"

        candidates[label] = {
            "method": "tsb",
            "alpha": alpha,
            "beta": beta
        }

    return candidates


# ============================================================
# 8. COMMON FORECAST FUNCTION
# ============================================================

def forecast_one(
    history,
    config,
    horizon,
    history_end_date
):

    method = config["method"]

    if method == "croston":

        return croston_forecast(
            history,
            config["alpha"],
            horizon,
            sba=False
        )

    if method == "sba":

        return croston_forecast(
            history,
            config["alpha"],
            horizon,
            sba=True
        )

    if method == "tsb":

        return tsb_forecast(
            history,
            config["alpha"],
            config["beta"],
            horizon
        )

    if method == "seasonal":

        return seasonal_naive_forecast(
            history,
            horizon
        )

    if method == "moving_average":

        return moving_average_forecast(
            history,
            horizon
        )

    if method == "weekday_average":

        return weekday_average_forecast(
            history,
            history_end_date,
            horizon
        )

    if method == "zero":

        return np.zeros(horizon)

    raise ValueError(f"Unknown forecasting method: {method}")


# ============================================================
# 9. EVALUATION METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()

    predicted = np.maximum(predicted, 0)

    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    actual_total = np.sum(actual)
    forecast_total = np.sum(predicted)

    if actual_total > 0:

        wape = (
            np.sum(np.abs(errors))
            / actual_total
            * 100
        )

        bias = (
            (forecast_total - actual_total)
            / actual_total
            * 100
        )

    else:
        wape = np.nan
        bias = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
        "Forecast_Bias_percent": bias,
        "Actual_total": actual_total,
        "Forecast_total": forecast_total
    }


# ============================================================
# 10. GENERATE FORECASTS FOR ALL PRODUCT-STORE SERIES
# ============================================================

def generate_model_forecasts(
    panel,
    candidates,
    train_end,
    forecast_start,
    forecast_end
):

    horizon_dates = pd.date_range(
        forecast_start,
        forecast_end,
        freq="D"
    )

    horizon = len(horizon_dates)

    history = panel.loc[:, :train_end]

    actual = panel.loc[:, forecast_start:forecast_end].to_numpy(
        dtype=float
    )

    if actual.shape[1] != horizon:
        raise ValueError("Forecast actual dates do not match horizon.")

    history_end_date = train_end

    forecasts = {}

    print(
        f"\nTraining through {train_end.date()} "
        f"and forecasting {horizon} days..."
    )

    for model_name, config in candidates.items():

        print("Evaluating:", model_name)

        model_predictions = []

        for _, row in history.iterrows():

            values = row.to_numpy(dtype=float)

            prediction = forecast_one(
                values,
                config,
                horizon,
                history_end_date
            )

            model_predictions.append(prediction)

        forecasts[model_name] = np.maximum(
            np.asarray(model_predictions, dtype=float),
            0
        )

    return forecasts, actual, horizon_dates


# ============================================================
# 11. MODEL METRICS TABLE
# ============================================================

def build_metrics_table(forecasts, actual, period_name):

    records = []

    for model_name, predicted in forecasts.items():

        metrics = calculate_metrics(
            actual,
            predicted
        )

        records.append({
            "Period": period_name,
            "Model": model_name,
            **metrics
        })

    result = pd.DataFrame(records)

    result = result.sort_values(
        ["WAPE_percent", "MAE"],
        ascending=[True, True],
        na_position="last"
    ).reset_index(drop=True)

    result.insert(
        0,
        "Rank",
        np.arange(1, len(result) + 1)
    )

    return result


# ============================================================
# 12. PRODUCT-STORE LEVEL REPORT
# ============================================================

def create_product_store_report(
    panel,
    selected_forecast,
    actual,
    selected_model,
    forecast_dates
):

    records = []

    combinations = list(panel.index)

    for i, (product, store) in enumerate(combinations):

        actual_values = actual[i]
        forecast_values = selected_forecast[i]

        metrics = calculate_metrics(
            actual_values,
            forecast_values
        )

        records.append({
            "product_id": product,
            "store_id": store,
            "selected_model": selected_model,
            "actual_total": metrics["Actual_total"],
            "forecast_total": metrics["Forecast_total"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE_percent": metrics["WAPE_percent"],
            "Forecast_Bias_percent": metrics["Forecast_Bias_percent"]
        })

    return pd.DataFrame(records)


# ============================================================
# 13. SAVE DAILY FORECAST OUTPUT
# ============================================================

def create_daily_output(
    panel,
    selected_forecast,
    actual,
    selected_model,
    forecast_dates
):

    records = []

    for i, (product, store) in enumerate(panel.index):

        for j, forecast_date in enumerate(forecast_dates):

            records.append({
                "date": forecast_date,
                "product_id": product,
                "store_id": store,
                "actual_quantity": actual[i, j],
                "forecast_quantity": selected_forecast[i, j],
                "absolute_error": abs(
                    actual[i, j] - selected_forecast[i, j]
                ),
                "selected_model": selected_model
            })

    return pd.DataFrame(records)


# ============================================================
# 14. DAILY AGGREGATE SUMMARY
# ============================================================

def create_daily_summary(
    daily_output
):

    summary = (
        daily_output.groupby("date")
        .agg(
            actual_quantity=("actual_quantity", "sum"),
            forecast_quantity=("forecast_quantity", "sum")
        )
        .reset_index()
    )

    summary["absolute_error"] = abs(
        summary["actual_quantity"]
        - summary["forecast_quantity"]
    )

    return summary


# ============================================================
# 15. CHART GENERATION
# ============================================================

def create_charts(
    validation_metrics,
    december_summary,
    selected_model
):

    try:

        import matplotlib.pyplot as plt

        # Model comparison chart
        comparison = validation_metrics.sort_values(
            "WAPE_percent",
            ascending=True
        ).head(12)

        plt.figure(figsize=(12, 7))

        plt.barh(
            comparison["Model"],
            comparison["WAPE_percent"]
        )

        plt.xlabel("WAPE (%)")
        plt.ylabel("Forecasting Model")
        plt.title("October Validation - Model Comparison")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "validation_model_comparison.png",
            dpi=200
        )

        plt.close()

        # Actual vs forecast chart
        plt.figure(figsize=(13, 6))

        plt.plot(
            december_summary["date"],
            december_summary["actual_quantity"],
            label="Actual Demand"
        )

        plt.plot(
            december_summary["date"],
            december_summary["forecast_quantity"],
            label="Forecast Demand"
        )

        plt.xlabel("Date")
        plt.ylabel("Quantity")
        plt.title(
            f"December 2025 Actual vs Forecast - {selected_model}"
        )

        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "december_actual_vs_forecast.png",
            dpi=200
        )

        plt.close()

        print("\nCharts saved successfully.")

    except ImportError:

        print(
            "\nMatplotlib not installed. "
            "CSV reports were still generated."
        )


# ============================================================
# 16. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("RETAILPULSE AI - DEMAND FORECASTING")
    print("=" * 65)

    sales, products, stores = load_data()

    panel = create_daily_panel(
        sales,
        products,
        stores
    )

    candidates = create_candidates()

    # --------------------------------------------------------
    # OCTOBER VALIDATION
    # Train: up to October 1
    # Forecast: October 2 to October 31
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("OCTOBER 2025 VALIDATION")
    print("=" * 65)

    validation_forecasts, validation_actual, validation_dates = (
        generate_model_forecasts(
            panel,
            candidates,
            VALIDATION_TRAIN_END,
            VALIDATION_START,
            VALIDATION_END
        )
    )

    validation_metrics = build_metrics_table(
        validation_forecasts,
        validation_actual,
        "October_Validation"
    )

    validation_metrics.to_csv(
        OUTPUT_DIR / "intermittent_model_validation.csv",
        index=False
    )

    # Select best NON-ZERO model based only on October validation.
    nonzero_validation = validation_metrics[
        validation_metrics["Model"] != "All_Zero_Baseline"
    ].copy()

    nonzero_validation = nonzero_validation.dropna(
        subset=["WAPE_percent", "MAE"]
    )

    if nonzero_validation.empty:
        raise ValueError("No valid non-zero model available.")

    selected_model = nonzero_validation.iloc[0]["Model"]

    print("\nOCTOBER VALIDATION RESULTS")
    print(
        validation_metrics[
            [
                "Rank",
                "Model",
                "MAE",
                "RMSE",
                "WAPE_percent",
                "Forecast_Bias_percent"
            ]
        ].to_string(index=False)
    )

    print("\nSelected non-zero model:", selected_model)

    # --------------------------------------------------------
    # DECEMBER FINAL TEST
    # Train: up to December 1
    # Forecast: December 2 to December 31
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("DECEMBER 2025 FINAL TEST")
    print("=" * 65)

    december_forecasts, december_actual, december_dates = (
        generate_model_forecasts(
            panel,
            candidates,
            FINAL_TRAIN_END,
            TEST_START,
            TEST_END
        )
    )

    december_metrics = build_metrics_table(
        december_forecasts,
        december_actual,
        "December_Final_Test"
    )

    december_metrics.to_csv(
        OUTPUT_DIR / "intermittent_model_december_test.csv",
        index=False
    )

    print("\nDECEMBER MODEL COMPARISON")
    print(
        december_metrics[
            [
                "Rank",
                "Model",
                "MAE",
                "RMSE",
                "WAPE_percent",
                "Forecast_Bias_percent",
                "Actual_total",
                "Forecast_total"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # SELECTED MODEL OUTPUT
    # --------------------------------------------------------

    selected_forecast = december_forecasts[selected_model]

    daily_output = create_daily_output(
        panel,
        selected_forecast,
        december_actual,
        selected_model,
        december_dates
    )

    daily_output.to_csv(
        OUTPUT_DIR / "december_2025_demand_forecast.csv",
        index=False
    )

    product_store_report = create_product_store_report(
        panel,
        selected_forecast,
        december_actual,
        selected_model,
        december_dates
    )

    product_store_report.to_csv(
        OUTPUT_DIR / "december_product_store_performance.csv",
        index=False
    )

    daily_summary = create_daily_summary(
        daily_output
    )

    daily_summary.to_csv(
        OUTPUT_DIR / "december_daily_demand_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # COMBINED METRICS
    # --------------------------------------------------------

    combined_metrics = pd.concat(
        [
            validation_metrics,
            december_metrics
        ],
        ignore_index=True
    )

    combined_metrics.to_csv(
        OUTPUT_DIR / "final_forecasting_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY REPORT
    # --------------------------------------------------------

    selected_december_metrics = december_metrics[
        december_metrics["Model"] == selected_model
    ].iloc[0]

    overall_best_december = december_metrics.iloc[0]

    summary_lines = [
        "RETAILPULSE AI - DEMAND FORECASTING REPORT",
        "=" * 55,
        "",
        f"Validation period: {VALIDATION_START.date()} to {VALIDATION_END.date()}",
        f"Validation training end: {VALIDATION_TRAIN_END.date()}",
        f"Test period: {TEST_START.date()} to {TEST_END.date()}",
        f"Final training end: {FINAL_TRAIN_END.date()}",
        "",
        f"Selected non-zero model: {selected_model}",
        "",
        "Selected model - December metrics:",
        f"MAE: {selected_december_metrics['MAE']:.4f}",
        f"RMSE: {selected_december_metrics['RMSE']:.4f}",
        f"WAPE: {selected_december_metrics['WAPE_percent']:.2f}%",
        f"Forecast Bias: {selected_december_metrics['Forecast_Bias_percent']:.2f}%",
        f"Actual total: {selected_december_metrics['Actual_total']:.2f}",
        f"Forecast total: {selected_december_metrics['Forecast_total']:.2f}",
        "",
        "Best overall December model:",
        f"{overall_best_december['Model']}",
        f"WAPE: {overall_best_december['WAPE_percent']:.2f}%",
        "",
        "Note:",
        "The selected non-zero model was chosen using October validation.",
        "December is kept as a separate final test period.",
        "The all-zero baseline is included only as a benchmark.",
        "No model accuracy is guaranteed.",
        "",
        "Output folder:",
        str(OUTPUT_DIR)
    ]

    summary_text = "\n".join(summary_lines)

    (OUTPUT_DIR / "forecast_summary.txt").write_text(
        summary_text,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    create_charts(
        validation_metrics,
        daily_summary,
        selected_model
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FORECASTING COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nSelected model:", selected_model)

    print("\nSelected model December metrics:")

    print(
        selected_december_metrics[
            [
                "MAE",
                "RMSE",
                "WAPE_percent",
                "Forecast_Bias_percent",
                "Actual_total",
                "Forecast_total"
            ]
        ].to_string()
    )

    print("\nAll files saved inside:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    for file in sorted(OUTPUT_DIR.iterdir()):
        if file.is_file():
            print("-", file.name)


if __name__ == "__main__":
    main()
