import os
import sys
import argparse
import pandas as pd

try:
    from .prophet.prophet_utils import prepare_prophet_df
    from .prophet.train_prophet_model import train_and_forecast
except Exception:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from backend.model.VM.prophet.prophet_utils import prepare_prophet_df
    from backend.model.VM.prophet.train_prophet_model import train_and_forecast

FORECAST_METRICS = {
    "cpu_utilization_mean": "cpu_utilization_forecast",
    "cost_usd_sum": "cost_usd_forecast",
    "disk_total_bytes_sum": "disk_total_bytes_forecast",
    "network_total_bytes_sum": "network_total_bytes_forecast"
}

def forecast_vm_data(train_df, forecast_days=5):
    all_forecasts = []

    for instance_id in train_df["instance_id"].unique():
        instance_df = train_df[train_df["instance_id"] == instance_id]

        instance_forecast = None

        for metric, out_col in FORECAST_METRICS.items():
            if metric not in instance_df.columns:
                continue

            ts_df = prepare_prophet_df(
                instance_df,
                date_col="date",
                value_col=metric
            )

            if len(ts_df) < 10:
                continue

            forecast = train_and_forecast(
                ts_df,
                periods=forecast_days
            )

            forecast = forecast.tail(forecast_days)

            forecast = forecast[["ds", "yhat"]]
            forecast.rename(columns={
                "ds": "date",
                "yhat": out_col
            }, inplace=True)

            if instance_forecast is None:
                instance_forecast = forecast
            else:
                instance_forecast = instance_forecast.merge(
                    forecast,
                    on="date",
                    how="left"
                )

        if instance_forecast is not None:
            instance_forecast["instance_id"] = instance_id
            all_forecasts.append(instance_forecast)

    if not all_forecasts:
        return pd.DataFrame(columns=["date", "instance_id"] + list(FORECAST_METRICS.values()))

    final_df = pd.concat(all_forecasts, ignore_index=True)
    # Normalize types
    final_df["date"] = pd.to_datetime(final_df["date"])
    final_df = final_df.sort_values(["instance_id", "date"]).reset_index(drop=True)

    return final_df

def load_csv_if_exists(path):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate VM forecasts per-instance.")
    parser.add_argument('--train', help='Path to train CSV (defaults to backend/model/VM/data/train/train_vm_data.csv)', default=None)
    parser.add_argument('--test', help='Path to test CSV to include in training (defaults to backend/model/VM/data/test/test_vm_data.csv)', default=None)
    parser.add_argument('--include-test', help='If set, include the test CSV in the training pool (so model sees train+test)', action='store_true')
    parser.add_argument('--forecast-days', help='Number of days to forecast (overrides automatic calculation)', type=int, default=None)
    parser.add_argument('--year', help='Year for validation end (defaults to latest year in combined data)', type=int, default=None)
    parser.add_argument('--out', help='Output forecasts CSV path (defaults to backend/model/VM/data/forecast/vm_forecasts.csv)', default=None)
    args = parser.parse_args(argv)

    base = os.path.dirname(__file__)
    default_train = os.path.join(base, 'data', 'train', 'train_vm_data.csv')
    default_test = os.path.join(base, 'data', 'test', 'test_vm_data.csv')
    default_out = os.path.join(base, 'data', 'forecast', 'vm_forecasts.csv')

    train_path = args.train if args.train is not None else default_train
    test_path = args.test if args.test is not None else default_test
    out_path = args.out if args.out is not None else default_out

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train CSV not found: {train_path}")

    train_df = pd.read_csv(train_path)
    train_df['date'] = pd.to_datetime(train_df['date'])
    train_df = train_df.sort_values('date')

    combined_df = train_df.copy()

    if args.include_test:
        test_df = load_csv_if_exists(test_path)
        if test_df is None:
            print(f"Warning: include-test requested but test file not found at {test_path}. Continuing with train only.")
        else:
            test_df['date'] = pd.to_datetime(test_df['date'])
            combined_df = pd.concat([combined_df, test_df], ignore_index=True)
            combined_df = combined_df.sort_values(['instance_id','date']).drop_duplicates(subset=['instance_id','date'], keep='last').reset_index(drop=True)

    if args.year is None:
        year = int(combined_df['date'].dt.year.max())
    else:
        year = int(args.year)

    validation_end = pd.Timestamp(year=year, month=12, day=31).date()

    last_data_date = combined_df['date'].max().date()

    sep30 = pd.Timestamp(year=year, month=9, day=30).date()
    if last_data_date < sep30:
        print(f"Warning: combined train+test data latest date is {last_data_date}, which is before {sep30}.")
        print("If you want forecasts to start on Oct 1 you must supply test data that covers Aug+Sep so combined data reaches Sep 30.")
    else:
        pass

    # Compute forecast_days
    if args.forecast_days is not None:
        forecast_days = int(args.forecast_days)
    else:
        # forecast should start the day after last_data_date
        delta_days = (validation_end - last_data_date).days
        if delta_days < 1:
            print(f"Note: validation_end {validation_end} is on or before last data date {last_data_date}. Setting forecast_days=1.")
            forecast_days = 1
        else:
            forecast_days = delta_days

    print(f"Using train file: {train_path}")
    if args.include_test:
        print(f"Including test file: {test_path if os.path.exists(test_path) else '(not found)'}")
    print(f"Last combined data date: {last_data_date}")
    print(f"Forecasting {forecast_days} day(s) up to {validation_end} for year {year}")
    forecasts = forecast_vm_data(combined_df, forecast_days=forecast_days)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    forecasts.to_csv(out_path, index=False)
    print(f"Saved forecasts to: {out_path}")

if __name__ == "__main__":
    main()