# model/vm/forecastvmdata.py
import pandas as pd
from .prophet.prophet_utils import prepare_prophet_df
from .prophet.train_prophet_model import train_and_forecast

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

            # Keep ONLY future rows
            forecast = forecast.tail(forecast_days)

            # Keep only ds and yhat
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

    final_df = pd.concat(all_forecasts, ignore_index=True)

    return final_df
