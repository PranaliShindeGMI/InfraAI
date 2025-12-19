# model/vm/prophet/train_prophet_model.py
from prophet import Prophet

def train_and_forecast(df, periods=14, freq="D"):
    """
    Train Prophet and forecast future values.
    """
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
