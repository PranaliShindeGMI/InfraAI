# model/vm/prophet/prophet_utils.py
import pandas as pd

def prepare_prophet_df(df, date_col, value_col):
    """
    Convert dataframe to Prophet format:
    ds = date
    y  = value
    """
    prophet_df = df[[date_col, value_col]].copy()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    return prophet_df
