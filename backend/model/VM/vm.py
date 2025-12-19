# model/vm/vm.py
import pandas as pd
import os

from .forecast_vm_data import forecast_vm_data

from .train_test_split_vm_data import train_test_split_vm_data

# from model.VM.forecastvmdata import forecast_vm_data

def run_vm_forecasting():
    try:
        train_test_split_vm_data()
        
        # Construct absolute path to the training data
        current_dir = os.path.dirname(__file__)
        train_csv_path = os.path.join(current_dir, "data", "train", "train_vm_data.csv")
        
        train_df = pd.read_csv(train_csv_path)
        print(f"Loaded training data: {train_df.shape[0]} rows")

    except Exception as e:
        print(f"Error during VM data forecasting process: {e}")
        return
    forecasts = forecast_vm_data(
        train_df,
        forecast_days=5
    )
    
    # Save forecasts to a single CSV file
    current_dir = os.path.dirname(__file__)
    forecast_dir = os.path.join(current_dir, "data", "forecast")
    os.makedirs(forecast_dir, exist_ok=True)

    if not forecasts.empty:
        filepath = os.path.join(forecast_dir, "vm_forecasts.csv")
        forecasts.to_csv(filepath, index=False)
        print(f"Saved forecasts to {filepath}")

    return forecasts

if __name__ == "__main__":
    run_vm_forecasting()
