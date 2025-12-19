# backend/model/train_forecast_vm.py
import pandas as pd
import os
from prophet_classifier import ProphetClassifier
from random_search_tuner import RandomSearchTuner
from piecewise_loss import PiecewiseLoss

def train_and_forecast_vm_metrics(
    metric_column: str = 'cost_usd_sum',
    forecast_days: int = 14,
    tune_hyperparameters: bool = True
):
    """
    Train Prophet model with hyperparameter tuning and forecast VM metrics.
    
    Args:
        metric_column: Which metric to forecast
        forecast_days: Number of days to forecast ahead
        tune_hyperparameters: Whether to tune hyperparameters or use defaults
    """
    
    # Load train and test data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_path = os.path.join(data_dir, 'train', 'train_vm_data.csv')
    test_path = os.path.join(data_dir, 'test', 'test_vm_data.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    train_prophet = pd.DataFrame({
        'ds': pd.to_datetime(train_df['date']),
        'y': train_df[metric_column]
    })
    
    test_prophet = pd.DataFrame({
        'ds': pd.to_datetime(test_df['date']),
        'y': test_df[metric_column]
    })
    
    print(f"\n{'='*60}")
    print(f"Training Prophet model for: {metric_column}")
    print(f"Training period: {train_prophet['ds'].min()} to {train_prophet['ds'].max()}")
    print(f"Test period: {test_prophet['ds'].min()} to {test_prophet['ds'].max()}")
    print('='*60)
    
    if tune_hyperparameters:
        # Define parameter grid for random search
        param_grid = {  
            'changepoint_prior_scale': [0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.1, 1.0, 10.0],
            'weekly_fourier_order': [5, 10, 15],
            'monthly_fourier_order': [0, 5, 10],
            'seasonality_mode': ['additive']
        }
        
        # Initialize tuner and loss function
        tuner = RandomSearchTuner(ProphetClassifier, logger=None, n_iterations=20)
        pwl = PiecewiseLoss(
            under_predict_function=lambda x: x * 2.0,  # Penalize under-predictions more
            over_predict_function=lambda x: x
        )
        
        # Generate parameter combinations
        all_params = tuner.generate_all_params(param_grid)
        
        # Find best parameters
        print("\nTuning hyperparameters...")
        best_params, best_loss = tuner.search(all_params, train_prophet, test_prophet, pwl)
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best loss: {best_loss:.4f}")
    else:
        # Use default parameters
        best_params = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'weekly_fourier_order': 10,
            'monthly_fourier_order': 5,
            'seasonality_mode': 'additive'
        }
    
    # Train final model on full training data with best parameters
    print("\nTraining final model...")
    final_model = ProphetClassifier(**best_params)
    final_model.train(train_prophet)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_forecast = final_model.forecast(test_prophet[['ds']])
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    actual = test_prophet['y'].values
    predicted = test_forecast['yhat'].values
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
    r2 = r2_score(actual, predicted)
    
    print(f"\nTest Set Performance:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    # Generate future forecast
    print(f"\nForecasting next {forecast_days} days...")
    last_date = pd.to_datetime(test_prophet['ds'].max())
    future_dates = pd.DataFrame({
        'ds': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    })
    
    future_forecast = final_model.forecast(future_dates)
    
    print("\nFuture Forecast:")
    print(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    
    # Save forecast
    output_path = os.path.join(data_dir, 'forecasts', f'forecast_{metric_column}.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    future_forecast.to_csv(output_path, index=False)
    print(f"\nForecast saved to: {output_path}")
    
    return final_model, future_forecast


if __name__ == "__main__":
    # Train and forecast for key metrics
    metrics = ['cost_usd_sum', 'cpu_utilization_mean', 'memory_used_gb_mean']
    
    for metric in metrics:
        train_and_forecast_vm_metrics(
            metric_column=metric,
            forecast_days=14,
            tune_hyperparameters=True
        )
        print("\n" + "="*60 + "\n")