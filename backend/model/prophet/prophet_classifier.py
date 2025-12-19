# backend/model/prophet_classifier.py
import pandas as pd
from prophet import Prophet
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

class ProphetClassifier:
    """
    Wrapper around Facebook Prophet for time-series forecasting.
    Provides a consistent interface for training and forecasting.
    """
    
    def __init__(self,
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 seasonality_mode: str = 'additive',
                 weekly_fourier_order: int = 10,
                 monthly_fourier_order: int = 5,
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 interval_width: float = 0.95,
                 logger: Optional[object] = None):
        """
        Initialize Prophet model with custom parameters.
        
        Args:
            changepoint_prior_scale: Flexibility of trend (higher = more flexible)
            seasonality_prior_scale: Strength of seasonality (higher = stronger)
            seasonality_mode: 'additive' or 'multiplicative'
            weekly_fourier_order: Complexity of weekly seasonality
            monthly_fourier_order: Complexity of monthly seasonality (0 to disable)
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality
            interval_width: Width of uncertainty intervals
            logger: Optional logger object
        """
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.weekly_fourier_order = weekly_fourier_order
        self.monthly_fourier_order = monthly_fourier_order
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.interval_width = interval_width
        self.logger = logger
        
        self.model = None
        
    def train(self, train_df: pd.DataFrame):
        """
        Train the Prophet model on training data.
        
        Args:
            train_df: DataFrame with 'ds' (datetime) and 'y' (target) columns
        """
        # Validate input
        if 'ds' not in train_df.columns or 'y' not in train_df.columns:
            raise ValueError("Training data must have 'ds' and 'y' columns")
        
        # Initialize Prophet model
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=False,  # We'll add custom weekly seasonality
            daily_seasonality=self.daily_seasonality,
            interval_width=self.interval_width
        )
        
        # Add custom weekly seasonality with specified Fourier order
        if self.weekly_seasonality:
            self.model.add_seasonality(
                name='weekly',
                period=7,
                fourier_order=self.weekly_fourier_order
            )
        
        # Add custom monthly seasonality if specified
        if self.monthly_fourier_order > 0:
            self.model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=self.monthly_fourier_order
            )
        
        # Fit the model
        if self.logger:
            self.logger.info(f"Training Prophet model with {len(train_df)} data points...")
        
        self.model.fit(train_df)
        
        if self.logger:
            self.logger.info("Training complete.")
    
    def forecast(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forecasts for specified dates.
        
        Args:
            forecast_df: DataFrame with 'ds' column containing dates to forecast
            
        Returns:
            DataFrame with predictions including yhat, yhat_lower, yhat_upper
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting. Call train() first.")
        
        # Generate forecast
        forecast = self.model.predict(forecast_df)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def get_params(self) -> dict:
        """Get model parameters as dictionary"""
        return {
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'seasonality_mode': self.seasonality_mode,
            'weekly_fourier_order': self.weekly_fourier_order,
            'monthly_fourier_order': self.monthly_fourier_order,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'interval_width': self.interval_width
        }