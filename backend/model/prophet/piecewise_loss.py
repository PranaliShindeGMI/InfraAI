# backend/model/piecewise_loss.py
import numpy as np
from typing import Callable

class PiecewiseLoss:
    """
    Custom loss function that applies different penalties for under-predictions vs over-predictions.
    Useful when one type of error is more costly than the other (e.g., under-predicting costs).
    """
    
    def __init__(self, 
                 under_predict_function: Callable = lambda x: x * 2.0,
                 over_predict_function: Callable = lambda x: x):
        """
        Args:
            under_predict_function: Function to apply to under-prediction errors (default: 2x penalty)
            over_predict_function: Function to apply to over-prediction errors (default: 1x penalty)
        """
        self.under_predict_function = under_predict_function
        self.over_predict_function = over_predict_function
    
    def calculate_loss(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate piecewise loss between actual and predicted values.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            Total loss value
        """
        errors = actual - predicted
        
        # Separate under-predictions (positive errors) and over-predictions (negative errors)
        under_pred_errors = np.abs(errors[errors > 0])
        over_pred_errors = np.abs(errors[errors < 0])
        
        # Apply different penalty functions
        under_pred_loss = np.sum(self.under_predict_function(under_pred_errors))
        over_pred_loss = np.sum(self.over_predict_function(over_pred_errors))
        
        total_loss = under_pred_loss + over_pred_loss
        
        return total_loss
    
    def calculate_mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Error for comparison"""
        return np.mean(np.abs(actual - predicted))
    
    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100