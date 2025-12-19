# backend/model/random_search_tuner.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import itertools
import random

class RandomSearchTuner:
    """
    Random search hyperparameter tuner for Prophet models.
    Tests random combinations of parameters to find the best configuration.
    """
    
    def __init__(self, model_class, logger=None, n_iterations: int = 25):
        """
        Args:
            model_class: The model class to tune (e.g., ProphetClassifier)
            logger: Optional logger object
            n_iterations: Number of random parameter combinations to try
        """
        self.model_class = model_class
        self.logger = logger
        self.n_iterations = n_iterations
    
    def generate_all_params(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all possible parameter combinations from param_grid.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of possible values
            
        Returns:
            List of parameter dictionaries
        """
        keys = param_grid.keys()
        values = param_grid.values()
        
        # Generate all combinations
        all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # Randomly sample n_iterations combinations
        if len(all_combinations) > self.n_iterations:
            sampled_combinations = random.sample(all_combinations, self.n_iterations)
        else:
            sampled_combinations = all_combinations
        
        if self.logger:
            self.logger.info(f"Generated {len(sampled_combinations)} parameter combinations to test.")
        
        return sampled_combinations
    
    def search(self, 
               param_combinations: List[Dict[str, Any]], 
               train_df: pd.DataFrame, 
               test_df: pd.DataFrame,
               loss_function) -> Tuple[Dict[str, Any], float]:
        """
        Search for best parameters by testing each combination.
        
        Args:
            param_combinations: List of parameter dictionaries to test
            train_df: Training data with 'ds' and 'y' columns
            test_df: Test data with 'ds' and 'y' columns
            loss_function: Loss function object with calculate_loss method
            
        Returns:
            Tuple of (best_params, best_loss)
        """
        best_params = None
        best_loss = float('inf')
        
        for i, params in enumerate(param_combinations):
            if self.logger:
                self.logger.info(f"Testing parameter combination {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Initialize model with current parameters
                model = self.model_class(**params, logger=self.logger)
                
                # Train on training data
                model.train(train_df)
                
                # Forecast on test data
                forecast = model.forecast(test_df[['ds']])
                
                # Calculate loss
                actual = test_df['y'].values
                predicted = forecast['yhat'].values
                loss = loss_function.calculate_loss(actual, predicted)
                
                # Track best parameters
                if loss < best_loss:
                    best_loss = loss
                    best_params = params
                    if self.logger:
                        self.logger.info(f"New best loss: {best_loss:.4f}")
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed with params {params}: {str(e)}")
                continue
        
        if self.logger:
            self.logger.info(f"\nBest parameters found: {best_params}")
            self.logger.info(f"Best loss: {best_loss:.4f}")
        
        return best_params, best_loss