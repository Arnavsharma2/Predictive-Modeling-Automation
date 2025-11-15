"""
Hyperparameter search service using Optuna.
"""
from typing import Dict, Any, Optional, Callable, List
import pandas as pd
import numpy as np

from app.ml.optimization.optuna_optimizer import OptunaOptimizer
from app.core.logging import get_logger

logger = get_logger(__name__)


class HyperparameterSearch:
    """
    High-level service for hyperparameter optimization.
    
    Provides convenient methods for optimizing model hyperparameters
    using Optuna with integration into the training pipeline.
    """
    
    # Default hyperparameter search spaces for common algorithms
    DEFAULT_SEARCH_SPACES = {
        "random_forest": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 30, "log": False},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]}
        },
        "xgboost": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 15},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 0.0, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0, "log": True}
        },
        "lightgbm": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 15},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 10, "high": 300},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 0.0, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0, "log": True}
        },
        "catboost": {
            "iterations": {"type": "int", "low": 50, "high": 500},
            "depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0}
        },
        "gradient_boosting": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0}
        }
    }
    
    def __init__(
        self,
        study_name: Optional[str] = None,
        n_trials: int = 100,
        timeout: Optional[float] = None
    ):
        """
        Initialize hyperparameter search.
        
        Args:
            study_name: Name of the study
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds
        """
        self.optimizer = OptunaOptimizer(
            study_name=study_name,
            n_trials=n_trials,
            timeout=timeout
        )
        self.study_name = study_name
    
    def get_search_space(
        self,
        algorithm: str,
        custom_space: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get hyperparameter search space for an algorithm.
        
        Args:
            algorithm: Algorithm name
            custom_space: Custom search space to override defaults
        
        Returns:
            Search space dictionary
        """
        if custom_space:
            return custom_space
        
        if algorithm in self.DEFAULT_SEARCH_SPACES:
            return self.DEFAULT_SEARCH_SPACES[algorithm]
        
        logger.warning(f"No default search space for algorithm '{algorithm}', using empty space")
        return {}
    
    def optimize_hyperparameters(
        self,
        objective_func: Callable,
        search_space: Dict[str, Dict[str, Any]],
        metric_name: str = "rmse",
        direction: str = "minimize",
        n_trials: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a model.
        
        Args:
            objective_func: Objective function that takes (trial, X, y) and returns score
            search_space: Hyperparameter search space
            metric_name: Name of the metric to optimize
            direction: "minimize" or "maximize"
            n_trials: Number of trials (overrides initialization)
        
        Returns:
            Dictionary with best hyperparameters and optimization results
        """
        # Create study
        self.optimizer.direction = direction
        study = self.optimizer.create_study(study_name=self.study_name)
        
        # Define objective wrapper
        def objective(trial):
            # Suggest hyperparameters
            params = self.optimizer.suggest_hyperparameters(trial, search_space)
            
            # Call objective function
            score = objective_func(trial, params)
            
            return score
        
        # Run optimization
        self.optimizer.optimize(objective, n_trials=n_trials)
        
        # Get results
        best_params = self.optimizer.get_best_params()
        best_value = self.optimizer.get_best_value()
        best_trial = self.optimizer.get_best_trial()
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "best_trial_number": best_trial.number,
            "n_trials": len(study.trials),
            "study_summary": self.optimizer.get_study_summary()
        }
    
    def create_objective_function(
        self,
        train_func: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        metric_name: str = "rmse",
        cv_folds: int = 5
    ) -> Callable:
        """
        Create an objective function for hyperparameter optimization.
        
        Args:
            train_func: Training function that takes (X, y, hyperparameters) and returns metrics
            X: Feature matrix
            y: Target vector
            metric_name: Metric to optimize
            cv_folds: Number of cross-validation folds
        
        Returns:
            Objective function for Optuna
        """
        def objective(trial, hyperparameters):
            """
            Objective function for Optuna.
            
            Args:
                trial: Optuna trial object
                hyperparameters: Suggested hyperparameters
            
            Returns:
                Score to optimize
            """
            try:
                # Train model with suggested hyperparameters
                results = train_func(X, y, hyperparameters)
                
                # Get the metric to optimize
                if metric_name in results:
                    score = results[metric_name]
                elif f"test_{metric_name}" in results:
                    score = results[f"test_{metric_name}"]
                else:
                    logger.warning(f"Metric '{metric_name}' not found in results, using first metric")
                    score = list(results.values())[0] if results else float('inf')
                
                # Report intermediate value for pruning
                if hasattr(trial, 'report'):
                    trial.report(score, step=0)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
                
                return score
            
            except Exception as e:
                logger.error(f"Error in objective function: {e}", exc_info=True)
                # Return a bad score so this trial is not selected
                return float('inf') if self.optimizer.direction == "minimize" else float('-inf')
        
        return objective

