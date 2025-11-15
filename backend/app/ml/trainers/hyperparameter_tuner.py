"""
Hyperparameter tuning with Optuna-based Bayesian optimization.
"""
from typing import Dict, Any, Optional, Callable
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import time

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.model_selection import RandomizedSearchCV
from app.core.logging import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """
    Automatic hyperparameter tuning using Optuna (Bayesian optimization) or
    RandomizedSearchCV as fallback.
    """

    def __init__(
        self,
        model_factory: Callable,
        problem_type: str = "regression",
        time_budget_minutes: float = 5.0,
        n_trials: int = 50,
        cv_folds: int = 5,
        use_optuna: bool = True
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            model_factory: Function that creates a model instance
            problem_type: Type of problem (regression or classification)
            time_budget_minutes: Maximum time for tuning in minutes
            n_trials: Maximum number of trials for Optuna
            cv_folds: Number of cross-validation folds
            use_optuna: Whether to use Optuna (if available)
        """
        self.model_factory = model_factory
        self.problem_type = problem_type
        self.time_budget_seconds = time_budget_minutes * 60
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.best_params_ = None
        self.best_score_ = None

    def _get_search_space_random_forest(self, trial=None):
        """Get search space for Random Forest."""
        if trial:  # Optuna trial
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
        else:  # RandomizedSearchCV
            return {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [5, 10, 15, 20, 25, 30, None],
                "min_samples_split": [2, 5, 10, 15, 20],
                "min_samples_leaf": [1, 2, 4, 6, 8, 10],
                "max_features": ["sqrt", "log2", None],
            }

    def _get_search_space_xgboost(self, trial=None):
        """Get search space for XGBoost."""
        if trial:  # Optuna trial
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        else:  # RandomizedSearchCV
            return {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [3, 4, 5, 6, 8, 10],
                "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.01, 0.1, 1.0],
                "reg_lambda": [0.1, 1.0, 10.0],
            }

    def _get_search_space_lightgbm(self, trial=None):
        """Get search space for LightGBM."""
        if trial:  # Optuna trial
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        else:  # RandomizedSearchCV
            return {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [3, 5, 7, 10, 15],
                "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
                "num_leaves": [20, 31, 50, 75, 100],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.01, 0.1, 1.0],
                "reg_lambda": [0.1, 1.0, 10.0],
            }

    def _get_search_space_catboost(self, trial=None):
        """Get search space for CatBoost."""
        if trial:  # Optuna trial
            return {
                "iterations": trial.suggest_int("iterations", 100, 500),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            }
        else:  # RandomizedSearchCV
            return {
                "iterations": [100, 200, 300, 400, 500],
                "depth": [3, 4, 5, 6, 8, 10],
                "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
                "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0, 10.0],
            }

    def get_search_space(self, algorithm: str, trial=None):
        """
        Get search space for a given algorithm.

        Args:
            algorithm: Algorithm name (random_forest, xgboost, lightgbm, catboost)
            trial: Optuna trial object (None for RandomizedSearchCV)

        Returns:
            Dictionary of hyperparameters to search
        """
        algorithm = algorithm.lower()

        if algorithm == "random_forest":
            return self._get_search_space_random_forest(trial)
        elif algorithm == "xgboost":
            return self._get_search_space_xgboost(trial)
        elif algorithm == "lightgbm":
            return self._get_search_space_lightgbm(trial)
        elif algorithm == "catboost":
            return self._get_search_space_catboost(trial)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _objective_optuna(self, trial, X, y, algorithm: str, **model_kwargs):
        """Objective function for Optuna optimization."""
        # Get hyperparameters from search space
        params = self.get_search_space(algorithm, trial=trial)

        # Add any additional model kwargs
        params.update(model_kwargs)

        # Create model with suggested parameters
        model = self.model_factory(params)

        # Determine scoring metric
        if self.problem_type == "regression":
            scoring = "neg_mean_squared_error"
        else:
            scoring = "accuracy"

        # Cross-validation score
        scores = cross_val_score(
            model, X, y, cv=self.cv_folds, scoring=scoring, n_jobs=-1
        )

        # For regression, we want to minimize MSE (maximize negative MSE)
        # For classification, we want to maximize accuracy
        return scores.mean()

    def tune_optuna(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: str,
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.

        Args:
            X: Feature matrix
            y: Target vector
            algorithm: Algorithm name
            **model_kwargs: Additional model parameters

        Returns:
            Dictionary with best parameters and score
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Install it with: pip install optuna")

        logger.info(f"Starting Optuna hyperparameter tuning for {algorithm}")
        logger.info(f"Time budget: {self.time_budget_seconds/60:.1f} minutes, Max trials: {self.n_trials}")

        # Validate that data is numeric
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.error(f"Non-numeric columns detected in X: {non_numeric}")
            logger.error("Hyperparameter tuning requires preprocessed numerical data")
            raise ValueError(f"Data contains non-numeric columns: {non_numeric}. Ensure preprocessing is applied before tuning.")

        # Create study
        direction = "maximize"  # Always maximize (even for MSE it's negative)
        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=42)
        )

        # Optimize with timeout
        start_time = time.time()
        last_trial_time = start_time

        def objective(trial):
            nonlocal last_trial_time
            current_time = time.time()
            
            # Check if we've exceeded time budget
            if current_time - start_time > self.time_budget_seconds:
                raise optuna.exceptions.OptunaError("Time budget exceeded")
            
            # Check if this trial is taking too long (more than 5 minutes per trial)
            if current_time - last_trial_time > 300:  # 5 minutes
                logger.warning(f"Trial {trial.number} is taking longer than expected. Skipping...")
                raise optuna.exceptions.TrialPruned("Trial taking too long")
            
            try:
                result = self._objective_optuna(trial, X, y, algorithm, **model_kwargs)
                last_trial_time = time.time()
                return result
            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {e}")
                last_trial_time = time.time()
                # Return a very bad score so this trial is not selected
                if self.problem_type == "regression":
                    return float('inf')  # Worst possible MSE
                else:
                    return 0.0  # Worst possible accuracy

        # Progress callback to log trial completion
        def progress_callback(study, trial):
            elapsed = (time.time() - start_time) / 60
            value_str = f"{trial.value:.4f}" if trial.value is not None else "N/A"
            logger.info(
                f"Trial {trial.number + 1}/{self.n_trials} completed. "
                f"Value: {value_str}, "
                f"State: {trial.state.name}, "
                f"Elapsed: {elapsed:.2f} min"
            )

        # Wrapper to stop optimization after n_trials
        trials_completed = [0]  # Use list to allow modification in nested function

        def stop_after_n_trials(study, trial):
            """Stop optimization after n_trials complete."""
            trials_completed[0] += 1
            # Call original progress callback
            progress_callback(study, trial)
            # Stop after n_trials
            if trials_completed[0] >= self.n_trials:
                logger.info(f"Reached {self.n_trials} trials. Stopping optimization...")
                study.stop()

        try:
            logger.info("Starting study.optimize()...")
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.time_budget_seconds,
                show_progress_bar=False,
                n_jobs=1,  # Use 1 job for main optimization, parallel CV inside
                catch=(Exception,),  # Catch all exceptions to prevent hanging
                callbacks=[stop_after_n_trials]
            )
            logger.info("study.optimize() completed successfully")
        except (optuna.exceptions.OptunaError, KeyboardInterrupt, Exception) as e:
            logger.warning(f"Optimization stopped: {e}. Using best trial so far.")
            # Continue with best trial found so far

        # Ensure we have at least one completed trial
        logger.info("Checking completed trials...")
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        logger.info(f"Found {len(completed_trials)} completed trials out of {len(study.trials)} total trials")

        if not completed_trials:
            logger.error("No completed trials found. This may indicate a problem with the optimization.")
            raise RuntimeError("Hyperparameter optimization failed: no trials completed successfully")

        logger.info("Getting best params from study...")
        self.best_params_ = study.best_params
        logger.info("Getting best score from study...")
        self.best_score_ = study.best_value
        logger.info("Successfully retrieved best params and score")

        logger.info(f"Best score: {self.best_score_:.4f}")
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"Completed {len(completed_trials)}/{len(study.trials)} trials successfully in {(time.time() - start_time)/60:.2f} minutes")

        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "n_trials": len(study.trials)
        }

    def tune_randomized(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: str,
        n_iter: int = 20,
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using RandomizedSearchCV (fallback).

        Args:
            X: Feature matrix
            y: Target vector
            algorithm: Algorithm name
            n_iter: Number of parameter settings sampled
            **model_kwargs: Additional model parameters

        Returns:
            Dictionary with best parameters and score
        """
        # Adaptively adjust n_iter and cv_folds based on dataset size
        # to prevent extremely long training times on large datasets
        n_samples = len(X)
        adaptive_n_iter = n_iter
        adaptive_cv = self.cv_folds
        
        if n_samples > 100000:
            # Very large datasets: use minimal tuning
            adaptive_n_iter = min(5, n_iter)
            adaptive_cv = 3
            logger.info(f"Large dataset detected ({n_samples:,} samples). Using reduced tuning: n_iter={adaptive_n_iter}, cv={adaptive_cv}")
        elif n_samples > 50000:
            # Large datasets: reduce tuning
            adaptive_n_iter = min(10, n_iter)
            adaptive_cv = 3
            logger.info(f"Large dataset detected ({n_samples:,} samples). Using reduced tuning: n_iter={adaptive_n_iter}, cv={adaptive_cv}")
        elif n_samples > 10000:
            # Medium-large datasets: moderate tuning
            adaptive_n_iter = min(15, n_iter)
            adaptive_cv = 4
            logger.info(f"Medium-large dataset detected ({n_samples:,} samples). Using moderate tuning: n_iter={adaptive_n_iter}, cv={adaptive_cv}")
        else:
            # Smaller datasets: use defaults
            logger.info(f"Dataset size: {n_samples:,} samples. Using standard tuning: n_iter={adaptive_n_iter}, cv={adaptive_cv}")
        
        total_fits = adaptive_n_iter * adaptive_cv
        logger.info(f"Starting RandomizedSearchCV hyperparameter tuning for {algorithm}")
        logger.info(f"Total fits: {total_fits} ({adaptive_n_iter} iterations × {adaptive_cv} CV folds)")

        # Validate that data is numeric
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.error(f"Non-numeric columns detected in X: {non_numeric}")
            logger.error("Hyperparameter tuning requires preprocessed numerical data")
            raise ValueError(f"Data contains non-numeric columns: {non_numeric}. Ensure preprocessing is applied before tuning.")

        # Get search space
        param_distributions = self.get_search_space(algorithm, trial=None)

        # Create base model
        base_model = self.model_factory(model_kwargs)

        # Determine scoring metric
        if self.problem_type == "regression":
            scoring = "neg_mean_squared_error"
        else:
            scoring = "accuracy"

        # Randomized search with adaptive parameters
        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=adaptive_n_iter,
            cv=adaptive_cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        search.fit(X, y)

        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_

        logger.info(f"Best score: {self.best_score_:.4f}")
        logger.info(f"Best parameters: {self.best_params_}")

        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "n_iter": adaptive_n_iter,
            "cv_folds": adaptive_cv
        }

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: str,
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using the best available method.

        Args:
            X: Feature matrix
            y: Target vector
            algorithm: Algorithm name
            **model_kwargs: Additional model parameters

        Returns:
            Dictionary with best parameters and score
        """
        if self.use_optuna:
            return self.tune_optuna(X, y, algorithm, **model_kwargs)
        else:
            logger.warning("Optuna not available, falling back to RandomizedSearchCV")
            return self.tune_randomized(X, y, algorithm, **model_kwargs)
