"""
Optuna optimizer for hyperparameter optimization.
"""
import optuna
from typing import Dict, Any, Optional, Callable, List
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimizer.
    
    Supports Bayesian optimization, multi-objective optimization,
    and early stopping/pruning.
    """
    
    def __init__(
        self,
        study_name: Optional[str] = None,
        direction: str = "minimize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        storage: Optional[str] = None
    ):
        """
        Initialize Optuna optimizer.
        
        Args:
            study_name: Name of the study
            direction: "minimize" or "maximize"
            sampler: Optuna sampler (default: TPESampler)
            pruner: Optuna pruner (default: MedianPruner)
            n_trials: Number of trials to run
            timeout: Maximum time in seconds (None for no limit)
            storage: Storage URL for study persistence (None for in-memory)
        """
        self.study_name = study_name
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        
        # Default sampler: Tree-structured Parzen Estimator (TPE)
        self.sampler = sampler or TPESampler(seed=42)
        
        # Default pruner: Median pruner for early stopping
        self.pruner = pruner or MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10
        )
        
        self.storage = storage
        self.study: Optional[optuna.Study] = None
    
    def create_study(
        self,
        study_name: Optional[str] = None,
        directions: Optional[List[str]] = None
    ) -> optuna.Study:
        """
        Create or load an Optuna study.
        
        Args:
            study_name: Name of the study
            directions: List of directions for multi-objective optimization
        
        Returns:
            Optuna study object
        """
        study_name = study_name or self.study_name or "default_study"
        
        if directions:
            # Multi-objective optimization
            self.study = optuna.create_study(
                study_name=study_name,
                directions=directions,
                sampler=self.sampler,
                pruner=self.pruner,
                storage=self.storage,
                load_if_exists=True
            )
        else:
            # Single-objective optimization
            self.study = optuna.create_study(
                study_name=study_name,
                direction=self.direction,
                sampler=self.sampler,
                pruner=self.pruner,
                storage=self.storage,
                load_if_exists=True
            )
        
        logger.info(f"Created/loaded study '{study_name}' with {len(directions) if directions else 1} objective(s)")
        return self.study
    
    def optimize(
        self,
        objective_func: Callable,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        show_progress_bar: bool = True
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_func: Objective function that takes a trial and returns a score
            n_trials: Number of trials (overrides initialization)
            timeout: Timeout in seconds (overrides initialization)
            show_progress_bar: Whether to show progress bar
        
        Returns:
            Optimized study
        """
        if self.study is None:
            self.create_study()
        
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout
        
        logger.info(f"Starting optimization with {n_trials} trials")
        
        try:
            self.study.optimize(
                objective_func,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=show_progress_bar,
                catch=(Exception,),  # Catch all exceptions to prevent hanging
                n_jobs=1  # Use single job to avoid resource contention
            )
        except (optuna.exceptions.OptunaError, KeyboardInterrupt, Exception) as e:
            logger.warning(f"Optimization stopped: {e}. Using best trial so far.")
            # Continue with best trial found so far
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed_trials:
                logger.error("No completed trials found. This may indicate a problem with the optimization.")
                raise RuntimeError(f"Hyperparameter optimization failed: {str(e)}")
        
        # Check if we have any completed trials
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            logger.info(f"Optimization completed. Best trial: {self.study.best_trial.number}")
        else:
            logger.warning("No completed trials found after optimization")
        
        return self.study
    
    def suggest_hyperparameters(
        self,
        trial: optuna.Trial,
        hyperparameter_space: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameters from a search space.
        
        Args:
            trial: Optuna trial object
            hyperparameter_space: Dictionary defining search space
                Example:
                {
                    "n_estimators": {"type": "int", "low": 50, "high": 500},
                    "max_depth": {"type": "int", "low": 3, "high": 20},
                    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True}
                }
        
        Returns:
            Dictionary of suggested hyperparameters
        """
        suggested = {}
        
        for param_name, param_config in hyperparameter_space.items():
            param_type = param_config.get("type", "float")
            
            if param_type == "int":
                if "choices" in param_config:
                    suggested[param_name] = trial.suggest_int(
                        param_name,
                        param_config["choices"][0],
                        param_config["choices"][-1]
                    )
                else:
                    suggested[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
            
            elif param_type == "float":
                if "choices" in param_config:
                    suggested[param_name] = trial.suggest_float(
                        param_name,
                        param_config["choices"][0],
                        param_config["choices"][-1]
                    )
                else:
                    suggested[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False),
                        step=param_config.get("step")
                    )
            
            elif param_type == "categorical":
                suggested[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
        
        return suggested
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get best hyperparameters from the study.
        
        Returns:
            Dictionary of best hyperparameters
        """
        if self.study is None:
            raise ValueError("No study created. Call create_study() first.")
        
        return self.study.best_params
    
    def get_best_value(self) -> float:
        """
        Get best objective value from the study.
        
        Returns:
            Best objective value
        """
        if self.study is None:
            raise ValueError("No study created. Call create_study() first.")
        
        return self.study.best_value
    
    def get_best_trial(self) -> optuna.Trial:
        """
        Get best trial from the study.
        
        Returns:
            Best trial object
        """
        if self.study is None:
            raise ValueError("No study created. Call create_study() first.")
        
        return self.study.best_trial
    
    def get_trials_dataframe(self) -> Any:
        """
        Get all trials as a pandas DataFrame.
        
        Returns:
            DataFrame with trial results
        """
        if self.study is None:
            raise ValueError("No study created. Call create_study() first.")
        
        return self.study.trials_dataframe()
    
    def get_study_summary(self) -> Dict[str, Any]:
        """
        Get summary of the study.
        
        Returns:
            Dictionary with study summary
        """
        if self.study is None:
            raise ValueError("No study created. Call create_study() first.")
        
        return {
            "study_name": self.study.study_name,
            "n_trials": len(self.study.trials),
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "best_trial_number": self.study.best_trial.number,
            "direction": self.study.direction
        }

