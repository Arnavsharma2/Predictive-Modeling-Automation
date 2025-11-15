"""
Classification model trainer.
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold, TimeSeriesSplit
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from app.ml.trainers.trainer import BaseTrainer
from app.ml.trainers.hyperparameter_tuner import HyperparameterTuner
from app.ml.evaluation.metrics import MetricsCalculator
from app.core.logging import get_logger

logger = get_logger(__name__)


class ClassificationTrainer(BaseTrainer):
    """Trainer for classification models."""
    
    def __init__(self, algorithm: str = "random_forest"):
        """
        Initialize classification trainer.

        Args:
            algorithm: Algorithm to use (random_forest, xgboost, lightgbm, catboost)
        """
        super().__init__(model_type="classification")
        self.algorithm = algorithm.lower()
        self.model = None
        self.feature_names = None
        self.classes_ = None
    
    def _create_model(self, hyperparameters: Optional[Dict[str, Any]] = None, n_classes: Optional[int] = None):
        """Create model instance with hyperparameters."""
        hyperparameters = hyperparameters or {}
        
        if self.algorithm == "random_forest":
            # Better defaults with regularization to prevent overfitting
            default_params = {
                "n_estimators": 200,  # More trees for better generalization
                "max_depth": 15,  # Limit depth to prevent overfitting
                "min_samples_split": 10,  # Require more samples to split
                "min_samples_leaf": 4,  # Require more samples in leaf nodes
                "max_features": "sqrt",  # Use sqrt of features for each tree
                "random_state": 42,
                "n_jobs": -1  # Use all cores
            }
            default_params.update(hyperparameters)
            return RandomForestClassifier(**default_params)

        elif self.algorithm == "xgboost":
            # Better defaults with regularization
            default_params = {
                "n_estimators": 200,
                "max_depth": 5,  # Shallower trees to prevent overfitting
                "learning_rate": 0.05,  # Lower learning rate for better generalization
                "subsample": 0.8,  # Use 80% of samples per tree
                "colsample_bytree": 0.8,  # Use 80% of features per tree
                "reg_alpha": 0.1,  # L1 regularization
                "reg_lambda": 1.0,  # L2 regularization
                "random_state": 42,
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "n_jobs": -1
            }
            # Add num_class for multi-class if needed
            if n_classes and n_classes > 2:
                default_params["num_class"] = n_classes

            default_params.update(hyperparameters)
            return XGBClassifier(**default_params)

        elif self.algorithm == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")
            # LightGBM defaults - faster training and often better accuracy
            default_params = {
                "n_estimators": 200,
                "max_depth": 10,
                "learning_rate": 0.05,
                "num_leaves": 31,  # Maximum number of leaves in one tree
                "subsample": 0.8,  # Bagging fraction
                "colsample_bytree": 0.8,  # Feature fraction
                "reg_alpha": 0.1,  # L1 regularization
                "reg_lambda": 1.0,  # L2 regularization
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1  # Suppress warnings
            }
            default_params.update(hyperparameters)
            model = LGBMClassifier(**default_params)
            # Store categorical features if provided (will be used in fit)
            if hasattr(self, 'categorical_feature_indices') and self.categorical_feature_indices:
                model.categorical_feature = self.categorical_feature_indices
            return model

        elif self.algorithm == "catboost":
            # Double-check at runtime in case import failed at module load time
            global CATBOOST_AVAILABLE, CatBoostClassifier
            if not CATBOOST_AVAILABLE:
                try:
                    from catboost import CatBoostClassifier as _CatBoostClassifier
                    CatBoostClassifier = _CatBoostClassifier
                    # If we can import now, update the flag
                    CATBOOST_AVAILABLE = True
                except ImportError as e:
                    logger.error(f"CatBoost import failed: {e}")
                    raise ImportError(
                        "CatBoost is not installed. Install it with: pip install catboost. "
                        "If using Docker, ensure the container was rebuilt after adding catboost to requirements.txt. "
                        f"Original error: {e}"
                    )
            # CatBoost defaults - handles categorical features natively, robust to overfitting
            default_params = {
                "iterations": 200,
                "depth": 6,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,  # L2 regularization
                "random_state": 42,
                "verbose": False,
                "thread_count": -1
            }
            # Add categorical features if provided
            if hasattr(self, 'categorical_feature_indices') and self.categorical_feature_indices:
                # CatBoost can use feature indices or feature names
                default_params["cat_features"] = self.categorical_feature_indices
            
            # CatBoost doesn't accept 'n_jobs', convert it to 'thread_count' if present
            catboost_params = hyperparameters.copy() if hyperparameters else {}
            if "n_jobs" in catboost_params:
                # Convert n_jobs to thread_count for CatBoost
                if "thread_count" not in catboost_params:
                    catboost_params["thread_count"] = catboost_params["n_jobs"]
                del catboost_params["n_jobs"]
            
            default_params.update(catboost_params)
            return CatBoostClassifier(**default_params)

        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparameters: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        do_cross_validation: bool = True,
        cv_folds: int = 5,
        do_hyperparameter_tuning: Optional[bool] = None,
        tuning_method: str = "optuna",  # optuna, grid, or random
        tuning_params: Optional[Dict[str, Any]] = None,
        tuning_time_minutes: float = 5.0,
        use_early_stopping: bool = True,
        early_stopping_rounds: int = 10,
        categorical_features: Optional[List[str]] = None,
        auto_select_algorithm: bool = False,
        algorithms_to_try: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a classification model.

        Args:
            X: Feature matrix
            y: Target labels
            hyperparameters: Model hyperparameters
            test_size: Proportion of data for testing
            random_state: Random seed
            do_cross_validation: Whether to perform cross-validation
            cv_folds: Number of CV folds
            do_hyperparameter_tuning: Whether to tune hyperparameters (None = auto-enable for datasets > 1000 samples)
            tuning_method: Tuning method (optuna, grid, or random)
            tuning_params: Parameters for hyperparameter tuning (for grid/random methods)
            tuning_time_minutes: Time budget for hyperparameter tuning
            use_early_stopping: Whether to use early stopping for gradient boosting models
            early_stopping_rounds: Number of rounds without improvement before stopping
            categorical_features: List of categorical feature names (for LightGBM/CatBoost native support)
            auto_select_algorithm: If True, try multiple algorithms and select the best
            algorithms_to_try: List of algorithms to try if auto_select_algorithm is True
            **kwargs: Additional parameters

        Returns:
            Dictionary with trained model, metrics, and metadata
        """
        # Auto-select algorithm if requested
        if auto_select_algorithm:
            algorithms_to_try = algorithms_to_try or ["random_forest", "xgboost", "lightgbm"]
            logger.info(f"Auto-selecting best algorithm from: {algorithms_to_try}")
            
            best_algorithm = None
            best_score = 0.0  # Higher is better for classification (accuracy)
            best_results = None
            
            # Try each algorithm with quick CV
            for algo in algorithms_to_try:
                try:
                    logger.info(f"Testing algorithm: {algo}")
                    temp_trainer = ClassificationTrainer(algorithm=algo)
                    # Quick training without hyperparameter tuning
                    temp_results = temp_trainer.train(
                        X, y,
                        hyperparameters=hyperparameters,
                        test_size=test_size,
                        random_state=random_state,
                        do_cross_validation=True,
                        cv_folds=3,  # Use fewer folds for speed
                        do_hyperparameter_tuning=False,
                        use_early_stopping=use_early_stopping,
                        early_stopping_rounds=early_stopping_rounds,
                        categorical_features=categorical_features,
                        **kwargs
                    )
                    
                    # Use CV mean score to select best
                    cv_mean = temp_results.get("cv_scores", {}).get("mean", 0.0)
                    if cv_mean > best_score:
                        best_score = cv_mean
                        best_algorithm = algo
                        best_results = temp_results
                        logger.info(f"New best algorithm: {algo} (CV Accuracy: {cv_mean:.4f})")
                except Exception as e:
                    logger.warning(f"Algorithm {algo} failed: {e}. Skipping.")
                    continue
            
            if best_algorithm is None:
                logger.warning("All algorithms failed, falling back to original algorithm")
            else:
                logger.info(f"Selected best algorithm: {best_algorithm} (CV Accuracy: {best_score:.4f})")
                self.algorithm = best_algorithm
                # Return best results
                return best_results
        
        logger.info(f"Starting classification training with algorithm: {self.algorithm}")

        # Store feature names
        self.feature_names = list(X.columns)
        
        # Identify categorical feature indices for LightGBM/CatBoost
        self.categorical_feature_indices = None
        if categorical_features and self.algorithm in ["lightgbm", "catboost"]:
            # Find indices of categorical features
            self.categorical_feature_indices = [
                i for i, col in enumerate(self.feature_names) if col in categorical_features
            ]
            if self.categorical_feature_indices:
                logger.info(f"Using native categorical features for {self.algorithm}: {len(self.categorical_feature_indices)} features")

        # Get unique classes
        unique_classes = y.unique()
        n_classes = len(unique_classes)
        self.classes_ = np.sort(unique_classes)

        logger.info(f"Training for {n_classes} classes: {self.classes_.tolist()}")

        # Auto-enable hyperparameter tuning for datasets with > 1000 samples
        # But disable for very large datasets (> 200k) to avoid extremely long training times
        if do_hyperparameter_tuning is None:
            n_samples = len(X)
            if n_samples > 200000:
                do_hyperparameter_tuning = False
                logger.info(f"Auto-disabling hyperparameter tuning for very large dataset ({n_samples:,} samples) to avoid long training times")
                logger.info("Set do_hyperparameter_tuning=True explicitly if you want to enable it")
            else:
                do_hyperparameter_tuning = n_samples > 1000
                if do_hyperparameter_tuning:
                    logger.info(f"Auto-enabling hyperparameter tuning (dataset has {n_samples:,} samples)")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Prepare validation set for early stopping (for gradient boosting models)
        X_train_fit = X_train
        y_train_fit = y_train
        X_val = None
        y_val = None
        eval_set = None
        
        gradient_boosting_algorithms = ["xgboost", "lightgbm", "catboost"]
        if use_early_stopping and self.algorithm in gradient_boosting_algorithms and len(X_train) > 100:
            # Create validation set from training data (10% of training data)
            val_size = 0.1
            X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
            )
            eval_set = [(X_val, y_val)]
            logger.info(f"Created validation set for early stopping: {len(X_val)} samples")
        
        # Hyperparameter tuning
        if do_hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")

            if tuning_method == "optuna":
                # Use Optuna-based tuning
                # Wrapper for _create_model that handles n_classes
                def model_factory(params):
                    return self._create_model(params, n_classes=n_classes)

                tuner = HyperparameterTuner(
                    model_factory=model_factory,
                    problem_type="classification",
                    time_budget_minutes=tuning_time_minutes,
                    cv_folds=cv_folds
                )

                # Additional model kwargs (like random_state, n_jobs)
                model_kwargs = {
                    "random_state": random_state,
                    "n_jobs": -1
                }

                tuning_results = tuner.tune(X_train, y_train, self.algorithm, **model_kwargs)
                best_params = tuning_results["best_params"]

                # Merge with base hyperparameters
                if hyperparameters:
                    best_params.update(hyperparameters)

                logger.info(f"Best hyperparameters: {best_params}")

                # Create final model with best parameters
                self.model = self._create_model(best_params, n_classes=n_classes)
                # Add early stopping if supported
                fit_params = {}
                if use_early_stopping and self.algorithm in gradient_boosting_algorithms and eval_set:
                    if self.algorithm == "xgboost":
                        fit_params["eval_set"] = eval_set
                        fit_params["early_stopping_rounds"] = early_stopping_rounds
                        fit_params["verbose"] = False
                    elif self.algorithm == "lightgbm":
                        fit_params["eval_set"] = eval_set
                        if "early_stopping_rounds" not in best_params:
                            best_params["early_stopping_rounds"] = early_stopping_rounds
                            self.model = self._create_model(best_params, n_classes=n_classes)
                    elif self.algorithm == "catboost":
                        fit_params["eval_set"] = eval_set
                        fit_params["early_stopping_rounds"] = early_stopping_rounds
                
                if fit_params:
                    self.model.fit(X_train_fit, y_train_fit, **fit_params)
                else:
                    self.model.fit(X_train_fit, y_train_fit)

            elif tuning_method == "grid" and tuning_params:
                # Use GridSearchCV
                base_model = self._create_model(hyperparameters, n_classes=n_classes)
                model = GridSearchCV(
                    base_model,
                    tuning_params,
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=1
                )
                model.fit(X_train, y_train)
                best_params = model.best_params_
                logger.info(f"Best hyperparameters: {best_params}")

                # Create final model with best parameters
                self.model = self._create_model(best_params, n_classes=n_classes)
                # Add early stopping if supported
                fit_params = {}
                if use_early_stopping and self.algorithm in gradient_boosting_algorithms and eval_set:
                    if self.algorithm == "xgboost":
                        fit_params["eval_set"] = eval_set
                        fit_params["early_stopping_rounds"] = early_stopping_rounds
                        fit_params["verbose"] = False
                    elif self.algorithm == "lightgbm":
                        fit_params["eval_set"] = eval_set
                        if "early_stopping_rounds" not in best_params:
                            best_params["early_stopping_rounds"] = early_stopping_rounds
                            self.model = self._create_model(best_params, n_classes=n_classes)
                    elif self.algorithm == "catboost":
                        fit_params["eval_set"] = eval_set
                        fit_params["early_stopping_rounds"] = early_stopping_rounds
                
                if fit_params:
                    self.model.fit(X_train_fit, y_train_fit, **fit_params)
                else:
                    self.model.fit(X_train_fit, y_train_fit)

            elif tuning_method == "random" and tuning_params:
                # Use RandomizedSearchCV
                base_model = self._create_model(hyperparameters, n_classes=n_classes)
                n_iter = kwargs.get('n_iter', 20)
                model = RandomizedSearchCV(
                    base_model,
                    tuning_params,
                    cv=cv_folds,
                    scoring='accuracy',
                    n_iter=n_iter,
                    n_jobs=-1,
                    random_state=random_state,
                    verbose=1
                )
                model.fit(X_train, y_train)
                best_params = model.best_params_
                logger.info(f"Best hyperparameters: {best_params}")

                # Create final model with best parameters
                self.model = self._create_model(best_params, n_classes=n_classes)
                # Add early stopping if supported
                fit_params = {}
                if use_early_stopping and self.algorithm in gradient_boosting_algorithms and eval_set:
                    if self.algorithm == "xgboost":
                        fit_params["eval_set"] = eval_set
                        fit_params["early_stopping_rounds"] = early_stopping_rounds
                        fit_params["verbose"] = False
                    elif self.algorithm == "lightgbm":
                        fit_params["eval_set"] = eval_set
                        if "early_stopping_rounds" not in best_params:
                            best_params["early_stopping_rounds"] = early_stopping_rounds
                            self.model = self._create_model(best_params, n_classes=n_classes)
                    elif self.algorithm == "catboost":
                        fit_params["eval_set"] = eval_set
                        fit_params["early_stopping_rounds"] = early_stopping_rounds
                
                if fit_params:
                    self.model.fit(X_train_fit, y_train_fit, **fit_params)
                else:
                    self.model.fit(X_train_fit, y_train_fit)
            else:
                logger.warning(f"Tuning method '{tuning_method}' requires tuning_params, using default parameters")
                # Train with default parameters
                self.model = self._create_model(hyperparameters, n_classes=n_classes)
                # Add early stopping if supported
                fit_params = {}
                if use_early_stopping and self.algorithm in gradient_boosting_algorithms and eval_set:
                    if self.algorithm == "xgboost":
                        fit_params["eval_set"] = eval_set
                        fit_params["early_stopping_rounds"] = early_stopping_rounds
                        fit_params["verbose"] = False
                    elif self.algorithm == "lightgbm":
                        fit_params["eval_set"] = eval_set
                        if "early_stopping_rounds" not in (hyperparameters or {}):
                            model_params = hyperparameters or {}
                            model_params["early_stopping_rounds"] = early_stopping_rounds
                            self.model = self._create_model(model_params, n_classes=n_classes)
                    elif self.algorithm == "catboost":
                        fit_params["eval_set"] = eval_set
                        fit_params["early_stopping_rounds"] = early_stopping_rounds
                
                if fit_params:
                    self.model.fit(X_train_fit, y_train_fit, **fit_params)
                else:
                    self.model.fit(X_train_fit, y_train_fit)
        else:
            # Train model with default parameters
            self.model = self._create_model(hyperparameters, n_classes=n_classes)
            # Add early stopping if supported
            fit_params = {}
            if use_early_stopping and self.algorithm in gradient_boosting_algorithms and eval_set:
                if self.algorithm == "xgboost":
                    fit_params["eval_set"] = eval_set
                    fit_params["early_stopping_rounds"] = early_stopping_rounds
                    fit_params["verbose"] = False
                elif self.algorithm == "lightgbm":
                    fit_params["eval_set"] = eval_set
                    if "early_stopping_rounds" not in (hyperparameters or {}):
                        model_params = hyperparameters or {}
                        model_params["early_stopping_rounds"] = early_stopping_rounds
                        self.model = self._create_model(model_params, n_classes=n_classes)
                elif self.algorithm == "catboost":
                    fit_params["eval_set"] = eval_set
                    fit_params["early_stopping_rounds"] = early_stopping_rounds
            
            if fit_params:
                self.model.fit(X_train_fit, y_train_fit, **fit_params)
            else:
                self.model.fit(X_train_fit, y_train_fit)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = MetricsCalculator.calculate_classification_metrics(y_train, y_train_pred)
        test_metrics = MetricsCalculator.calculate_classification_metrics(y_test, y_test_pred)
        
        # Cross-validation scores
        cv_scores = None
        if do_cross_validation:
            logger.info("Performing cross-validation...")
            
            # Use stratified K-Fold for classification to maintain class distribution
            # Use time-series aware CV if timestamp column exists
            cv_strategy = None
            if 'timestamp' in X_train.columns or 'date' in X_train.columns:
                logger.info("Using time-series aware cross-validation")
                cv_strategy = TimeSeriesSplit(n_splits=cv_folds)
            else:
                logger.info("Using stratified K-Fold cross-validation for classification")
                cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            # Create a model copy for CV without early stopping
            # Early stopping requires eval_set which cross_val_score doesn't provide
            cv_model_params = self.model.get_params() if hasattr(self.model, 'get_params') else {}
            # Remove early_stopping_rounds for CV
            if 'early_stopping_rounds' in cv_model_params:
                cv_model_params = cv_model_params.copy()
                cv_model_params.pop('early_stopping_rounds', None)
                cv_model = self._create_model(cv_model_params, n_classes=n_classes)
            else:
                cv_model = self.model
            
            cv_scores = cross_val_score(
                cv_model,
                X_train,
                y_train,
                cv=cv_strategy,
                scoring='accuracy',
                n_jobs=-1
            )
            cv_scores = {
                "mean": float(cv_scores.mean()),
                "std": float(cv_scores.std()),
                "scores": [float(s) for s in cv_scores]
            }
        
        # Feature importance
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        results = {
            "model": self.model,
            "feature_names": self.feature_names,
            "classes_": self.classes_,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "cv_scores": cv_scores,
            "feature_importance": feature_importance,
            "algorithm": self.algorithm,
            "hyperparameters": self.model.get_params() if hasattr(self.model, 'get_params') else hyperparameters
        }
        
        logger.info(f"Training completed. Test Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")
        
        return results

