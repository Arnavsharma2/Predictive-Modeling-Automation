"""
Ensemble model trainer with stacking, blending, and voting methods.
"""
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from app.ml.trainers.trainer import BaseTrainer
from app.ml.trainers.regression_trainer import RegressionTrainer
from app.ml.trainers.classification_trainer import ClassificationTrainer
from app.ml.evaluation.metrics import MetricsCalculator
from app.core.logging import get_logger

logger = get_logger(__name__)


class EnsembleTrainer(BaseTrainer):
    """
    Ensemble trainer that combines multiple models using stacking, blending, or voting.
    """

    def __init__(
        self,
        problem_type: str = "regression",
        ensemble_method: str = "stacking",
        base_algorithms: Optional[List[str]] = None
    ):
        """
        Initialize ensemble trainer.

        Args:
            problem_type: Type of problem (regression or classification)
            ensemble_method: Ensemble method (stacking, blending, voting)
            base_algorithms: List of algorithms to use as base models
        """
        super().__init__(model_type=problem_type)
        self.problem_type = problem_type
        self.ensemble_method = ensemble_method.lower()
        self.base_algorithms = base_algorithms or self._get_default_algorithms()
        self.model = None
        self.base_models = []
        self.feature_names = None

    def _get_default_algorithms(self) -> List[str]:
        """Get default algorithms for the problem type."""
        # Use the best performing algorithms by default
        return ["random_forest", "xgboost", "lightgbm"]

    def _create_base_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int = 42
    ) -> List[Tuple[str, Any]]:
        """
        Create base models for ensemble.

        Args:
            X: Feature matrix
            y: Target vector
            random_state: Random seed

        Returns:
            List of (name, model) tuples
        """
        base_models = []

        for algorithm in self.base_algorithms:
            try:
                if self.problem_type == "regression":
                    trainer = RegressionTrainer(algorithm=algorithm)
                else:
                    trainer = ClassificationTrainer(algorithm=algorithm)

                # Create the model without training
                if self.problem_type == "regression":
                    model = trainer._create_model()
                else:
                    # For classification, we need to know the number of classes
                    n_classes = len(y.unique())
                    model = trainer._create_model(n_classes=n_classes)

                base_models.append((algorithm, model))
                logger.info(f"Added {algorithm} to ensemble")
            except Exception as e:
                logger.warning(f"Failed to add {algorithm} to ensemble: {e}")

        if not base_models:
            raise ValueError("No base models could be created")

        return base_models

    def _create_meta_learner(self):
        """Create meta-learner for stacking."""
        if self.problem_type == "regression":
            # Use Ridge regression for regression tasks
            return Ridge(alpha=1.0, random_state=42)
        else:
            # Use Logistic Regression for classification tasks
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )

    def train_stacking(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train ensemble using stacking method.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed
            cv_folds: Number of CV folds
            **kwargs: Additional parameters

        Returns:
            Dictionary with trained model, metrics, and metadata
        """
        logger.info("Training ensemble using stacking method")

        # Create base models
        base_models = self._create_base_models(X, y, random_state)

        # Create meta-learner
        meta_learner = self._create_meta_learner()

        # Create stacking ensemble
        if self.problem_type == "regression":
            self.model = StackingRegressor(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=cv_folds,
                n_jobs=-1
            )
        else:
            self.model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=cv_folds,
                n_jobs=-1
            )

        # Split data
        if self.problem_type == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        # Train stacking ensemble
        self.model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics
        if self.problem_type == "regression":
            train_metrics = MetricsCalculator.calculate_regression_metrics(y_train, y_train_pred)
            test_metrics = MetricsCalculator.calculate_regression_metrics(y_test, y_test_pred)
        else:
            train_metrics = MetricsCalculator.calculate_classification_metrics(y_train, y_train_pred)
            test_metrics = MetricsCalculator.calculate_classification_metrics(y_test, y_test_pred)

        results = {
            "model": self.model,
            "feature_names": self.feature_names,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "ensemble_method": "stacking",
            "base_algorithms": self.base_algorithms,
            "meta_learner": type(meta_learner).__name__
        }

        logger.info(f"Stacking training completed. Test metrics: {test_metrics}")

        return results

    def train_blending(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        blend_size: float = 0.2,
        random_state: int = 42,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train ensemble using blending method.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for final testing
            blend_size: Proportion of training data for blending
            random_state: Random seed
            **kwargs: Additional parameters

        Returns:
            Dictionary with trained model, metrics, and metadata
        """
        logger.info("Training ensemble using blending method")

        # Split data into train, blend, and test
        if self.problem_type == "classification":
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            X_train, X_blend, y_train, y_blend = train_test_split(
                X_temp, y_temp, test_size=blend_size, random_state=random_state, stratify=y_temp
            )
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            X_train, X_blend, y_train, y_blend = train_test_split(
                X_temp, y_temp, test_size=blend_size, random_state=random_state
            )

        logger.info(f"Train: {len(X_train)}, Blend: {len(X_blend)}, Test: {len(X_test)}")

        # Train base models on training data
        self.base_models = []
        blend_predictions = []

        for algorithm in self.base_algorithms:
            try:
                if self.problem_type == "regression":
                    trainer = RegressionTrainer(algorithm=algorithm)
                else:
                    trainer = ClassificationTrainer(algorithm=algorithm)

                # Train the model
                logger.info(f"Training {algorithm} for blending")
                if self.problem_type == "regression":
                    model = trainer._create_model()
                else:
                    n_classes = len(y.unique())
                    model = trainer._create_model(n_classes=n_classes)

                model.fit(X_train, y_train)
                self.base_models.append((algorithm, model))

                # Generate predictions on blend set
                blend_pred = model.predict(X_blend)
                blend_predictions.append(blend_pred)

            except Exception as e:
                logger.warning(f"Failed to train {algorithm} for blending: {e}")

        if not blend_predictions:
            raise ValueError("No base models could be trained")

        # Create blend features
        X_blend_meta = np.column_stack(blend_predictions)

        # Train meta-learner on blend set
        meta_learner = self._create_meta_learner()
        meta_learner.fit(X_blend_meta, y_blend)

        # Generate predictions on test set
        test_predictions = []
        for algorithm, model in self.base_models:
            test_pred = model.predict(X_test)
            test_predictions.append(test_pred)

        X_test_meta = np.column_stack(test_predictions)
        y_test_pred = meta_learner.predict(X_test_meta)

        # Also get train predictions for metrics
        train_predictions = []
        for algorithm, model in self.base_models:
            train_pred = model.predict(X_train)
            train_predictions.append(train_pred)

        X_train_meta = np.column_stack(train_predictions)
        y_train_pred = meta_learner.predict(X_train_meta)

        # Calculate metrics
        if self.problem_type == "regression":
            train_metrics = MetricsCalculator.calculate_regression_metrics(y_train, y_train_pred)
            test_metrics = MetricsCalculator.calculate_regression_metrics(y_test, y_test_pred)
        else:
            train_metrics = MetricsCalculator.calculate_classification_metrics(y_train, y_train_pred)
            test_metrics = MetricsCalculator.calculate_classification_metrics(y_test, y_test_pred)

        # Store the model (we'll save both base models and meta learner)
        self.model = {
            "base_models": self.base_models,
            "meta_learner": meta_learner
        }

        results = {
            "model": self.model,
            "feature_names": self.feature_names,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "ensemble_method": "blending",
            "base_algorithms": self.base_algorithms,
            "meta_learner": type(meta_learner).__name__
        }

        logger.info(f"Blending training completed. Test metrics: {test_metrics}")

        return results

    def train_voting(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        voting: str = "soft",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train ensemble using voting method.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed
            voting: Voting type ('hard' or 'soft' for classification, 'soft' for regression)
            **kwargs: Additional parameters

        Returns:
            Dictionary with trained model, metrics, and metadata
        """
        logger.info(f"Training ensemble using {voting} voting method")

        # Create base models
        base_models = self._create_base_models(X, y, random_state)

        # Create voting ensemble
        if self.problem_type == "regression":
            self.model = VotingRegressor(
                estimators=base_models,
                n_jobs=-1
            )
        else:
            self.model = VotingClassifier(
                estimators=base_models,
                voting=voting,
                n_jobs=-1
            )

        # Split data
        if self.problem_type == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        # Train voting ensemble
        self.model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics
        if self.problem_type == "regression":
            train_metrics = MetricsCalculator.calculate_regression_metrics(y_train, y_train_pred)
            test_metrics = MetricsCalculator.calculate_regression_metrics(y_test, y_test_pred)
        else:
            train_metrics = MetricsCalculator.calculate_classification_metrics(y_train, y_train_pred)
            test_metrics = MetricsCalculator.calculate_classification_metrics(y_test, y_test_pred)

        results = {
            "model": self.model,
            "feature_names": self.feature_names,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "ensemble_method": f"{voting}_voting",
            "base_algorithms": self.base_algorithms
        }

        logger.info(f"Voting training completed. Test metrics: {test_metrics}")

        return results

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train ensemble model using the specified method.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed
            **kwargs: Additional parameters

        Returns:
            Dictionary with trained model, metrics, and metadata
        """
        # Store feature names
        self.feature_names = list(X.columns)

        if self.ensemble_method == "stacking":
            return self.train_stacking(X, y, test_size, random_state, **kwargs)
        elif self.ensemble_method == "blending":
            return self.train_blending(X, y, test_size, random_state, **kwargs)
        elif self.ensemble_method == "voting":
            return self.train_voting(X, y, test_size, random_state, **kwargs)
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained ensemble.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        # Handle blending separately (model is a dict)
        if isinstance(self.model, dict):
            # Blending prediction
            base_predictions = []
            for algorithm, model in self.model["base_models"]:
                pred = model.predict(X)
                base_predictions.append(pred)

            X_meta = np.column_stack(base_predictions)
            return self.model["meta_learner"].predict(X_meta)
        else:
            # Stacking or voting prediction
            return self.model.predict(X)
