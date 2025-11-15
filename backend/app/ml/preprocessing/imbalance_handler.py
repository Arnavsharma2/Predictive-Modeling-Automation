"""
Imbalanced data handling utilities for classification tasks.
"""
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

from app.core.logging import get_logger

logger = get_logger(__name__)


class ImbalanceHandler:
    """
    Handle imbalanced datasets for classification tasks.
    """

    def __init__(
        self,
        strategy: str = "auto",
        oversampling_method: str = "smote",
        undersampling_method: str = "random",
        random_state: int = 42
    ):
        """
        Initialize imbalance handler.

        Args:
            strategy: Strategy to use ('auto', 'oversample', 'undersample', 'combine', 'class_weight', 'none')
            oversampling_method: Method for oversampling ('smote', 'adasyn', 'borderline_smote')
            undersampling_method: Method for undersampling ('random')
            random_state: Random seed
        """
        self.strategy = strategy.lower()
        self.oversampling_method = oversampling_method.lower()
        self.undersampling_method = undersampling_method.lower()
        self.random_state = random_state
        self.sampler = None
        self.class_weights = None
        self.is_fitted = False

    def _detect_imbalance(self, y: pd.Series) -> Dict[str, Any]:
        """
        Detect if dataset is imbalanced.

        Args:
            y: Target labels

        Returns:
            Dictionary with imbalance information
        """
        value_counts = y.value_counts()
        n_classes = len(value_counts)
        min_class_count = value_counts.min()
        max_class_count = value_counts.max()
        imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')

        # Consider imbalanced if ratio > 2:1
        is_imbalanced = imbalance_ratio > 2.0

        return {
            "is_imbalanced": is_imbalanced,
            "imbalance_ratio": imbalance_ratio,
            "n_classes": n_classes,
            "class_counts": value_counts.to_dict(),
            "min_class_count": min_class_count,
            "max_class_count": max_class_count
        }

    def _auto_select_strategy(self, imbalance_info: Dict[str, Any]) -> str:
        """
        Automatically select best strategy based on imbalance characteristics.

        Args:
            imbalance_info: Imbalance information dictionary

        Returns:
            Selected strategy
        """
        imbalance_ratio = imbalance_info["imbalance_ratio"]
        min_class_count = imbalance_info["min_class_count"]
        n_classes = imbalance_info["n_classes"]

        # For very severe imbalance (>10:1), use class weights (no data modification)
        if imbalance_ratio > 10.0:
            return "class_weight"

        # For moderate imbalance (2:1 to 10:1) with enough samples, use SMOTE
        if imbalance_ratio > 2.0 and min_class_count >= 50:
            return "oversample"

        # For severe imbalance with few samples, use combination
        if imbalance_ratio > 5.0 and min_class_count < 50:
            return "combine"

        # For binary classification with moderate imbalance, use oversampling
        if n_classes == 2 and imbalance_ratio > 2.0:
            return "oversample"

        # Default: use class weights
        return "class_weight"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ImbalanceHandler':
        """
        Fit the imbalance handler.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Self
        """
        # Detect imbalance
        imbalance_info = self._detect_imbalance(y)
        logger.info(f"Imbalance detection: ratio={imbalance_info['imbalance_ratio']:.2f}, "
                   f"is_imbalanced={imbalance_info['is_imbalanced']}")

        # Auto-select strategy if needed
        if self.strategy == "auto":
            self.strategy = self._auto_select_strategy(imbalance_info)
            logger.info(f"Auto-selected strategy: {self.strategy}")

        # Compute class weights if needed
        if self.strategy == "class_weight" or self.strategy == "combine":
            unique_classes = np.unique(y)
            class_weights = compute_class_weight(
                'balanced',
                classes=unique_classes,
                y=y.values
            )
            self.class_weights = dict(zip(unique_classes, class_weights))
            logger.info(f"Computed class weights: {self.class_weights}")

        # Create sampler if needed
        if self.strategy in ["oversample", "undersample", "combine"]:
            if not IMBLEARN_AVAILABLE:
                logger.warning("imbalanced-learn not available. Install with: pip install imbalanced-learn")
                logger.warning("Falling back to class_weight strategy")
                self.strategy = "class_weight"
                # Compute class weights as fallback
                unique_classes = np.unique(y)
                class_weights = compute_class_weight(
                    'balanced',
                    classes=unique_classes,
                    y=y.values
                )
                self.class_weights = dict(zip(unique_classes, class_weights))
            else:
                if self.strategy == "oversample":
                    if self.oversampling_method == "smote":
                        self.sampler = SMOTE(random_state=self.random_state, n_jobs=-1)
                    elif self.oversampling_method == "adasyn":
                        self.sampler = ADASYN(random_state=self.random_state, n_jobs=-1)
                    elif self.oversampling_method == "borderline_smote":
                        self.sampler = BorderlineSMOTE(random_state=self.random_state, n_jobs=-1)
                    else:
                        self.sampler = SMOTE(random_state=self.random_state, n_jobs=-1)

                elif self.strategy == "undersample":
                    self.sampler = RandomUnderSampler(random_state=self.random_state)

                elif self.strategy == "combine":
                    if self.oversampling_method == "smote":
                        self.sampler = SMOTETomek(random_state=self.random_state, n_jobs=-1)
                    else:
                        self.sampler = SMOTEENN(random_state=self.random_state, n_jobs=-1)

        self.is_fitted = True
        return self

    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and resample data.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Resampled (X, y) tuple
        """
        if not self.is_fitted:
            self.fit(X, y)

        if self.strategy == "none":
            return X, y

        if self.strategy == "class_weight":
            # No resampling, just return original data
            return X, y

        if self.sampler is None:
            logger.warning("No sampler available, returning original data")
            return X, y

        try:
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            
            # Convert back to DataFrame/Series if needed
            if isinstance(X, pd.DataFrame):
                X_resampled = pd.DataFrame(
                    X_resampled,
                    columns=X.columns,
                    index=range(len(X_resampled))
                )
            else:
                X_resampled = pd.DataFrame(X_resampled)

            if isinstance(y, pd.Series):
                y_resampled = pd.Series(
                    y_resampled,
                    name=y.name,
                    index=range(len(y_resampled))
                )
            else:
                y_resampled = pd.Series(y_resampled)

            logger.info(f"Resampled data: {len(X)} -> {len(X_resampled)} samples")
            return X_resampled, y_resampled

        except Exception as e:
            logger.error(f"Resampling failed: {e}. Returning original data.")
            return X, y

    def get_class_weights(self) -> Optional[Dict[Any, float]]:
        """
        Get computed class weights.

        Returns:
            Dictionary mapping class labels to weights, or None
        """
        return self.class_weights


def handle_imbalanced_data(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str = "auto",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, Optional[Dict[Any, float]]]:
    """
    Convenience function to handle imbalanced data.

    Args:
        X: Feature matrix
        y: Target labels
        strategy: Strategy to use
        random_state: Random seed

    Returns:
        Tuple of (resampled X, resampled y, class_weights)
    """
    handler = ImbalanceHandler(strategy=strategy, random_state=random_state)
    X_resampled, y_resampled = handler.fit_resample(X, y)
    class_weights = handler.get_class_weights()
    return X_resampled, y_resampled, class_weights

