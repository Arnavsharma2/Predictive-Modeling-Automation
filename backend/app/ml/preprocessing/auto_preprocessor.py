"""
Automatic preprocessing pipeline that intelligently selects and applies preprocessing strategies.
"""
from typing import Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
from .preprocessor import DataPreprocessor
from .feature_engineering import FeatureEngineer, TargetEncoder, FrequencyEncoder
from .outlier_detection import OutlierDetector
from .feature_selection import FeatureSelector
import warnings

warnings.filterwarnings('ignore')


class AutoPreprocessor:
    """
    Intelligent preprocessing pipeline that automatically analyzes data
    and applies appropriate preprocessing strategies.
    """

    def __init__(
        self,
        target_column: str,
        task: Literal['classification', 'regression'] = 'regression',
        enable_feature_engineering: bool = True,
        enable_outlier_handling: bool = True,
        enable_feature_selection: bool = True,
        max_features: Optional[int] = None,
        outlier_method: str = 'iqr',
        outlier_strategy: str = 'cap',
        feature_selection_method: str = 'model_based',
        verbose: bool = False
    ):
        """
        Initialize AutoPreprocessor.

        Args:
            target_column: Name of target column
            task: Type of ML task (classification or regression)
            enable_feature_engineering: Whether to apply feature engineering
            enable_outlier_handling: Whether to handle outliers
            enable_feature_selection: Whether to perform feature selection
            max_features: Maximum number of features to keep (None = auto)
            outlier_method: Method for outlier detection
            outlier_strategy: Strategy for handling outliers
            feature_selection_method: Method for feature selection
            verbose: Whether to print progress
        """
        self.target_column = target_column
        self.task = task
        self.enable_feature_engineering = enable_feature_engineering
        self.enable_outlier_handling = enable_outlier_handling
        self.enable_feature_selection = enable_feature_selection
        self.max_features = max_features
        self.outlier_method = outlier_method
        self.outlier_strategy = outlier_strategy
        self.feature_selection_method = feature_selection_method
        self.verbose = verbose

        # Components
        self.preprocessor: Optional[DataPreprocessor] = None
        self.outlier_detector: Optional[OutlierDetector] = None
        self.feature_selector: Optional[FeatureSelector] = None
        self.feature_engineer = FeatureEngineer()

        # Metadata
        self.data_characteristics: Dict[str, Any] = {}
        self.preprocessing_steps: list = []
        self.is_fitted = False
        self._fitted_data: Optional[pd.DataFrame] = None  # Store fitted data for fit_transform
        self.target_transform: Optional[str] = None  # Track if target was transformed
        self.target_scaler: Optional[Any] = None  # Store target transformation parameters
        self.original_columns: Optional[list] = None  # Track original input column names
        self._feature_name_mapping: Optional[Dict[str, str]] = None  # Mapping from technical to original names

    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[AutoPreprocessor] {message}")

    def _analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data characteristics to inform preprocessing decisions.

        Args:
            df: Input dataframe

        Returns:
            Dictionary of data characteristics
        """
        self._log("Analyzing data characteristics...")

        characteristics = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'missing_ratio': df.isnull().sum().sum() / (len(df) * len(df.columns)),
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'text_columns': [],
            'high_cardinality_columns': [],
            'skewed_columns': [],
            'has_outliers': False
        }

        # Separate target
        X = df.drop(columns=[self.target_column]) if self.target_column in df.columns else df

        # Enhanced column type detection
        for col in X.columns:
            # Check for datetime first (can be stored as object)
            is_datetime = False
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                is_datetime = True
            else:
                # Try to detect datetime from column name
                datetime_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'when', 'at']
                if any(keyword in col.lower() for keyword in datetime_keywords):
                    try:
                        # Try to convert to datetime
                        pd.to_datetime(X[col].dropna().head(100), errors='raise')
                        is_datetime = True
                    except (ValueError, TypeError, pd.errors.ParserError):
                        pass
            
            if is_datetime:
                characteristics['datetime_columns'].append(col)
                continue
            
            # Check for numeric
            if pd.api.types.is_numeric_dtype(X[col]):
                characteristics['numeric_columns'].append(col)

                # Check for skewness
                if len(X[col].dropna()) > 0:
                    skewness = abs(X[col].skew())
                    if skewness > 1:
                        characteristics['skewed_columns'].append(col)
                continue
            
            # Handle object/string columns
            if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
                # Check if it's actually numeric stored as string
                numeric_as_string = False
                try:
                    numeric_sample = pd.to_numeric(X[col].dropna().head(100), errors='raise')
                    if len(numeric_sample) > 0:
                        numeric_as_string = True
                        characteristics['numeric_columns'].append(col)
                        continue
                except (ValueError, TypeError):
                    pass
                
                # Analyze text characteristics
                unique_ratio = X[col].nunique() / len(X) if len(X) > 0 else 0
                non_null = X[col].dropna()
                
                if len(non_null) > 0:
                    avg_length = non_null.astype(str).str.len().mean()
                    max_length = non_null.astype(str).str.len().max()
                    
                    # Enhanced text detection
                    # Check for text patterns (emails, URLs, long text)
                    sample_text = non_null.astype(str).head(100).str.cat(sep=' ')
                    has_email = '@' in sample_text and '.' in sample_text
                    has_url = any(proto in sample_text.lower() for proto in ['http://', 'https://', 'www.'])
                    has_special_chars = (non_null.astype(str).str.contains(r'[^\w\s]', regex=True).sum() / len(non_null)) > 0.3
                    
                    # Text column criteria:
                    # - Average length > 50 OR
                    # - Max length > 100 OR
                    # - High unique ratio (>0.5) with avg length > 20 OR
                    # - Contains email/URL patterns OR
                    # - High special character ratio
                    is_text = (
                        avg_length > 50 or
                        max_length > 100 or
                        (unique_ratio > 0.5 and avg_length > 20) or
                        has_email or
                        has_url or
                        (has_special_chars and avg_length > 30)
                    )
                    
                    if is_text:
                        characteristics['text_columns'].append(col)
                    else:
                        # Categorical
                        characteristics['categorical_columns'].append(col)
                        if unique_ratio > 0.3:
                            characteristics['high_cardinality_columns'].append(col)
                else:
                    # All null - treat as categorical
                    characteristics['categorical_columns'].append(col)

        # Check for outliers in numeric columns
        if characteristics['numeric_columns']:
            for col in characteristics['numeric_columns'][:5]:  # Check first 5 numeric columns
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                outliers = ((X[col] < (q1 - 3 * iqr)) | (X[col] > (q3 + 3 * iqr))).sum()
                if outliers > len(X) * 0.01:  # More than 1% outliers
                    characteristics['has_outliers'] = True
                    break

        self._log(f"Data analysis complete: {characteristics['n_rows']} rows, {characteristics['n_columns']} columns")
        self._log(f"  - {len(characteristics['numeric_columns'])} numeric, {len(characteristics['categorical_columns'])} categorical")
        self._log(f"  - {len(characteristics['datetime_columns'])} datetime, {len(characteristics['text_columns'])} text")

        return characteristics

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AutoPreprocessor':
        """
        Fit the preprocessing pipeline.

        Args:
            df: Input dataframe
            y: Target variable (optional, will be extracted from df if not provided)

        Returns:
            Self
        """
        self._log("Starting AutoPreprocessor fitting...")

        # Analyze data
        self.data_characteristics = self._analyze_data(df)

        # Extract target if not provided
        if y is None and self.target_column in df.columns:
            y = df[self.target_column]
            X = df.drop(columns=[self.target_column])
        else:
            X = df

        # Store original column names for feature name mapping
        self.original_columns = list(X.columns)
        self._log(f"Stored {len(self.original_columns)} original columns")

        # Step 0: Analyze and transform target if needed (regression only)
        if self.task == 'regression' and y is not None:
            # Check if target is skewed
            y_skewness = abs(y.skew())
            self._log(f"Target variable skewness: {y_skewness:.3f}")

            if y_skewness > 1.0:  # Highly skewed
                self._log("  - Target is highly skewed, applying log transformation...")
                # Store original stats for inverse transform
                self.target_transform = 'log1p'
                # Apply log1p transformation (handles zeros)
                y = np.log1p(y)
                self.preprocessing_steps.append("target_log_transform")
                self._log(f"  - After log transform, skewness: {abs(y.skew()):.3f}")

        # Step 1: Handle outliers
        if self.enable_outlier_handling and self.data_characteristics['has_outliers']:
            self._log(f"Handling outliers using {self.outlier_method} method...")
            self.outlier_detector = OutlierDetector(
                method=self.outlier_method,
                threshold=3.0,
                contamination=0.05
            )
            X = self.outlier_detector.fit_detect_handle(
                X,
                strategy=self.outlier_strategy,
                numeric_columns=self.data_characteristics['numeric_columns']
            )
            self.preprocessing_steps.append(f"outlier_handling_{self.outlier_method}")

        # Step 2: Adaptive Feature engineering (CONSERVATIVE approach)
        if self.enable_feature_engineering:
            self._log("Applying adaptive feature engineering...")

            # Calculate feature budget based on dataset size
            # Rule: Keep features << samples/10 to avoid overfitting
            n_samples = len(X)
            n_existing_features = len(X.columns)
            # More conservative: aim for samples/20 ratio
            max_safe_features = max(n_samples // 20, 30)  # At least 30, but scale with data
            feature_budget = max_safe_features - n_existing_features

            self._log(f"  - Dataset: {n_samples} samples, {n_existing_features} features")
            self._log(f"  - Feature budget: {feature_budget} new features allowed (conservative: samples/20 ratio)")

            if feature_budget <= 0:
                self._log("  - Skipping feature engineering: already at feature limit")
            else:
                # Determine complexity level based on dataset characteristics
                is_small_dataset = n_samples < 1000
                is_medium_dataset = 1000 <= n_samples < 10000
                is_large_dataset = n_samples >= 10000

                has_many_categoricals = len(self.data_characteristics['categorical_columns']) > 5
                has_many_numerics = len(self.data_characteristics['numeric_columns']) > 10

                # SKIP polynomial features - they cause feature explosion
                # Only use for very specific small datasets
                if is_small_dataset and not has_many_categoricals and not has_many_numerics:
                    if len(self.data_characteristics['numeric_columns']) <= 3:
                        try:
                            self._log("  - Creating polynomial interaction features (very limited)...")
                            # Only take top 2 numeric features to limit explosion
                            X = self.feature_engineer.create_polynomial_features(
                                X,
                                self.data_characteristics['numeric_columns'][:2],
                                degree=2,
                                interaction_only=True
                            )
                            self.preprocessing_steps.append("polynomial_features")
                        except Exception as e:
                            self._log(f"  - Polynomial features failed: {e}")
                else:
                    self._log("  - Skipping polynomial features: too many features would be created")

                # Log transformation for skewed features (always useful, doesn't explode features)
                # But limit to most skewed features
                if self.data_characteristics['skewed_columns']:
                    try:
                        # Only transform the MOST skewed columns (top 50% by skewness)
                        cols_to_transform = self.data_characteristics['skewed_columns'][:max(len(self.data_characteristics['skewed_columns'])//2, 1)]
                        self._log(f"  - Applying log transform to {len(cols_to_transform)} most skewed columns...")
                        X = self.feature_engineer.create_log_features(
                            X,
                            cols_to_transform
                        )
                        self.preprocessing_steps.append("log_transform")
                    except Exception as e:
                        self._log(f"  - Log transform failed: {e}")

                # SKIP binning and clustering - they create categorical explosion with one-hot encoding
                self._log("  - Skipping binning/clustering: causes feature explosion with one-hot encoding")

        # Step 3: Basic preprocessing (scaling, encoding)
        self._log("Applying basic preprocessing (scaling, encoding)...")

        # Use target encoding or frequency encoding for high cardinality to avoid explosion
        # Target encoding is better for high cardinality but requires target variable
        # Frequency encoding is safer and doesn't require target
        high_cardinality_count = len(self.data_characteristics['high_cardinality_columns'])
        categorical_count = len(self.data_characteristics['categorical_columns'])
        
        if high_cardinality_count > 0 or categorical_count > 5:
            # Use target encoding if we have target and it's regression/classification
            if y is not None and self.task in ['regression', 'classification'] and high_cardinality_count > 0:
                # Use target encoding for high cardinality columns
                encode_method = 'target'
                self._log(f"  - Using target encoding for {high_cardinality_count} high-cardinality columns")
            else:
                # Fall back to frequency encoding
                encode_method = 'frequency'
                self._log(f"  - Using frequency encoding ({high_cardinality_count} high-cardinality, {categorical_count} categorical)")
        else:
            encode_method = 'onehot'
            self._log("  - Using one-hot encoding")

        # Add target back to X for preprocessing (preprocessor expects to remove it)
        X_with_target = X.copy()
        X_with_target[self.target_column] = y

        self.preprocessor = DataPreprocessor(
            target_column=self.target_column,
            handle_missing='mean',
            scale_numeric=True,
            scaling_method='standard',
            encode_categorical=encode_method,
            auto_detect_types=True
        )
        X_preprocessed = self.preprocessor.fit_transform(X_with_target)
        self.preprocessing_steps.append(f"scaling_encoding_{encode_method}")

        # Step 4: Adaptive Feature selection
        if self.enable_feature_selection and len(X_preprocessed.columns) > 10:
            self._log(f"Performing adaptive feature selection using {self.feature_selection_method}...")

            # Determine number of features to keep based on sample size
            if self.max_features:
                n_features = self.max_features
            else:
                n_samples = len(X_preprocessed)
                n_current_features = len(X_preprocessed.columns)

                # Adaptive strategy based on dataset size and feature/sample ratio
                # Rule of thumb: Keep features < samples/10 for good generalization
                # But be LESS aggressive to retain more information
                max_by_samples = n_samples // 10

                # More conservative reduction strategy - keep more features to preserve information
                if n_current_features <= 50:
                    # Keep most features if we don't have many
                    n_features = int(n_current_features * 0.95)  # Keep 95% instead of 90%
                elif n_current_features <= 100:
                    # Light reduction for moderate feature counts - keep more features
                    n_features = int(n_current_features * 0.85)  # Keep 85% instead of 70%
                elif n_current_features <= 200:
                    # Medium reduction - less aggressive
                    n_features = int(n_current_features * 0.75)  # Keep 75% instead of 50%
                else:
                    # More aggressive reduction for high-dimensional data
                    n_features = int(n_current_features * 0.6)  # Keep 60% instead of 40%

                # Apply sample-based constraint
                n_features = min(n_features, max_by_samples)

                # Bounds - more reasonable limits
                n_features = max(n_features, 10)  # At least 10 features
                n_features = min(n_features, 200)  # Cap at 200 features

            self._log(f"  - Selecting top {n_features} features from {len(X_preprocessed.columns)} (ratio: {n_features/len(X_preprocessed.columns):.2%})")

            self.feature_selector = FeatureSelector(
                method=self.feature_selection_method,
                n_features=n_features,
                task=self.task
            )

            # Ensure all columns are numeric before feature selection
            non_numeric = X_preprocessed.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                self._log(f"  - WARNING: Non-numeric columns detected after preprocessing: {non_numeric}")
                self._log(f"  - Attempting to convert to numeric...")
                try:
                    for col in non_numeric:
                        X_preprocessed[col] = pd.to_numeric(X_preprocessed[col], errors='coerce')
                    # Check again
                    non_numeric_after = X_preprocessed.select_dtypes(exclude=[np.number]).columns.tolist()
                    if non_numeric_after:
                        raise ValueError(f"Could not convert columns to numeric: {non_numeric_after}")
                except Exception as convert_error:
                    self._log(f"  - Feature selection failed: could not convert non-numeric columns: {convert_error}")
                    self.feature_selector = None
                    # Continue without feature selection rather than failing completely
            
            if self.feature_selector:
                try:
                    # Store preprocessor feature names BEFORE feature selection
                    # This ensures we can map back to original technical names
                    preprocessor_feature_names = list(X_preprocessed.columns)

                    self.feature_selector.fit(X_preprocessed, y)
                    self.preprocessing_steps.append(f"feature_selection_{self.feature_selection_method}")
                    self._log(f"  - Selected {len(self.feature_selector.selected_features)} features")

                    # Map selected features back to preprocessor feature names
                    # The feature selector stores column names from X_preprocessed
                    # which might be generic (feature_0, feature_1, etc.) if preprocessor failed
                    # We need to ensure we use the actual technical names from the preprocessor
                    selected_technical_names = []
                    for sel_feature in self.feature_selector.selected_features:
                        if isinstance(sel_feature, int):
                            # Index-based selection
                            if sel_feature < len(preprocessor_feature_names):
                                selected_technical_names.append(preprocessor_feature_names[sel_feature])
                        elif isinstance(sel_feature, str):
                            # Name-based selection
                            if sel_feature in preprocessor_feature_names:
                                # Already a proper name
                                selected_technical_names.append(sel_feature)
                            elif sel_feature.startswith('feature_'):
                                # Generic name - try to map to index
                                try:
                                    idx = int(sel_feature.split('_')[1])
                                    if idx < len(preprocessor_feature_names):
                                        selected_technical_names.append(preprocessor_feature_names[idx])
                                    else:
                                        selected_technical_names.append(sel_feature)
                                except (ValueError, IndexError):
                                    selected_technical_names.append(sel_feature)
                            else:
                                selected_technical_names.append(sel_feature)
                        else:
                            selected_technical_names.append(str(sel_feature))

                    # Log samples for debugging
                    self._log(f"  - Mapped selected features to technical names")
                    if selected_technical_names:
                        sample_count = min(3, len(selected_technical_names))
                        self._log(f"  - Sample technical names: {selected_technical_names[:sample_count]}")

                    # Replace the selected_features with properly mapped names
                    self.feature_selector.selected_features = selected_technical_names
                except Exception as e:
                    self._log(f"  - Feature selection failed: {e}")
                    self.feature_selector = None

        # Store the final preprocessed data for fit_transform
        if self.feature_selector:
            self._fitted_data = self.feature_selector.transform(X_preprocessed)
        else:
            self._fitted_data = X_preprocessed

        # Store transformed target for fit_transform to return
        self._transformed_target = y if y is not None else None

        self.is_fitted = True
        self._log(f"AutoPreprocessor fitting complete! Applied steps: {', '.join(self.preprocessing_steps)}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.

        Args:
            df: Input dataframe

        Returns:
            Preprocessed dataframe
        """
        if not self.is_fitted:
            raise ValueError("AutoPreprocessor must be fitted before transforming")

        X = df.drop(columns=[self.target_column]) if self.target_column in df.columns else df

        # Apply outlier handling
        if self.outlier_detector:
            X = self.outlier_detector.handle_outliers(
                X,
                strategy=self.outlier_strategy,
                numeric_columns=self.data_characteristics['numeric_columns']
            )

        # Apply feature engineering steps in the same order as fit()
        if self.enable_feature_engineering:
            # Polynomial features - use same number of columns as in fit()
            if 'polynomial_features' in self.preprocessing_steps:
                X = self.feature_engineer.create_polynomial_features(
                    X,
                    self.data_characteristics['numeric_columns'][:2],  # Match fit() - use top 2
                    degree=2,
                    interaction_only=True
                )

            # Log transformation
            if 'log_transform' in self.preprocessing_steps:
                X = self.feature_engineer.create_log_features(
                    X,
                    self.data_characteristics['skewed_columns']
                )

            # Binning
            if 'binning' in self.preprocessing_steps:
                X = self.feature_engineer.create_binned_features(
                    X,
                    self.data_characteristics['numeric_columns'][:3],
                    n_bins=5
                )

            # Clustering
            if 'clustering' in self.preprocessing_steps:
                X = self.feature_engineer.create_cluster_features(
                    X,
                    self.data_characteristics['numeric_columns'][:10],
                    n_clusters=min(5, len(X) // 20) if len(X) > 100 else 5
                )

        # Apply basic preprocessing
        X_preprocessed = self.preprocessor.transform(X)

        # Apply feature selection
        if self.feature_selector:
            X_preprocessed = self.feature_selector.transform(X_preprocessed)

        return X_preprocessed

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit and transform in one step.

        Returns:
            Tuple of (X_transformed, y_transformed) if y was provided, otherwise just X_transformed
        """
        self.fit(df, y)

        # Return the stored fitted data which already has all transformations applied
        if y is not None and self._transformed_target is not None:
            return self._fitted_data, self._transformed_target
        return self._fitted_data

    def get_feature_names(self) -> list:
        """Get names of features after preprocessing."""
        if not self.is_fitted:
            raise ValueError("AutoPreprocessor must be fitted first")

        if self.feature_selector:
            return self.feature_selector.selected_features
        else:
            return self.preprocessor.get_feature_names()

    def get_feature_name_mapping(self) -> Dict[str, str]:
        """
        Get mapping from technical feature names to readable original column names.

        Returns:
            Dictionary mapping technical feature names to readable names
        """
        if not self.is_fitted:
            raise ValueError("AutoPreprocessor must be fitted first")

        if self._feature_name_mapping is not None:
            return self._feature_name_mapping

        # Create the mapping
        from app.ml.utils.feature_names import create_feature_name_mapping
        technical_names = self.get_feature_names()

        self._feature_name_mapping = create_feature_name_mapping(
            technical_names,
            original_columns=self.original_columns
        )

        return self._feature_name_mapping

    def inverse_transform_target(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions back to original scale.

        Args:
            y_pred: Predictions in transformed space

        Returns:
            Predictions in original space
        """
        if self.target_transform == 'log1p':
            return np.expm1(y_pred)  # Inverse of log1p
        return y_pred

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing steps applied.

        Returns:
            Dictionary with preprocessing summary
        """
        return {
            'steps_applied': self.preprocessing_steps,
            'data_characteristics': self.data_characteristics,
            'final_n_features': len(self.get_feature_names()) if self.is_fitted else 0,
            'outlier_handling': self.outlier_detector is not None,
            'feature_selection': self.feature_selector is not None,
            'target_transform': self.target_transform
        }
