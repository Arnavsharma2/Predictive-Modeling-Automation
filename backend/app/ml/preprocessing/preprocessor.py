"""
Data preprocessing for ML models.
"""
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer, KNNImputer
# IterativeImputer is experimental and requires explicit enable
try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    ITERATIVE_IMPUTER_AVAILABLE = True
except ImportError:
    ITERATIVE_IMPUTER_AVAILABLE = False
    IterativeImputer = None

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import re
from datetime import datetime
from app.core.logging import get_logger

warnings.filterwarnings('ignore')

logger = get_logger(__name__)


class MissingIndicatorTransformer(BaseEstimator, TransformerMixin):
    """Add binary indicators for missing values."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Add missing indicators."""
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)

        # Create missing indicators
        missing_indicators = pd.DataFrame(
            X_df.isnull().astype(int),
            columns=[f"{col}_is_missing" for col in X_df.columns],
            index=X_df.index
        )

        # Combine original data with indicators
        result = pd.concat([X_df, missing_indicators], axis=1)
        return result.values if not isinstance(X, pd.DataFrame) else result


class DateTimeTransformer(BaseEstimator, TransformerMixin):
    """Transform datetime columns into multiple time-based features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transform datetime columns to time features."""
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        result_dfs = []

        for col_idx in range(X_df.shape[1]):
            # Get column data
            if isinstance(X_df, pd.DataFrame):
                col_data = X_df.iloc[:, col_idx]
            else:
                col_data = X_df[:, col_idx]

            # Convert to datetime
            try:
                dt_series = pd.to_datetime(col_data, errors='coerce')
            except:
                # If conversion fails, just pass through as numeric
                result_dfs.append(np.array(col_data).reshape(-1, 1))
                continue

            # Extract features
            features = []
            features.append(dt_series.dt.year.fillna(0).values)
            features.append(dt_series.dt.month.fillna(0).values)
            features.append(dt_series.dt.day.fillna(0).values)
            features.append(dt_series.dt.dayofweek.fillna(0).values)
            features.append(dt_series.dt.hour.fillna(0).values)
            features.append(dt_series.dt.quarter.fillna(0).values)
            # Cyclical encoding
            features.append(np.sin(2 * np.pi * dt_series.dt.month.fillna(0) / 12).values)
            features.append(np.cos(2 * np.pi * dt_series.dt.month.fillna(0) / 12).values)
            features.append(np.sin(2 * np.pi * dt_series.dt.dayofweek.fillna(0) / 7).values)
            features.append(np.cos(2 * np.pi * dt_series.dt.dayofweek.fillna(0) / 7).values)

            result_dfs.append(np.column_stack(features))

        return np.hstack(result_dfs) if result_dfs else np.array(X_df)


class TextTransformer(BaseEstimator, TransformerMixin):
    """Transform text columns into numeric features with enhanced TF-IDF and n-grams."""

    def __init__(self, max_features: int = 100, ngram_range: tuple = (1, 2), use_char_ngrams: bool = False):
        """
        Initialize TextTransformer.
        
        Args:
            max_features: Maximum number of TF-IDF features per column
            ngram_range: Range of n-grams to extract (e.g., (1, 2) for unigrams and bigrams)
            use_char_ngrams: Whether to also use character n-grams
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_char_ngrams = use_char_ngrams
        self.tfidf_vectorizers = []
        self.char_ngram_vectorizers = []

    def fit(self, X, y=None):
        """Fit TF-IDF vectorizers for each text column."""
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X

        self.tfidf_vectorizers = []
        self.char_ngram_vectorizers = []
        
        for col_idx in range(X_df.shape[1]):
            # Get column data
            if isinstance(X_df, pd.DataFrame):
                col_data = X_df.iloc[:, col_idx].fillna('').astype(str)
            else:
                col_data = pd.Series(X_df[:, col_idx]).fillna('').astype(str)

            # Word-level TF-IDF with n-grams
            vectorizer = TfidfVectorizer(
                max_features=min(self.max_features, 100),
                stop_words='english',
                ngram_range=self.ngram_range,
                min_df=2,
                max_df=0.95,  # Ignore terms that appear in more than 95% of documents
                lowercase=True,
                analyzer='word'
            )
            try:
                vectorizer.fit(col_data)
                self.tfidf_vectorizers.append(vectorizer)
            except Exception as e:
                # If TF-IDF fails, use None to indicate basic features only
                self.tfidf_vectorizers.append(None)

            # Character-level n-grams (optional, for short texts or when word-level fails)
            if self.use_char_ngrams:
                char_vectorizer = TfidfVectorizer(
                    max_features=min(self.max_features // 2, 50),
                    ngram_range=(2, 4),  # Character bigrams, trigrams, 4-grams
                    min_df=2,
                    analyzer='char',
                    lowercase=True
                )
                try:
                    char_vectorizer.fit(col_data)
                    self.char_ngram_vectorizers.append(char_vectorizer)
                except:
                    self.char_ngram_vectorizers.append(None)
            else:
                self.char_ngram_vectorizers.append(None)

        return self

    def transform(self, X):
        """Transform text columns to numeric features."""
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        result_features = []

        for col_idx in range(X_df.shape[1]):
            # Get column data
            if isinstance(X_df, pd.DataFrame):
                col_data = X_df.iloc[:, col_idx].fillna('').astype(str)
            else:
                col_data = pd.Series(X_df[:, col_idx]).fillna('').astype(str)

            # Enhanced basic text features
            text_length = col_data.str.len().values.reshape(-1, 1)
            word_count = col_data.str.split().str.len().fillna(0).values.reshape(-1, 1)
            char_count = col_data.str.len().values.reshape(-1, 1)
            
            # Additional text statistics
            sentence_count = col_data.str.count(r'[.!?]+').values.reshape(-1, 1)
            avg_word_length = (col_data.str.len() / (word_count.flatten() + 1e-8)).values.reshape(-1, 1)
            
            # Special character counts
            digit_count = col_data.str.count(r'\d').values.reshape(-1, 1)
            uppercase_count = col_data.str.count(r'[A-Z]').values.reshape(-1, 1)
            special_char_count = col_data.str.count(r'[^\w\s]').values.reshape(-1, 1)

            features = [
                text_length, word_count, char_count, sentence_count,
                avg_word_length, digit_count, uppercase_count, special_char_count
            ]

            # Word-level TF-IDF features if available
            if col_idx < len(self.tfidf_vectorizers) and self.tfidf_vectorizers[col_idx] is not None:
                try:
                    tfidf_features = self.tfidf_vectorizers[col_idx].transform(col_data).toarray()
                    features.append(tfidf_features)
                except Exception:
                    pass

            # Character-level n-grams if enabled
            if (self.use_char_ngrams and 
                col_idx < len(self.char_ngram_vectorizers) and 
                self.char_ngram_vectorizers[col_idx] is not None):
                try:
                    char_features = self.char_ngram_vectorizers[col_idx].transform(col_data).toarray()
                    features.append(char_features)
                except Exception:
                    pass

            result_features.append(np.hstack(features))

        return np.hstack(result_features) if result_features else np.array(X_df)


class DataPreprocessor:
    """Data preprocessing pipeline."""
    
    def __init__(
        self,
        target_column: Optional[str] = None,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        datetime_columns: Optional[List[str]] = None,
        text_columns: Optional[List[str]] = None,
        handle_missing: str = "mean",  # mean, median, mode, drop, knn, iterative
        add_missing_indicators: bool = False,  # Add binary flags for missing values
        scale_numeric: bool = True,
        scaling_method: str = "standard",  # standard, minmax
        encode_categorical: str = "onehot",  # onehot, label, target, frequency, leave_one_out, james_stein
        drop_columns: Optional[List[str]] = None,
        auto_detect_types: bool = True
    ):
        """
        Initialize preprocessor.

        Args:
            target_column: Name of target column
            numeric_columns: List of numeric column names
            categorical_columns: List of categorical column names
            datetime_columns: List of datetime column names
            text_columns: List of text column names
            handle_missing: Strategy for handling missing values
            scale_numeric: Whether to scale numeric features
            scaling_method: Scaling method (standard or minmax)
            encode_categorical: Encoding method for categorical features
            drop_columns: Columns to drop
            auto_detect_types: Automatically detect column types
        """
        self.target_column = target_column
        self.numeric_columns = numeric_columns or []
        self.categorical_columns = categorical_columns or []
        self.datetime_columns = datetime_columns or []
        self.text_columns = text_columns or []
        self.handle_missing = handle_missing
        self.add_missing_indicators = add_missing_indicators
        self.scale_numeric = scale_numeric
        self.scaling_method = scaling_method
        self.encode_categorical = encode_categorical
        self.drop_columns = drop_columns or []
        self.auto_detect_types = auto_detect_types

        self.preprocessor = None
        self.feature_names = None
        self.is_fitted = False
        self.detected_types: Dict[str, str] = {}
        self.missing_indicator_columns = []
        self.target_encoder = None  # For target encoding of categorical columns
        self.target_encoded_categorical_cols = []  # Track which categorical cols were target-encoded
    
    def _is_id_column(self, col: str, df: pd.DataFrame) -> bool:
        """Check if column is likely an ID column that should be excluded."""
        col_lower = col.lower()

        # Check column name patterns
        id_patterns = ['id', '_id', 'customer_id', 'customerid', 'user_id', 'userid', 'transaction_id']
        if any(pattern in col_lower for pattern in id_patterns):
            # Additional check: ID columns usually have high cardinality
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:  # More than 90% unique values
                return True

        return False

    def _is_datetime_column(self, col: str, df: pd.DataFrame) -> bool:
        """Check if column is a datetime column."""
        # Already datetime type
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return True

        # Check column name patterns
        col_lower = col.lower()
        datetime_patterns = ['date', 'time', 'timestamp', 'created', 'updated', 'modified', '_at', '_on']
        if any(pattern in col_lower for pattern in datetime_patterns):
            # Try to parse as datetime
            try:
                pd.to_datetime(df[col].dropna().head(100), errors='coerce')
                return True
            except:
                pass

        # Try to infer from data (sample first 100 rows)
        if df[col].dtype == 'object':
            try:
                sample = df[col].dropna().head(100)
                parsed = pd.to_datetime(sample, errors='coerce')
                # If more than 80% successfully parsed, consider it datetime
                if parsed.notna().sum() / len(sample) > 0.8:
                    return True
            except:
                pass

        return False

    def _is_text_column(self, col: str, df: pd.DataFrame) -> bool:
        """Check if column contains text data (not categorical)."""
        if df[col].dtype != 'object':
            return False

        # Get non-null values
        non_null = df[col].dropna()
        if len(non_null) == 0:
            return False

        # Check average string length (text columns typically have longer strings)
        avg_length = non_null.astype(str).str.len().mean()
        if avg_length > 50:  # Text columns usually have longer strings
            return True

        # Check unique ratio (text columns have high cardinality)
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.5 and avg_length > 20:
            return True

        # Check for text patterns (emails, URLs, etc.)
        sample = non_null.head(100).astype(str)
        has_email = sample.str.contains('@', regex=False).any()
        has_url = sample.str.contains('http', regex=False).any()
        has_long_words = (sample.str.len() > 100).any()

        if has_email or has_url or has_long_words:
            return True

        return False

    def _is_boolean_column(self, col: str, df: pd.DataFrame) -> bool:
        """Check if column is boolean."""
        if pd.api.types.is_bool_dtype(df[col]):
            return True

        # Check for binary values
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:
            # Check if values are boolean-like
            val_set = {str(v).lower() for v in unique_vals}
            boolean_sets = [
                {'true', 'false'},
                {'yes', 'no'},
                {'y', 'n'},
                {'1', '0'},
                {'1.0', '0.0'},
                {'t', 'f'}
            ]
            if val_set in boolean_sets:
                return True

        return False

    def _infer_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Infer numeric, categorical, datetime, and text columns."""
        numeric = list(self.numeric_columns) if self.numeric_columns else []
        categorical = list(self.categorical_columns) if self.categorical_columns else []
        datetime_cols = list(self.datetime_columns) if self.datetime_columns else []
        text_cols = list(self.text_columns) if self.text_columns else []

        # If auto-detection is disabled and all types are specified, return as-is
        if not self.auto_detect_types:
            return numeric, categorical, datetime_cols, text_cols

        for col in df.columns:
            if col == self.target_column:
                continue
            if col in self.drop_columns:
                continue

            # Skip if already classified
            if col in numeric or col in categorical or col in datetime_cols or col in text_cols:
                continue

            # Automatically exclude ID-like columns
            if self._is_id_column(col, df):
                self.drop_columns.append(col)
                self.detected_types[col] = 'id'
                continue

            # Detect datetime columns
            if self._is_datetime_column(col, df):
                datetime_cols.append(col)
                self.detected_types[col] = 'datetime'
                continue

            # Detect text columns
            if self._is_text_column(col, df):
                text_cols.append(col)
                self.detected_types[col] = 'text'
                continue

            # Detect boolean columns (treat as categorical)
            if self._is_boolean_column(col, df):
                categorical.append(col)
                self.detected_types[col] = 'boolean'
                continue

            # Detect numeric columns
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int16', 'float16']:
                # Check if it's discrete numeric that should be categorical
                unique_count = df[col].nunique()
                if unique_count <= 10 and df[col].dtype in ['int64', 'int32', 'int16']:
                    # Low cardinality integer - could be categorical
                    categorical.append(col)
                    self.detected_types[col] = 'categorical_discrete'
                else:
                    numeric.append(col)
                    self.detected_types[col] = 'numeric'
                continue

            # Remaining object columns - check if categorical
            if df[col].dtype == 'object':
                unique_count = df[col].nunique()
                unique_ratio = unique_count / len(df)

                # High cardinality - drop or treat specially
                if unique_count > 100 or unique_ratio > 0.5:
                    # For very high cardinality, we'll handle this in encoding
                    self.detected_types[col] = 'high_cardinality'
                    # Don't drop - let encoding strategies handle it
                    categorical.append(col)
                elif unique_count > 50 or unique_ratio > 0.3:
                    self.detected_types[col] = 'medium_cardinality'
                    categorical.append(col)
                else:
                    categorical.append(col)
                    self.detected_types[col] = 'categorical'

        return numeric, categorical, datetime_cols, text_cols
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.

        Args:
            df: Training dataframe

        Returns:
            Self
        """
        # Infer column types if not specified
        numeric_cols, categorical_cols, datetime_cols, text_cols = self._infer_column_types(df)

        # Store detected columns for later use
        self.numeric_columns = numeric_cols
        self.categorical_columns = categorical_cols
        self.datetime_columns = datetime_cols
        self.text_columns = text_cols

        if not numeric_cols and not categorical_cols and not datetime_cols and not text_cols:
            raise ValueError("No features found for preprocessing")
        
        # Build preprocessing transformers
        transformers = []
        
        # Numeric preprocessing
        if numeric_cols:
            numeric_steps = []
            
            # Add missing indicators BEFORE imputation if requested
            if self.add_missing_indicators:
                # Store column names for missing indicators
                self.missing_indicator_columns = [f"{col}_is_missing" for col in numeric_cols]
                # Add missing indicator transformer
                numeric_steps.append(('missing_indicator', MissingIndicatorTransformer()))
            
            # Handle missing values
            if self.handle_missing in ["mean", "median"]:
                numeric_steps.append(('imputer', SimpleImputer(strategy=self.handle_missing)))
            elif self.handle_missing == "mode":
                numeric_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            elif self.handle_missing == "knn":
                # KNN imputation (uses k=5 neighbors by default)
                numeric_steps.append(('imputer', KNNImputer(n_neighbors=5)))
            elif self.handle_missing == "iterative":
                # Iterative imputation (uses Bayesian Ridge by default)
                if not ITERATIVE_IMPUTER_AVAILABLE:
                    logger.warning("IterativeImputer not available, falling back to KNN imputation")
                    numeric_steps.append(('imputer', KNNImputer(n_neighbors=5)))
                else:
                    numeric_steps.append(('imputer', IterativeImputer(random_state=42, max_iter=10)))
            # drop is handled separately
            
            # Scaling
            if self.scale_numeric:
                if self.scaling_method == "standard":
                    numeric_steps.append(('scaler', StandardScaler()))
                elif self.scaling_method == "minmax":
                    numeric_steps.append(('scaler', MinMaxScaler()))
            
            if numeric_steps:
                transformers.append(('numeric', Pipeline(numeric_steps), numeric_cols))
            else:
                transformers.append(('numeric', 'passthrough', numeric_cols))
        
        # DateTime preprocessing
        if datetime_cols:
            transformers.append(('datetime', DateTimeTransformer(), datetime_cols))

        # Text preprocessing
        if text_cols:
            transformers.append(('text', TextTransformer(max_features=100), text_cols))

        # Categorical preprocessing
        # Handle target-dependent encodings separately (they need y)
        if categorical_cols:
            if self.encode_categorical == "onehot":
                transformers.append(('categorical', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols))
            elif self.encode_categorical == "label":
                transformers.append(('categorical', LabelEncoder(), categorical_cols))
            elif self.encode_categorical == "frequency":
                from .feature_engineering import FrequencyEncoder
                transformers.append(('categorical', FrequencyEncoder(), categorical_cols))
            elif self.encode_categorical in ["target", "leave_one_out", "james_stein"]:
                # These require target y, so we'll handle them separately before ColumnTransformer
                # Store categorical cols for later encoding
                self.target_encoded_categorical_cols = categorical_cols
                # Don't add to transformers - we'll encode them manually
            else:
                transformers.append(('categorical', 'passthrough', categorical_cols))
        
        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            sparse_threshold=0
        )
        
        # Prepare data for fitting
        X = df.drop(columns=[self.target_column] if self.target_column else [])
        if self.drop_columns:
            X = X.drop(columns=self.drop_columns, errors='ignore')
        
        # Extract target for target-dependent encodings
        y_target = None
        if self.target_column and self.target_column in df.columns:
            y_target = df[self.target_column]
        
        # Handle target-dependent categorical encoding BEFORE ColumnTransformer
        if self.target_encoded_categorical_cols and y_target is not None:
            from .feature_engineering import TargetEncoder, LeaveOneOutEncoder, JamesSteinEncoder
            
            # Create encoder based on method
            if self.encode_categorical == "target":
                self.target_encoder = TargetEncoder()
            elif self.encode_categorical == "leave_one_out":
                self.target_encoder = LeaveOneOutEncoder()
            elif self.encode_categorical == "james_stein":
                self.target_encoder = JamesSteinEncoder()
            
            # Fit encoder on categorical columns
            if self.target_encoder:
                categorical_data = X[self.target_encoded_categorical_cols]
                self.target_encoder.fit(categorical_data, y_target)
                
                # Transform categorical columns to numeric
                X_encoded = self.target_encoder.transform(categorical_data)
                
                # Replace categorical columns with encoded numeric values
                # Convert to DataFrame if needed
                if isinstance(X_encoded, np.ndarray):
                    # Handle case where X_encoded might be 1D or 2D
                    if X_encoded.ndim == 1:
                        X_encoded = X_encoded.reshape(-1, 1)
                    encoded_df = pd.DataFrame(
                        X_encoded,
                        columns=[f"{col}_encoded" for col in self.target_encoded_categorical_cols],
                        index=X.index
                    )
                else:
                    encoded_df = pd.DataFrame(X_encoded, index=X.index)
                
                # Drop original categorical columns and add encoded ones
                X = X.drop(columns=self.target_encoded_categorical_cols)
                X = pd.concat([X, encoded_df], axis=1)
                
                # Update numeric_cols to include encoded categorical columns
                numeric_cols = numeric_cols + list(encoded_df.columns)
                # Remove categorical_cols from the list since they're now numeric
                categorical_cols = [col for col in categorical_cols if col not in self.target_encoded_categorical_cols]
        
        # Handle missing values by dropping rows if needed
        if self.handle_missing == "drop":
            X = X.dropna()
        
        # Fit preprocessor
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        self.feature_names = self._get_feature_names(numeric_cols, categorical_cols)
        
        self.is_fitted = True
        return self
    
    def _get_feature_names(self, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
        """Get feature names after preprocessing."""
        feature_names = []

        # Numeric features
        feature_names.extend(numeric_cols)

        # Categorical features
        if categorical_cols and self.encode_categorical == "onehot":
            # Get one-hot encoded feature names
            for col in categorical_cols:
                # This is approximate - actual names depend on unique values
                # In practice, we'd need to fit the encoder first
                feature_names.append(f"{col}_encoded")
        elif categorical_cols and self.encode_categorical in ["label", "frequency"]:
            # Label and frequency encoding preserve column names
            feature_names.extend(categorical_cols)
        elif self.target_encoded_categorical_cols:
            # Target-encoded categorical columns are now numeric with _encoded suffix
            feature_names.extend([f"{col}_encoded" for col in self.target_encoded_categorical_cols])

        return feature_names
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming")

        # Prepare data
        X = df.copy()
        if self.target_column and self.target_column in X.columns:
            X = X.drop(columns=[self.target_column])
        if self.drop_columns:
            X = X.drop(columns=self.drop_columns, errors='ignore')
        
        # Apply target-dependent encoding to categorical columns if needed
        if self.target_encoder and self.target_encoded_categorical_cols:
            # Check which categorical columns actually exist in the data
            existing_cat_cols = [col for col in self.target_encoded_categorical_cols if col in X.columns]
            
            if existing_cat_cols:
                # Extract categorical columns
                categorical_data = X[existing_cat_cols]
                
                # Transform using fitted encoder
                X_encoded = self.target_encoder.transform(categorical_data)
                
                # Convert to DataFrame if needed
                if isinstance(X_encoded, np.ndarray):
                    # Handle case where X_encoded might be 1D or 2D
                    if X_encoded.ndim == 1:
                        X_encoded = X_encoded.reshape(-1, 1)
                    encoded_df = pd.DataFrame(
                        X_encoded,
                        columns=[f"{col}_encoded" for col in existing_cat_cols],
                        index=X.index
                    )
                else:
                    encoded_df = pd.DataFrame(X_encoded, index=X.index)
                
                # Drop original categorical columns and add encoded ones
                X = X.drop(columns=existing_cat_cols)
                X = pd.concat([X, encoded_df], axis=1)
        
        # Handle missing values
        if self.handle_missing == "drop":
            X = X.dropna()
        
        # Transform
        X_processed = self.preprocessor.transform(X)
        
        # Convert to dataframe with feature names
        if isinstance(X_processed, np.ndarray):
            # Get actual feature names from transformer
            try:
                feature_names = list(self.preprocessor.get_feature_names_out())
            except AttributeError:
                # Fallback to numeric feature count
                feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

            return pd.DataFrame(X_processed, columns=feature_names, index=X.index)

        return pd.DataFrame(X_processed)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")

        try:
            # Use sklearn's get_feature_names_out but fall back gracefully
            return list(self.preprocessor.get_feature_names_out())
        except AttributeError:
            return self.feature_names or []

