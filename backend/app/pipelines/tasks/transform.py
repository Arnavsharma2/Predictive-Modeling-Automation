"""
Transform tasks for ETL pipeline.
"""
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

from prefect import task
from app.core.logging import get_logger
from app.services.validation import DataValidationService

logger = get_logger(__name__)


def _transform_dataframe_impl(
    df: pd.DataFrame,
    cleaning_config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Transform DataFrame: cleaning, normalization, feature engineering.
    
    Args:
        df: Input DataFrame
        cleaning_config: Configuration for data cleaning
        
    Returns:
        Transformed DataFrame
    """
    logger.info(f"Transforming DataFrame with {len(df)} rows")
    
    cleaning_config = cleaning_config or {}
    df_transformed = df.copy()
    
    # Handle missing values
    missing_strategy = cleaning_config.get("missing_values_strategy", "drop")
    if missing_strategy == "drop":
        threshold = cleaning_config.get("missing_threshold", 0.5)
        # Drop columns with more than threshold missing values
        cols_to_drop = df_transformed.columns[
            df_transformed.isnull().sum() / len(df_transformed) > threshold
        ]
        if len(cols_to_drop) > 0:
            logger.info(f"Dropping columns with high missing values: {cols_to_drop.tolist()}")
            df_transformed = df_transformed.drop(columns=cols_to_drop)
        
        # Drop rows with all null values
        df_transformed = df_transformed.dropna(how="all")
        
    elif missing_strategy == "fill":
        fill_method = cleaning_config.get("fill_method", "mean")
        if fill_method == "mean":
            numeric_cols = df_transformed.select_dtypes(include=["number"]).columns
            df_transformed[numeric_cols] = df_transformed[numeric_cols].fillna(
                df_transformed[numeric_cols].mean()
            )
        elif fill_method == "median":
            numeric_cols = df_transformed.select_dtypes(include=["number"]).columns
            df_transformed[numeric_cols] = df_transformed[numeric_cols].fillna(
                df_transformed[numeric_cols].median()
            )
        elif fill_method == "mode":
            for col in df_transformed.columns:
                mode_value = df_transformed[col].mode()
                if len(mode_value) > 0:
                    df_transformed[col] = df_transformed[col].fillna(mode_value[0])
        elif fill_method == "forward_fill":
            df_transformed = df_transformed.fillna(method="ffill")
        elif fill_method == "backward_fill":
            df_transformed = df_transformed.fillna(method="bfill")
    
    # Remove duplicates
    if cleaning_config.get("remove_duplicates", True):
        initial_count = len(df_transformed)
        df_transformed = df_transformed.drop_duplicates()
        removed = initial_count - len(df_transformed)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
    
    # Data type conversion
    if "type_conversions" in cleaning_config:
        for col, target_type in cleaning_config["type_conversions"].items():
            if col in df_transformed.columns:
                try:
                    if target_type == "datetime":
                        df_transformed[col] = pd.to_datetime(df_transformed[col], errors="coerce")
                    elif target_type == "numeric":
                        df_transformed[col] = pd.to_numeric(df_transformed[col], errors="coerce")
                    elif target_type == "string":
                        df_transformed[col] = df_transformed[col].astype(str)
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to {target_type}: {e}")
    
    # Normalization (standardize formats, units, etc.)
    if "normalization" in cleaning_config:
        norm_config = cleaning_config["normalization"]
        # Example: normalize date formats, currency, etc.
        # This can be extended based on specific needs
    
    logger.info(f"Transformation complete. Final shape: {df_transformed.shape}")
    return df_transformed


@task(name="transform_dataframe")
def transform_dataframe(
    df: pd.DataFrame,
    cleaning_config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Prefect task wrapper for transform_dataframe."""
    return _transform_dataframe_impl(df, cleaning_config)


def _feature_engineering_impl(
    df: pd.DataFrame,
    feature_config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Create derived features and aggregations.
    
    Args:
        df: Input DataFrame
        feature_config: Configuration for feature engineering
        
    Returns:
        DataFrame with additional features
    """
    logger.info("Performing feature engineering")
    
    feature_config = feature_config or {}
    df_features = df.copy()
    
    # Time-based features (if timestamp column exists)
    timestamp_cols = [col for col in df_features.columns if "time" in col.lower() or "date" in col.lower()]
    for col in timestamp_cols:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df_features[col]):
                df_features[col] = pd.to_datetime(df_features[col], errors="coerce")
            
            df_features[f"{col}_year"] = df_features[col].dt.year
            df_features[f"{col}_month"] = df_features[col].dt.month
            df_features[f"{col}_day"] = df_features[col].dt.day
            df_features[f"{col}_dayofweek"] = df_features[col].dt.dayofweek
        except Exception as e:
            logger.warning(f"Could not create time features from {col}: {e}")
    
    # Statistical features
    numeric_cols = df_features.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0 and feature_config.get("statistical_features", False):
        for col in numeric_cols:
            df_features[f"{col}_zscore"] = (df_features[col] - df_features[col].mean()) / df_features[col].std()
    
    # Interaction features
    if "interactions" in feature_config:
        for col1, col2 in feature_config["interactions"]:
            if col1 in df_features.columns and col2 in df_features.columns:
                if pd.api.types.is_numeric_dtype(df_features[col1]) and pd.api.types.is_numeric_dtype(df_features[col2]):
                    df_features[f"{col1}_x_{col2}"] = df_features[col1] * df_features[col2]
    
    logger.info(f"Feature engineering complete. New shape: {df_features.shape}")
    return df_features


@task(name="feature_engineering")
def feature_engineering(
    df: pd.DataFrame,
    feature_config: Dict[str, Any] = None
) -> pd.DataFrame:
    """Prefect task wrapper for feature_engineering."""
    return _feature_engineering_impl(df, feature_config)


def _validate_data_impl(
    df: pd.DataFrame,
    validation_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Validate transformed data.
    
    Args:
        df: DataFrame to validate
        validation_config: Validation configuration
        
    Returns:
        Validation result dictionary
    """
    logger.info("Validating transformed data")
    
    validation_config = validation_config or {}
    validator = DataValidationService()
    
    result = validator.comprehensive_validation(
        data=df,
        expected_columns=validation_config.get("expected_columns"),
        required_columns=validation_config.get("required_columns"),
        completeness_threshold=validation_config.get("completeness_threshold", 0.95),
        validation_rules=validation_config.get("validation_rules")
    )
    
    if not result["valid"]:
        logger.warning(f"Data validation found issues: {result['all_issues']}")
    else:
        logger.info("Data validation passed")
    
    return result


@task(name="validate_data")
def validate_data(
    df: pd.DataFrame,
    validation_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Prefect task wrapper for validate_data."""
    return _validate_data_impl(df, validation_config)


@task(name="transform_data")
async def transform_data(
    raw_data: Any,
    source_type: str,
    transform_config: Dict[str, Any] = None
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generic transform task that handles different data types.
    
    Args:
        raw_data: Raw extracted data
        source_type: Type of data source
        transform_config: Transformation configuration
        
    Returns:
        Transformed DataFrame
    """
    logger.info(f"Transforming data from {source_type} source")
    
    transform_config = transform_config or {}
    
    # Convert to DataFrame if needed
    if isinstance(raw_data, pd.DataFrame):
        df = raw_data
    elif isinstance(raw_data, dict):
        # API response or single record
        df = pd.DataFrame([raw_data])
    elif isinstance(raw_data, list):
        # List of records
        df = pd.DataFrame(raw_data)
    else:
        raise ValueError(f"Cannot transform data of type {type(raw_data)}")
    
    # Apply transformations (use non-task implementations to avoid Prefect task nesting)
    import asyncio
    df = await asyncio.to_thread(_transform_dataframe_impl, df, transform_config.get("cleaning", {}))
    df = await asyncio.to_thread(_feature_engineering_impl, df, transform_config.get("features", {}))
    
    # Validate
    validation_result = await asyncio.to_thread(_validate_data_impl, df, transform_config.get("validation", {}))
    
    return df, validation_result

