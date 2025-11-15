"""
Data profiling service.
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from app.core.logging import get_logger

logger = get_logger(__name__)


def _convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types.
    Handles inf, -inf, and nan values by converting them to None for JSON compatibility.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    # Check for integer types
    if isinstance(obj, np.integer):
        return int(obj)
    # Check for floating types
    elif isinstance(obj, np.floating):
        val = float(obj)
        # Handle inf, -inf, and nan values for JSON compatibility
        if np.isinf(val) or np.isnan(val):
            return None
        return val
    # Check for boolean types
    elif isinstance(obj, np.bool_):
        return bool(obj)
    # Check for arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Check for dictionaries
    elif isinstance(obj, dict):
        return {str(k): _convert_numpy_types(v) for k, v in obj.items()}
    # Check for lists
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    # Check for Python float types that might be inf/nan
    elif isinstance(obj, float):
        if np.isinf(obj) or np.isnan(obj):
            return None
        return obj
    # Try to convert using item() method for numpy scalars
    elif hasattr(obj, 'item') and isinstance(obj, np.generic):
        try:
            return _convert_numpy_types(obj.item())
        except (ValueError, AttributeError):
            return obj
    # Check if it's a numpy generic type
    elif isinstance(obj, np.generic):
        try:
            val = obj.item()
            # Check if the extracted value is inf/nan
            if isinstance(val, float) and (np.isinf(val) or np.isnan(val)):
                return None
            return val
        except (ValueError, AttributeError):
            return str(obj)
    
    return obj


def _safe_float(value: Any) -> Optional[float]:
    """
    Safely convert a value to float, handling inf and nan values.
    
    Args:
        value: Value to convert
        
    Returns:
        float value or None if inf/nan
    """
    if value is None:
        return None
    try:
        val = float(value)
        if np.isinf(val) or np.isnan(val):
            return None
        return val
    except (ValueError, TypeError):
        return None


class DataProfilingService:
    """Service for generating data profiles."""
    
    def generate_profile(self, data: pd.DataFrame, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate comprehensive data profile.
        
        Args:
            data: DataFrame to profile
            sample_size: Maximum number of rows to profile (None = all)
            
        Returns:
            Profile dictionary
        """
        try:
            # Sample data if needed
            if sample_size and len(data) > sample_size:
                data = data.sample(n=sample_size, random_state=42)
            
            profile = {
                "overview": self._get_overview(data),
                "columns": {},
                "correlations": self._get_correlations(data),
                "missing_data": self._get_missing_data_summary(data)
            }
            
            # Profile each column
            for column in data.columns:
                profile["columns"][column] = self._profile_column(data[column], column)
            
            # Convert any remaining numpy types to Python native types
            return _convert_numpy_types(profile)
        except Exception as e:
            logger.error(f"Error generating profile: {e}", exc_info=True)
            raise
    
    def _get_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get overview statistics."""
        return {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "memory_usage_bytes": int(data.memory_usage(deep=True).sum()),
            "duplicate_rows": int(data.duplicated().sum()),
            "duplicate_percentage": float(data.duplicated().sum() / len(data) * 100) if len(data) > 0 else 0.0
        }
    
    def _profile_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Profile a single column."""
        profile = {
            "name": column_name,
            "dtype": str(series.dtype),
            "nullable": bool(series.isna().any()),
            "null_count": int(series.isna().sum()),
            "null_percentage": float(series.isna().sum() / len(series) * 100) if len(series) > 0 else 0.0,
            "unique_count": int(series.nunique()),
            "unique_percentage": float(series.nunique() / len(series) * 100) if len(series) > 0 else 0.0
        }
        
        # Numeric statistics
        if series.dtype in ['int64', 'float64']:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if not numeric_series.isna().all():
                profile["numeric"] = {
                    "mean": _safe_float(numeric_series.mean()),
                    "median": _safe_float(numeric_series.median()),
                    "std": _safe_float(numeric_series.std()),
                    "min": _safe_float(numeric_series.min()),
                    "max": _safe_float(numeric_series.max()),
                    "q25": _safe_float(numeric_series.quantile(0.25)),
                    "q75": _safe_float(numeric_series.quantile(0.75)),
                    "skewness": _safe_float(numeric_series.skew()),
                    "kurtosis": _safe_float(numeric_series.kurtosis())
                }
            else:
                profile["numeric"] = {
                    "mean": None,
                    "median": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "q25": None,
                    "q75": None,
                    "skewness": None,
                    "kurtosis": None
                }
        
        # Categorical statistics
        if series.dtype == 'object' or series.nunique() < 20:
            value_counts = series.value_counts()
            # Convert numpy types to Python native types
            top_values = {}
            for key, value in value_counts.head(10).items():
                # Convert key and value to Python native types
                if pd.isna(key):
                    key = None
                elif isinstance(key, (np.integer, np.floating)):
                    key = key.item()
                elif isinstance(key, np.bool_):
                    key = bool(key)
                else:
                    key = str(key) if not isinstance(key, (str, int, float, bool)) else key
                
                if isinstance(value, (np.integer, np.floating)):
                    top_values[key] = value.item()
                elif isinstance(value, np.bool_):
                    top_values[key] = bool(value)
                else:
                    top_values[key] = int(value) if isinstance(value, (int, float)) else value
            
            profile["categorical"] = {
                "top_values": top_values,
                "value_counts": int(len(value_counts))
            }
        
        return profile
    
    def _get_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get correlation matrix for numeric columns."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                # Convert correlation matrix to Python native types
                matrix_dict = {}
                for col in corr_matrix.columns:
                    matrix_dict[str(col)] = {
                        str(other_col): _safe_float(corr_matrix.loc[col, other_col])
                        for other_col in corr_matrix.columns
                    }
                return {
                    "matrix": matrix_dict,
                    "high_correlations": self._find_high_correlations(corr_matrix)
                }
            return {"matrix": {}, "high_correlations": []}
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return {"matrix": {}, "high_correlations": []}
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> list:
        """Find highly correlated pairs."""
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    # Convert column names to strings and correlation value to float
                    col1 = str(corr_matrix.columns[i])
                    col2 = str(corr_matrix.columns[j])
                    corr_float = _safe_float(corr_value) or 0.0
                    high_corr.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": corr_float
                    })
        return high_corr
    
    def _get_missing_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of missing data."""
        missing_counts = data.isna().sum()
        missing_percentages = (missing_counts / len(data) * 100).round(2)
        
        # Convert numpy types to Python native types
        missing_by_column = {}
        for col, pct in missing_percentages.items():
            if isinstance(pct, (np.integer, np.floating)):
                missing_by_column[str(col)] = float(pct)
            elif pd.isna(pct):
                missing_by_column[str(col)] = 0.0
            else:
                missing_by_column[str(col)] = float(pct)
        
        return {
            "columns_with_missing": int((missing_counts > 0).sum()),
            "total_missing_values": int(missing_counts.sum()),
            "missing_by_column": missing_by_column
        }

