"""
Data quality monitoring service.
"""
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload

from app.models.database.data_sources import DataSource
from app.models.database.data_points import DataPoint
from app.models.database.data_quality_reports import DataQualityReport, QualityStatus
from app.core.logging import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    Compatible with NumPy 2.0+ (avoids deprecated np.float_, np.int_).
    Handles inf, -inf, and nan values by converting them to None for JSON compatibility.
    
    Args:
        obj: Object that may contain NumPy types
        
    Returns:
        Object with NumPy types converted to Python types
    """
    # Handle NumPy scalar types first (using base classes compatible with NumPy 2.0)
    # Check for integer types (avoid np.int_ which was removed in NumPy 2.0)
    if isinstance(obj, np.integer):
        return int(obj)
    # Check for floating types (avoid np.float_ which was removed in NumPy 2.0)
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
    # Handle collections
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    # Handle NaN values
    elif pd.isna(obj):
        return None
    # Check for Python float types that might be inf/nan
    elif isinstance(obj, float):
        if np.isinf(obj) or np.isnan(obj):
            return None
        return obj
    # Try to extract Python value from NumPy scalar using .item() method
    elif hasattr(obj, 'item') and isinstance(obj, np.generic):
        try:
            val = obj.item()
            # Check if the extracted value is inf/nan
            if isinstance(val, float) and (np.isinf(val) or np.isnan(val)):
                return None
            return convert_numpy_types(val)
        except (AttributeError, ValueError, TypeError):
            pass
    # Final check: if it's a NumPy generic type, try to convert it
    if isinstance(obj, np.generic):
        try:
            val = obj.item() if hasattr(obj, 'item') else obj
            # Check if the extracted value is inf/nan
            if isinstance(val, float) and (np.isinf(val) or np.isnan(val)):
                return None
            return val
        except (AttributeError, ValueError, TypeError):
            pass
    return obj


class DataQualityService:
    """Service for monitoring data quality."""
    
    def __init__(self):
        self.quality_thresholds = {
            "completeness": 0.95,  # 95% non-null values
            "uniqueness": 0.99,  # 99% unique values
            "validity": 0.95,  # 95% valid values
            "consistency": 0.90,  # 90% consistent values
            "timeliness": 24  # Data should be updated within 24 hours
        }
    
    async def check_data_quality(
        self,
        db: AsyncSession,
        data_source_id: int,
        data: Optional[pd.DataFrame] = None,
        created_by: Optional[int] = None
    ) -> DataQualityReport:
        """
        Perform comprehensive data quality checks.
        
        Args:
            db: Database session
            data_source_id: ID of the data source to check
            data: DataFrame to check (if None, fetches from database)
            created_by: User ID who triggered the check
            
        Returns:
            DataQualityReport object
        """
        try:
            # Get data source
            data_source_result = await db.execute(
                select(DataSource).where(DataSource.id == data_source_id)
            )
            data_source = data_source_result.scalar_one_or_none()
            
            if not data_source:
                raise ValueError(f"Data source {data_source_id} not found")
            
            # Get data if not provided
            if data is None:
                data = await self._get_data_from_source(db, data_source_id)
            
            if data is None or len(data) == 0:
                raise ValueError("No data available for quality check")
            
            # Perform quality checks
            quality_results = self._perform_quality_checks(data)
            
            # Check data freshness
            freshness_check = await self._check_data_freshness(db, data_source_id)
            
            # Determine overall quality status
            overall_status = self._determine_quality_status(quality_results, freshness_check)
            
            # Convert NumPy types to Python types for JSON serialization
            quality_results = convert_numpy_types(quality_results)
            freshness_check = convert_numpy_types(freshness_check)
            schema_info = convert_numpy_types(self._get_schema_info(data))
            
            # Create quality report
            quality_report = DataQualityReport(
                data_source_id=data_source_id,
                status=overall_status,
                quality_metrics=quality_results,
                freshness_metrics=freshness_check,
                schema_info=schema_info,
                sample_size=len(data),
                created_by=created_by
            )
            
            db.add(quality_report)
            await db.commit()
            await db.refresh(quality_report)
            
            return quality_report
        except Exception as e:
            logger.error(f"Error checking data quality: {e}", exc_info=True)
            raise
    
    def _perform_quality_checks(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform various data quality checks."""
        results = {
            "completeness": {},
            "uniqueness": {},
            "validity": {},
            "consistency": {},
            "summary": {}
        }
        
        total_rows = len(data)
        
        for column in data.columns:
            col_data = data[column]
            
            # Completeness check
            null_count = col_data.isna().sum()
            completeness = 1.0 - (null_count / total_rows) if total_rows > 0 else 0.0
            results["completeness"][column] = {
                "score": float(completeness),
                "null_count": int(null_count),
                "null_percentage": float(null_count / total_rows * 100) if total_rows > 0 else 0.0,
                "pass": bool(completeness >= self.quality_thresholds["completeness"])
            }
            
            # Uniqueness check
            unique_count = col_data.nunique()
            uniqueness = unique_count / total_rows if total_rows > 0 else 0.0
            results["uniqueness"][column] = {
                "score": float(uniqueness),
                "unique_count": int(unique_count),
                "duplicate_count": int(total_rows - unique_count),
                "pass": bool(uniqueness >= self.quality_thresholds["uniqueness"])
            }
            
            # Validity check (basic type checking)
            validity_score = self._check_validity(col_data)
            results["validity"][column] = validity_score
            
            # Consistency check (check for outliers and anomalies)
            consistency_score = self._check_consistency(col_data)
            results["consistency"][column] = consistency_score
        
        # Calculate summary metrics (convert to Python types immediately)
        avg_completeness = float(np.mean([r["score"] for r in results["completeness"].values()]))
        avg_uniqueness = float(np.mean([r["score"] for r in results["uniqueness"].values()]))
        avg_validity = float(np.mean([r["score"] for r in results["validity"].values()]))
        avg_consistency = float(np.mean([r["score"] for r in results["consistency"].values()]))
        
        results["summary"] = {
            "overall_score": float((avg_completeness + avg_uniqueness + avg_validity + avg_consistency) / 4),
            "completeness_score": float(avg_completeness),
            "uniqueness_score": float(avg_uniqueness),
            "validity_score": float(avg_validity),
            "consistency_score": float(avg_consistency),
            "total_rows": int(total_rows),
            "total_columns": int(len(data.columns)),
            "columns_passing": int(sum(1 for r in results["completeness"].values() if r["pass"]))
        }
        
        return results
    
    def _check_validity(self, series: pd.Series) -> Dict[str, Any]:
        """Check validity of a column."""
        try:
            # Basic type validation
            if series.dtype == 'object':
                # Check for empty strings
                empty_strings = (series == '').sum()
                valid_count = len(series) - series.isna().sum() - empty_strings
            else:
                # For numeric types, check for inf and -inf
                numeric_series = pd.to_numeric(series, errors='coerce')
                invalid_count = (numeric_series.isna() | np.isinf(numeric_series)).sum()
                valid_count = len(series) - invalid_count
            
            validity_score = valid_count / len(series) if len(series) > 0 else 0.0
            
            return {
                "score": float(validity_score),
                "valid_count": int(valid_count),
                "invalid_count": int(len(series) - valid_count),
                "pass": bool(validity_score >= self.quality_thresholds["validity"])
            }
        except Exception as e:
            logger.error(f"Error checking validity: {e}")
            return {
                "score": 0.0,
                "valid_count": 0,
                "invalid_count": len(series),
                "pass": False,
                "error": str(e)
            }
    
    def _check_consistency(self, series: pd.Series) -> Dict[str, Any]:
        """Check consistency of a column."""
        try:
            if series.dtype in ['int64', 'float64']:
                # Check for outliers using IQR method
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
                else:
                    outliers = 0
                
                consistency_score = 1.0 - (outliers / len(series)) if len(series) > 0 else 1.0
                
                return {
                    "score": float(consistency_score),
                    "outlier_count": int(outliers),
                    "outlier_percentage": float(outliers / len(series) * 100) if len(series) > 0 else 0.0,
                    "pass": bool(consistency_score >= self.quality_thresholds["consistency"])
                }
            else:
                # For categorical data, check for unexpected values
                # This is a simplified check - can be enhanced
                return {
                    "score": 1.0,
                    "outlier_count": 0,
                    "outlier_percentage": 0.0,
                    "pass": True
                }
        except Exception as e:
            logger.error(f"Error checking consistency: {e}")
            return {
                "score": 1.0,
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "pass": True,
                "error": str(e)
            }
    
    def _get_schema_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get schema information for the data."""
        schema = {}
        for column in data.columns:
            schema[column] = {
                "dtype": str(data[column].dtype),
                "nullable": bool(data[column].isna().any()),
                "sample_values": data[column].dropna().head(5).tolist() if len(data[column].dropna()) > 0 else []
            }
        return schema
    
    async def _check_data_freshness(
        self,
        db: AsyncSession,
        data_source_id: int
    ) -> Dict[str, Any]:
        """Check data freshness."""
        try:
            # Get the most recent data point
            latest_point_result = await db.execute(
                select(DataPoint)
                .where(DataPoint.source_id == data_source_id)
                .order_by(DataPoint.timestamp.desc())
                .limit(1)
            )
            latest_point = latest_point_result.scalar_one_or_none()
            
            if not latest_point:
                return {
                    "is_fresh": False,
                    "last_update": None,
                    "hours_since_update": None,
                    "pass": False,
                    "message": "No data points found"
                }
            
            last_update = latest_point.timestamp
            hours_since_update = (datetime.now(timezone.utc) - last_update.replace(tzinfo=None)).total_seconds() / 3600
            
            is_fresh = bool(hours_since_update <= self.quality_thresholds["timeliness"])
            
            return {
                "is_fresh": is_fresh,
                "last_update": last_update.isoformat() if last_update else None,
                "hours_since_update": float(hours_since_update),
                "pass": is_fresh,
                "threshold_hours": self.quality_thresholds["timeliness"]
            }
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}", exc_info=True)
            return {
                "is_fresh": False,
                "last_update": None,
                "hours_since_update": None,
                "pass": False,
                "error": str(e)
            }
    
    def _determine_quality_status(
        self,
        quality_results: Dict[str, Any],
        freshness_check: Dict[str, Any]
    ) -> QualityStatus:
        """Determine overall quality status."""
        summary = quality_results.get("summary", {})
        overall_score = summary.get("overall_score", 0.0)
        
        # Check if freshness passes
        freshness_pass = freshness_check.get("pass", False)
        
        # Determine status
        if overall_score >= 0.95 and freshness_pass:
            return QualityStatus.EXCELLENT
        elif overall_score >= 0.85 and freshness_pass:
            return QualityStatus.GOOD
        elif overall_score >= 0.70:
            return QualityStatus.FAIR
        elif overall_score >= 0.50:
            return QualityStatus.POOR
        else:
            return QualityStatus.CRITICAL
    
    async def _get_data_from_source(
        self,
        db: AsyncSession,
        data_source_id: int,
        limit: int = 10000
    ) -> Optional[pd.DataFrame]:
        """Get data from data source."""
        try:
            data_points_result = await db.execute(
                select(DataPoint)
                .where(DataPoint.source_id == data_source_id)
                .order_by(DataPoint.timestamp.desc())
                .limit(limit)
            )
            data_points = data_points_result.scalars().all()
            
            if not data_points:
                return None
            
            data_list = []
            for point in data_points:
                if isinstance(point.data, dict):
                    data_list.append(point.data)
            
            if data_list:
                return pd.DataFrame(data_list)
            
            return None
        except Exception as e:
            logger.error(f"Error getting data from source: {e}", exc_info=True)
            return None
    
    async def validate_schema(
        self,
        data: pd.DataFrame,
        expected_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate data against expected schema.
        
        Args:
            data: DataFrame to validate
            expected_schema: Expected schema definition
            
        Returns:
            Validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required columns
        required_columns = expected_schema.get("required_columns", [])
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            results["valid"] = False
            results["errors"].append(f"Missing required columns: {missing_columns}")
        
        # Check column types
        column_types = expected_schema.get("column_types", {})
        for column, expected_type in column_types.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if actual_type != expected_type:
                    results["warnings"].append(
                        f"Column {column} has type {actual_type}, expected {expected_type}"
                    )
        
        return results

