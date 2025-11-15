"""
Drift monitoring service.
"""
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from app.models.database.drift_reports import DriftReport, DriftType, DriftSeverity
from app.models.database.ml_models import MLModel
from app.models.database.data_sources import DataSource
from app.models.database.data_points import DataPoint
from app.models.database.alerts import Alert, AlertType, AlertSeverity, AlertStatus
from app.ml.drift.drift_detector import DriftDetector
from app.ml.storage.model_storage import model_storage
from app.services.alerting.alert_service import AlertService
from app.core.logging import get_logger
import pandas as pd
import numpy as np

logger = get_logger(__name__)


class DriftMonitoringService:
    """Service for monitoring and detecting drift."""
    
    def __init__(self):
        self.drift_detector = DriftDetector()
        self.alert_service = AlertService()
    
    async def check_drift(
        self,
        db: AsyncSession,
        model_id: int,
        current_data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
        features: Optional[List[str]] = None,
        created_by: Optional[int] = None
    ) -> DriftReport:
        """
        Check for drift for a specific model.
        
        Args:
            db: Database session
            model_id: ID of the model to check
            current_data: Current production data (raw, will be preprocessed)
            reference_data: Reference training data (if None, will fetch from model)
            features: Features to check (if None, uses model features)
            created_by: User ID who triggered the check
            
        Returns:
            DriftReport object
        """
        try:
            # Get model
            model_result = await db.execute(
                select(MLModel).where(MLModel.id == model_id)
            )
            model = model_result.scalar_one_or_none()
            
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            # Load model package to get preprocessor
            preprocessor = None
            if model.model_path:
                try:
                    model_package = await model_storage.load_model(model.model_path, return_package=True)
                    if isinstance(model_package, dict):
                        preprocessor = model_package.get("preprocessor")
                        logger.info(f"Loaded preprocessor from model package for model {model_id}")
                except Exception as e:
                    logger.warning(f"Could not load preprocessor for model {model_id}: {e}. Proceeding without preprocessing.")
            
            # Get reference data if not provided
            if reference_data is None:
                reference_data = await self._get_reference_data(db, model)
            
            if reference_data is None or len(reference_data) == 0:
                raise ValueError("No reference data available for drift detection")
            
            # Apply preprocessing to both reference and current data if preprocessor is available
            if preprocessor is not None:
                logger.info("Applying preprocessing to reference and current data for drift detection")
                
                # Ensure data has the original columns expected by the preprocessor
                if model.original_columns:
                    # Reorder reference data to match original columns
                    missing_ref_cols = [col for col in model.original_columns if col not in reference_data.columns]
                    if missing_ref_cols:
                        logger.warning(f"Missing columns in reference data: {missing_ref_cols}. Filling with NaN.")
                        for col in missing_ref_cols:
                            reference_data[col] = np.nan
                    reference_data = reference_data[model.original_columns]
                    
                    # Reorder current data to match original columns
                    missing_curr_cols = [col for col in model.original_columns if col not in current_data.columns]
                    if missing_curr_cols:
                        logger.warning(f"Missing columns in current data: {missing_curr_cols}. Filling with NaN.")
                        for col in missing_curr_cols:
                            current_data[col] = np.nan
                    current_data = current_data[model.original_columns]
                
                # Infer and convert data types (numeric columns to float, categorical stay as object/string)
                for col in reference_data.columns:
                    try:
                        reference_data[col] = pd.to_numeric(reference_data[col], errors='ignore')
                    except:
                        pass  # Keep as string/object for categorical columns
                
                for col in current_data.columns:
                    try:
                        current_data[col] = pd.to_numeric(current_data[col], errors='ignore')
                    except:
                        pass  # Keep as string/object for categorical columns
                
                # Apply preprocessing transformation
                try:
                    reference_data = preprocessor.transform(reference_data)
                    current_data = preprocessor.transform(current_data)
                    
                    # Get feature names from preprocessor (try multiple methods)
                    feature_names = None
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        try:
                            feature_names = list(preprocessor.get_feature_names_out())
                        except:
                            pass
                    elif hasattr(preprocessor, 'get_feature_names'):
                        try:
                            feature_names = preprocessor.get_feature_names()
                            if callable(feature_names):
                                feature_names = feature_names()
                            if not isinstance(feature_names, list):
                                feature_names = list(feature_names) if feature_names else None
                        except:
                            pass
                    elif hasattr(preprocessor, 'feature_names') and preprocessor.feature_names:
                        feature_names = preprocessor.feature_names
                        if not isinstance(feature_names, list):
                            feature_names = list(feature_names) if feature_names else None
                    
                    # Ensure both DataFrames have the same feature names
                    # First, convert to DataFrame if needed
                    if not isinstance(reference_data, pd.DataFrame):
                        if feature_names and len(feature_names) == reference_data.shape[1]:
                            reference_data = pd.DataFrame(reference_data, columns=feature_names)
                        else:
                            # Fallback to generic feature names
                            reference_data = pd.DataFrame(
                                reference_data, 
                                columns=[f"feature_{i}" for i in range(reference_data.shape[1])]
                            )
                    
                    if not isinstance(current_data, pd.DataFrame):
                        if feature_names and len(feature_names) == current_data.shape[1]:
                            current_data = pd.DataFrame(current_data, columns=feature_names)
                        else:
                            # Fallback to generic feature names
                            current_data = pd.DataFrame(
                                current_data,
                                columns=[f"feature_{i}" for i in range(current_data.shape[1])]
                            )
                    
                    # Ensure both DataFrames have the same columns (use reference_data columns as source of truth)
                    # This handles cases where preprocessing might produce slightly different column names
                    if list(reference_data.columns) != list(current_data.columns):
                        logger.warning(
                            f"Column names differ after preprocessing. "
                            f"Reference: {list(reference_data.columns)[:10]}, "
                            f"Current: {list(current_data.columns)[:10]}. "
                            f"Aligning to reference columns."
                        )
                        # Use reference_data columns as source of truth
                        if len(reference_data.columns) == len(current_data.columns):
                            current_data.columns = reference_data.columns
                        else:
                            # If shapes differ, this is a more serious issue
                            raise ValueError(
                                f"Feature count mismatch after preprocessing: "
                                f"reference has {len(reference_data.columns)} features, "
                                f"current has {len(current_data.columns)} features"
                            )
                    
                    logger.info(f"Preprocessing applied. Reference shape: {reference_data.shape}, Current shape: {current_data.shape}")
                    logger.info(f"Reference columns: {list(reference_data.columns)[:10]}...")
                    logger.info(f"Current columns: {list(current_data.columns)[:10]}...")
                except Exception as e:
                    logger.error(f"Error applying preprocessing: {e}", exc_info=True)
                    raise ValueError(f"Error preprocessing data for drift detection: {str(e)}")
            
            # Get features to check
            if features is None:
                # Use model features if available, otherwise use common columns
                if model.features:
                    features = model.features
                else:
                    features = list(set(reference_data.columns) & set(current_data.columns))
            
            # Filter to common features
            common_features = list(set(features) & set(reference_data.columns) & set(current_data.columns))
            
            if not common_features:
                # Provide detailed diagnostic information
                ref_cols = list(reference_data.columns)
                curr_cols = list(current_data.columns)
                requested_features = features if features else "all"
                
                # Find which features are missing
                missing_in_ref = set(features or ref_cols) - set(ref_cols) if features else set()
                missing_in_curr = set(features or curr_cols) - set(curr_cols) if features else set()
                
                error_msg = (
                    f"No common features found between reference and current data.\n"
                    f"  - Requested features: {requested_features}\n"
                    f"  - Reference columns ({len(ref_cols)}): {ref_cols[:20]}{'...' if len(ref_cols) > 20 else ''}\n"
                    f"  - Current columns ({len(curr_cols)}): {curr_cols[:20]}{'...' if len(curr_cols) > 20 else ''}\n"
                    f"  - Model features: {model.features if model.features else 'None'}\n"
                )
                
                if missing_in_ref:
                    error_msg += f"  - Missing in reference: {list(missing_in_ref)[:10]}\n"
                if missing_in_curr:
                    error_msg += f"  - Missing in current: {list(missing_in_curr)[:10]}\n"
                
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"Checking drift on {len(common_features)} common features: {common_features[:10]}...")
            
            # Detect drift
            drift_results = self.drift_detector.detect_data_drift(
                reference_data=reference_data[common_features],
                current_data=current_data[common_features],
                features=common_features
            )
            
            # Determine severity
            severity = DriftSeverity.NONE
            if drift_results.get("drift_detected", False):
                severity_str = drift_results.get("drift_severity", "none")
                severity = DriftSeverity[severity_str.upper()] if severity_str.upper() in [s.name for s in DriftSeverity] else DriftSeverity.LOW
            
            # Create drift report
            drift_report = DriftReport(
                model_id=model_id,
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                drift_detected=drift_results.get("drift_detected", False),
                drift_results=drift_results,
                feature_results=drift_results.get("feature_results", {}),
                reference_samples=len(reference_data),
                current_samples=len(current_data),
                features_checked=len(common_features),
                detection_method="PSI_KS_CHI2",
                threshold_used={
                    "psi_threshold": self.drift_detector.psi_threshold,
                    "significance_level": self.drift_detector.significance_level
                },
                created_by=created_by
            )
            
            db.add(drift_report)
            await db.commit()
            await db.refresh(drift_report)
            
            # Send alert if drift detected
            if drift_report.drift_detected:
                await self._send_drift_alert(db, drift_report, model)
            
            return drift_report
        except Exception as e:
            logger.error(f"Error checking drift for model {model_id}: {e}", exc_info=True)
            raise
    
    async def _get_reference_data(
        self,
        db: AsyncSession,
        model: MLModel
    ) -> Optional[pd.DataFrame]:
        """Get reference data for a model."""
        try:
            # Try to get from data source used for training
            if model.data_source_id:
                data_source_result = await db.execute(
                    select(DataSource).where(DataSource.id == model.data_source_id)
                )
                data_source = data_source_result.scalar_one_or_none()
                
                if data_source:
                    # Get data points
                    data_points_result = await db.execute(
                        select(DataPoint)
                        .where(DataPoint.source_id == data_source.id)
                        .order_by(DataPoint.timestamp.desc())
                        .limit(10000)  # Limit to prevent memory issues
                    )
                    data_points = data_points_result.scalars().all()
                    
                    if data_points:
                        # Convert to DataFrame
                        data_list = []
                        for point in data_points:
                            if isinstance(point.data, dict):
                                data_list.append(point.data)
                        
                        if data_list:
                            return pd.DataFrame(data_list)
            
            return None
        except Exception as e:
            logger.error(f"Error getting reference data: {e}", exc_info=True)
            return None
    
    async def _send_drift_alert(
        self,
        db: AsyncSession,
        drift_report: DriftReport,
        model: MLModel
    ):
        """Send alert for detected drift."""
        try:
            # Determine alert severity
            alert_severity = AlertSeverity.MEDIUM
            if drift_report.severity == DriftSeverity.HIGH:
                alert_severity = AlertSeverity.HIGH
            elif drift_report.severity == DriftSeverity.LOW:
                alert_severity = AlertSeverity.LOW
            
            # Create alert
            alert = Alert(
                alert_type=AlertType.DATA_QUALITY,
                severity=alert_severity,
                status=AlertStatus.ACTIVE,
                title=f"Data Drift Detected for Model: {model.name}",
                message=f"Drift detected with severity: {drift_report.severity.value}. "
                       f"Features affected: {len([f for f, r in (drift_report.feature_results or {}).items() if r.get('drift_detected')])}",
                metadata={
                    "model_id": model.id,
                    "model_name": model.name,
                    "drift_report_id": drift_report.id,
                    "drift_type": drift_report.drift_type.value,
                    "severity": drift_report.severity.value,
                    "features_checked": drift_report.features_checked
                },
                model_id=model.id
            )
            
            db.add(alert)
            drift_report.alert_sent = True
            await db.commit()
            
            # Send notification via alert service
            await self.alert_service.send_notification(alert)
            
        except Exception as e:
            logger.error(f"Error sending drift alert: {e}", exc_info=True)
    
    async def trigger_retraining_if_needed(
        self,
        db: AsyncSession,
        drift_report: DriftReport
    ) -> bool:
        """
        Trigger retraining if drift severity is high.
        
        Args:
            db: Database session
            drift_report: Drift report
            
        Returns:
            True if retraining was triggered, False otherwise
        """
        try:
            if drift_report.severity == DriftSeverity.HIGH and not drift_report.retraining_triggered:
                # Import here to avoid circular dependency
                from app.services.training_job import TrainingJobService
                
                # Get model
                model_result = await db.execute(
                    select(MLModel).where(MLModel.id == drift_report.model_id)
                )
                model = model_result.scalar_one_or_none()
                
                if model and model.data_source_id:
                    # Create retraining job
                    training_job = await TrainingJobService.create_training_job(
                        db=db,
                        model_type=model.type.value,
                        data_source_id=model.data_source_id,
                        model_id=model.id,
                        created_by=drift_report.created_by
                    )
                    
                    drift_report.retraining_triggered = True
                    drift_report.notes = f"Retraining triggered automatically due to high drift severity. Training job ID: {training_job.id}"
                    await db.commit()
                    
                    logger.info(f"Retraining triggered for model {model.id} due to drift")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error triggering retraining: {e}", exc_info=True)
            return False
    
    async def get_drift_history(
        self,
        db: AsyncSession,
        model_id: int,
        limit: int = 50,
        offset: int = 0
    ) -> List[DriftReport]:
        """Get drift history for a model."""
        result = await db.execute(
            select(DriftReport)
            .where(DriftReport.model_id == model_id)
            .order_by(DriftReport.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())
    
    async def get_latest_drift_report(
        self,
        db: AsyncSession,
        model_id: int
    ) -> Optional[DriftReport]:
        """Get the latest drift report for a model."""
        result = await db.execute(
            select(DriftReport)
            .where(DriftReport.model_id == model_id)
            .order_by(DriftReport.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

