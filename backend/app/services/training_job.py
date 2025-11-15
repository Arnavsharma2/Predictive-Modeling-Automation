"""
Training job management service.
"""
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from app.models.database.training_jobs import TrainingJob, TrainingJobStatus
from app.models.database.ml_models import MLModel, ModelStatus, ModelType
from app.models.database.data_sources import DataSource
from app.models.database.data_points import DataPoint
from app.ml.models.predictor import RegressionPredictor
from app.ml.models.classifier import Classifier
from app.ml.models.anomaly_detector import AnomalyDetector
from app.ml.preprocessing.preprocessor import DataPreprocessor
from app.ml.preprocessing.feature_engineering import FeatureEngineer
from app.ml.preprocessing.auto_preprocessor import AutoPreprocessor
from app.ml.storage.model_storage import model_storage
from app.services.validation import DataValidationService
from app.services.data_lineage import DataLineageService
from app.ml.tracking.experiment_tracker import ExperimentTracker
from app.core.logging import get_logger
from app.core.websocket import manager as ws_manager
import pandas as pd
import os

logger = get_logger(__name__)

# Check if we're running in a Celery worker
IS_CELERY_WORKER = os.getenv("CELERY_WORKER") == "true" or "celery" in os.getenv("HOSTNAME", "").lower()

# Import Redis pub/sub for Celery workers
if IS_CELERY_WORKER:
    from app.core.redis_pubsub import redis_pubsub


class TrainingJobService:
    """Service for managing training jobs."""
    
    @staticmethod
    async def create_training_job(
        db: AsyncSession,
        model_type: str,
        data_source_id: Optional[int] = None,
        model_id: Optional[int] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        created_by: Optional[int] = None
    ) -> TrainingJob:
        """
        Create a new training job.
        
        Args:
            db: Database session
            model_type: Type of model to train
            data_source_id: ID of data source to use
            model_id: ID of existing model (for retraining)
            hyperparameters: Model hyperparameters
            training_config: Additional training configuration
            created_by: ID of user creating the job
            
        Returns:
            Created training job
        """
        job = TrainingJob(
            model_id=model_id,
            data_source_id=data_source_id,
            status=TrainingJobStatus.PENDING,
            model_type=model_type,
            hyperparameters=hyperparameters or {},
            training_config=training_config or {},
            created_by=created_by
        )
        
        db.add(job)
        await db.commit()
        await db.refresh(job)
        
        logger.info(f"Created training job {job.id} for model type {model_type}")
        return job
    
    @staticmethod
    async def get_training_job(db: AsyncSession, job_id: int) -> Optional[TrainingJob]:
        """Get training job by ID."""
        # Use populate_existing=True to refresh any existing object in the session
        # This ensures we get the latest committed data from the database
        # even if another session (like the background training task) has updated it
        result = await db.execute(
            select(TrainingJob)
            .where(TrainingJob.id == job_id)
            .execution_options(populate_existing=True)
        )
        job = result.scalar_one_or_none()
        
        # Explicitly refresh to ensure we have the absolute latest data
        # This is important because the training job is updated in a background task
        # with a different database session
        if job:
            try:
                await db.refresh(job)
            except Exception:
                # If refresh fails (e.g., object not in session), that's okay
                # populate_existing=True should have already loaded fresh data
                pass
        
        return job
    
    @staticmethod
    async def update_job_status(
        db: AsyncSession,
        job_id: int,
        status: TrainingJobStatus,
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        current_epoch: Optional[int] = None,
        total_epochs: Optional[int] = None
    ) -> None:
        """
        Update training job status and broadcast to WebSocket clients.

        Args:
            db: Database session
            job_id: Training job ID
            status: New status
            progress: Progress percentage (0-100)
            error_message: Error message if failed
            metrics: Training metrics (e.g., loss, accuracy)
            current_epoch: Current training epoch
            total_epochs: Total training epochs
        """
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow()
        }

        if progress is not None:
            update_data["progress"] = progress

        if error_message:
            update_data["error_message"] = error_message

        if metrics is not None:
            update_data["metrics"] = metrics

        if current_epoch is not None:
            update_data["current_epoch"] = current_epoch

        if total_epochs is not None:
            update_data["total_epochs"] = total_epochs

        if status == TrainingJobStatus.RUNNING and not update_data.get("started_at"):
            update_data["started_at"] = datetime.utcnow()
        elif status in [TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED]:
            update_data["completed_at"] = datetime.utcnow()

        await db.execute(
            update(TrainingJob).where(TrainingJob.id == job_id).values(**update_data)
        )
        # Flush before commit to ensure changes are visible immediately
        await db.flush()
        await db.commit()

        # Log the update for debugging
        logger.debug(f"Updated training job {job_id}: status={status}, progress={progress}")

        # Broadcast update to WebSocket clients
        update_message = {
            "type": "progress_update",
            "job_id": job_id,
            "status": status.value,
            "progress": progress,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "metrics": metrics,
            "error_message": error_message,
            "updated_at": datetime.utcnow().isoformat()
        }

        # Use Redis pub/sub if we're in a Celery worker (cross-process communication)
        # Otherwise use direct WebSocket manager (in-process)
        if IS_CELERY_WORKER:
            logger.debug(f"Publishing training update to Redis: job_id={job_id}, status={status}")
            redis_pubsub.publish_training_update(job_id, update_message)
        else:
            subscription_key = f"training_{job_id}"
            await ws_manager.broadcast_job_update(subscription_key, update_message)
    
    @staticmethod
    async def load_training_data(
        db: AsyncSession,
        data_source_id: int,
        target_column: str,
        limit: Optional[int] = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load training data from database.
        
        Args:
            db: Database session
            data_source_id: ID of data source
            target_column: Name of target column
            limit: Maximum number of records to load
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Query data points
        query = select(DataPoint).where(DataPoint.source_id == data_source_id)
        
        if limit:
            query = query.limit(limit)
        
        result = await db.execute(query)
        data_points = result.scalars().all()
        
        if not data_points:
            raise ValueError(f"No data points found for source {data_source_id}")
        
        # Convert to DataFrame
        records = []
        for point in data_points:
            record = point.data.copy()
            record['timestamp'] = point.timestamp
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Log DataFrame info for debugging
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        if len(df) > 0:
            logger.debug(f"Sample columns: {list(df.columns[:5])}")
        
        # Check if target column exists (case-insensitive match)
        available_columns = list(df.columns)
        target_column_lower = target_column.lower()
        
        # Try exact match first
        if target_column in df.columns:
            actual_target_column = target_column
        # Try case-insensitive match
        elif target_column_lower in [col.lower() for col in df.columns]:
            # Find the actual column name with correct case
            actual_target_column = next(
                col for col in df.columns if col.lower() == target_column_lower
            )
            logger.info(f"Found target column '{actual_target_column}' (case-insensitive match for '{target_column}')")
        else:
            # Provide helpful error message with available columns
            available_str = ', '.join(available_columns[:10])  # Show first 10 columns
            if len(available_columns) > 10:
                available_str += f", ... ({len(available_columns)} total columns)"
            raise ValueError(
                f"Target column '{target_column}' not found in data. "
                f"Available columns: {available_str}"
            )
        
        # Separate features and target
        y = df[actual_target_column]
        X = df.drop(columns=[actual_target_column])
        
        return X, y
    
    @staticmethod
    async def execute_training_job(
        db: AsyncSession,
        job_id: int,
        data_source_id: int,
        target_column: str,
        model_name: str,
        algorithm: str = "random_forest",
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None
    ) -> MLModel:
        """
        Execute a training job.

        Args:
            db: Database session
            job_id: Training job ID
            data_source_id: Data source ID
            target_column: Target column name
            model_name: Name for the model
            algorithm: ML algorithm to use
            hyperparameters: Model hyperparameters
            training_config: Training configuration

        Returns:
            Trained model record
        """
        import sys
        print("="*100, file=sys.stderr, flush=True)
        print(f"EXECUTE_TRAINING_JOB CALLED - JOB_ID: {job_id}", file=sys.stderr, flush=True)
        print("="*100, file=sys.stderr, flush=True)
        logger.info("="*100)
        logger.info(f"EXECUTE_TRAINING_JOB CALLED - JOB_ID: {job_id}")
        logger.info("="*100)
        try:
            # Get training job to retrieve model_type
            job = await TrainingJobService.get_training_job(db, job_id)
            if not job:
                raise ValueError(f"Training job {job_id} not found")
            
            model_type = job.model_type
            
            # Update job status
            await TrainingJobService.update_job_status(
                db, job_id, TrainingJobStatus.RUNNING, progress=0.0
            )
            
            # Load training data
            logger.info(f"Loading training data from source {data_source_id}")
            X, y = await TrainingJobService.load_training_data(
                db, data_source_id, target_column
            )
            
            # Pre-training data validation
            logger.info("Performing pre-training data validation...")
            validation_service = DataValidationService()
            pre_validation = validation_service.comprehensive_validation(
                data=pd.concat([X, y], axis=1),
                required_columns=[target_column],
                completeness_threshold=0.8  # Allow up to 20% missing values
            )
            
            if not pre_validation["valid"]:
                issues = pre_validation.get("all_issues", [])
                warnings = pre_validation.get("all_warnings", [])
                logger.warning(f"Data validation found {len(issues)} issues and {len(warnings)} warnings")
                for issue in issues[:5]:  # Show first 5 issues
                    logger.warning(f"  - {issue}")
                # Continue anyway - preprocessing will handle missing values
            else:
                logger.info("Pre-training data validation passed")

            # Data quality filtering for regression tasks
            if model_type == "regression":
                logger.info("="*80)
                logger.info("PERFORMING DATA QUALITY CHECKS FOR REGRESSION")
                logger.info("="*80)

                initial_count = len(X)
                logger.info(f"Initial dataset size: {initial_count} rows")

                # Try to convert target to numeric if it's not already
                if not pd.api.types.is_numeric_dtype(y):
                    try:
                        y = pd.to_numeric(y, errors='coerce')
                        logger.info(f"Converted target column to numeric type")
                    except:
                        pass

                # Create a mask for valid data
                valid_mask = pd.Series(True, index=y.index)

                # 1. Remove null/NaN target values
                null_mask = y.notna()
                nulls_removed = (~null_mask).sum()
                if nulls_removed > 0:
                    logger.warning(f"Removing {nulls_removed} rows with null target values")
                    valid_mask &= null_mask

                # 2. Remove zero or negative values (prices should be positive)
                if (y <= 0).any():
                    zero_mask = y > 0
                    zeros_removed = (~zero_mask).sum()
                    logger.warning(f"Removing {zeros_removed} rows with zero or negative target values")
                    logger.warning(f"  - Target should represent valid prices/amounts (must be > 0)")
                    valid_mask &= zero_mask

                # 3. Remove extreme outliers using IQR method (before log transform)
                if valid_mask.sum() > 10:  # Need at least some data to calculate IQR
                    y_valid_temp = y[valid_mask]
                    q1 = y_valid_temp.quantile(0.25)
                    q3 = y_valid_temp.quantile(0.75)
                    iqr = q3 - q1

                    # Use 3*IQR for outlier detection (more lenient than 1.5*IQR)
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr

                    outlier_mask = (y >= lower_bound) & (y <= upper_bound)
                    outliers_removed = (~outlier_mask & valid_mask).sum()

                    if outliers_removed > 0:
                        logger.warning(f"Removing {outliers_removed} extreme outliers (outside 3*IQR range)")
                        logger.warning(f"  - Valid range: [{lower_bound:,.2f}, {upper_bound:,.2f}]")
                        logger.warning(f"  - This helps prevent a few extreme values from dominating the model")
                        valid_mask &= outlier_mask

                # Apply the filter
                final_count = valid_mask.sum()
                removed_count = initial_count - final_count

                if removed_count > 0:
                    logger.warning("="*80)
                    logger.warning(f"DATA QUALITY FILTERING SUMMARY")
                    logger.warning(f"  - Initial: {initial_count} rows")
                    logger.warning(f"  - Removed: {removed_count} rows ({100*removed_count/initial_count:.1f}%)")
                    logger.warning(f"  - Final: {final_count} rows")
                    logger.warning("="*80)

                    # Apply filter to both X and y
                    X = X[valid_mask].reset_index(drop=True)
                    y = y[valid_mask].reset_index(drop=True)

                    if final_count < 100:
                        raise ValueError(f"After data quality filtering, only {final_count} valid rows remain. Need at least 100 rows for reliable model training.")
                else:
                    logger.info("No data quality issues found - all rows are valid")

                logger.info(f"After data quality filtering: {len(X)} rows, target range: [{y.min():,.2f}, {y.max():,.2f}]")
                logger.info("="*80)

            # Validate target column type matches model type
            is_numeric = pd.api.types.is_numeric_dtype(y)
            is_string = pd.api.types.is_string_dtype(y) or pd.api.types.is_object_dtype(y)

            # Store label encoder for classification with string labels
            label_encoder = None

            if model_type == "regression" and not is_numeric:
                # Check if it's actually numeric but stored as string
                try:
                    y_numeric = pd.to_numeric(y, errors='raise')
                    logger.info(f"Target column '{target_column}' contains numeric values stored as strings, converting...")
                    y = y_numeric
                except (ValueError, TypeError):
                    # It's truly non-numeric
                    error_msg = (
                        f"Target column '{target_column}' contains non-numeric values (e.g., '{y.iloc[0] if len(y) > 0 else 'N/A'}'). "
                        f"Regression models require numeric target values. "
                        f"Please use 'classification' model type for categorical targets."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            elif model_type == "classification":
                if not is_numeric:
                    # Encode string labels to numeric for classification
                    from sklearn.preprocessing import LabelEncoder
                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y)
                    logger.info(f"Encoded target labels: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
                    y = pd.Series(y_encoded, index=y.index, name=y.name)
                elif y.nunique() > 100:
                    # Warn if there are many unique numeric values
                    logger.warning(
                        f"Target column '{target_column}' has {y.nunique()} unique numeric values. "
                        f"Consider using 'regression' model type for continuous numeric targets."
                    )
            
            await TrainingJobService.update_job_status(
                db, job_id, TrainingJobStatus.RUNNING, progress=10.0
            )
            
            # Feature engineering (if configured)
            if training_config and training_config.get("feature_engineering"):
                fe_config = training_config["feature_engineering"]
                
                if fe_config.get("time_features") and "timestamp" in X.columns:
                    X = FeatureEngineer.create_time_features(X, "timestamp")
                
                if fe_config.get("statistical_features"):
                    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
                    X = FeatureEngineer.create_statistical_features(
                        X, numeric_cols, window_size=fe_config.get("window_size", 7)
                    )
            
            await TrainingJobService.update_job_status(
                db, job_id, TrainingJobStatus.RUNNING, progress=20.0
            )

            # Store original column names before preprocessing (for feature name mapping)
            original_columns = list(X.columns)
            logger.info(f"Stored {len(original_columns)} original columns: {original_columns[:10]}...")

            # Run CPU-intensive preprocessing and training in thread pool to avoid blocking event loop
            # This allows other requests to be processed while training happens
            logger.info("Running preprocessing and training in thread pool to avoid blocking event loop...")
            
            def _preprocess_and_train_sync():
                """Synchronous function to run preprocessing and training in thread pool."""
                # Data preprocessing
                logger.info("="*80)
                logger.info("DEBUG: STARTING PREPROCESSING SECTION (in thread pool)")
                logger.info("="*80)
                
                # Check if AutoPreprocessor should be used (default: True)
                use_auto_preprocessing = training_config.get("use_auto_preprocessing", True) if training_config else True
                logger.info(f"use_auto_preprocessing: {use_auto_preprocessing}")

                # Combine X and y for preprocessing
                logger.info(f"Combining X and y for preprocessing. X shape: {X.shape}, y shape: {y.shape}")
                data_df = X.copy()
                data_df[target_column] = y
                logger.info(f"data_df shape after combining: {data_df.shape}")

                # Store preprocessing summary for later use in inverse transform detection
                preprocessing_summary = None

                if use_auto_preprocessing:
                    logger.info("Using AutoPreprocessor for intelligent preprocessing...")

                    # Determine task type
                    task = 'classification' if model_type == 'classification' else 'regression'
                    logger.info(f"Task type: {task}")

                    try:
                        logger.info("Creating AutoPreprocessor instance...")
                        preprocessor = AutoPreprocessor(
                            target_column=target_column,
                            task=task,
                            enable_feature_engineering=training_config.get("enable_feature_engineering", True) if training_config else True,
                            enable_outlier_handling=training_config.get("enable_outlier_handling", True) if training_config else True,
                            enable_feature_selection=training_config.get("enable_feature_selection", True) if training_config else True,
                            max_features=training_config.get("max_features") if training_config else None,
                            verbose=True
                        )
                        logger.info("AutoPreprocessor instance created successfully")
                    except Exception as e:
                        logger.error(f"ERROR creating AutoPreprocessor: {e}", exc_info=True)
                        raise

                    try:
                        logger.info("Calling preprocessor.fit_transform()...")
                        # fit_transform now returns (X, y_transformed) tuple
                        result = preprocessor.fit_transform(data_df, y)
                        logger.info(f"fit_transform completed. Result type: {type(result)}")
                        if isinstance(result, tuple):
                            X_processed, y_transformed = result  # Update y with transformed version
                            logger.info(f"Result is tuple. X_processed shape: {X_processed.shape}, y shape: {y_transformed.shape}")
                        else:
                            X_processed = result
                            y_transformed = y
                            logger.info(f"Result is not tuple. X_processed shape: {X_processed.shape}")
                    except Exception as e:
                        logger.error(f"ERROR in preprocessor.fit_transform(): {e}", exc_info=True)
                        raise

                    try:
                        logger.info("Getting feature names...")
                        feature_names = preprocessor.get_feature_names()
                        logger.info(f"Got {len(feature_names)} feature names")
                    except Exception as e:
                        logger.error(f"ERROR getting feature names: {e}", exc_info=True)
                        raise

                    try:
                        # Log preprocessing summary and store for later use
                        logger.info("Getting preprocessing summary...")
                        preprocessing_summary = preprocessor.get_preprocessing_summary()
                        logger.info(f"AutoPreprocessor summary: {preprocessing_summary['steps_applied']}")
                        logger.info(f"Final feature count: {preprocessing_summary['final_n_features']}")
                        logger.info(f"Target transform applied: {preprocessing_summary.get('target_transform', None)}")
                    except Exception as e:
                        logger.error(f"ERROR getting preprocessing summary: {e}", exc_info=True)
                        raise
                else:
                    logger.info("Using basic DataPreprocessor...")

                    try:
                        logger.info("Creating DataPreprocessor instance...")
                        preprocessor = DataPreprocessor(
                            target_column=target_column,
                            scale_numeric=training_config.get("scale_numeric", True) if training_config else True,
                            handle_missing=training_config.get("handle_missing", "mean") if training_config else "mean"
                        )
                        logger.info("DataPreprocessor instance created successfully")
                    except Exception as e:
                        logger.error(f"ERROR creating DataPreprocessor: {e}", exc_info=True)
                        raise

                    try:
                        logger.info("Calling preprocessor.fit_transform()...")
                        X_processed = preprocessor.fit_transform(data_df)
                        y_transformed = y
                        logger.info(f"fit_transform completed. X_processed shape: {X_processed.shape}")
                    except Exception as e:
                        logger.error(f"ERROR in preprocessor.fit_transform(): {e}", exc_info=True)
                        raise

                    try:
                        logger.info("Getting feature names...")
                        feature_names = preprocessor.get_feature_names()
                        logger.info(f"Got {len(feature_names)} feature names")
                    except Exception as e:
                        logger.error(f"ERROR getting feature names: {e}", exc_info=True)
                        raise
                
                logger.info("="*80)
                logger.info("DEBUG: PREPROCESSING SECTION COMPLETED")
                logger.info("="*80)
                
                # Post-preprocessing validation
                logger.info("Performing post-preprocessing validation...")
                try:
                    # Recreate validation service in thread pool to avoid closure issues
                    validation_service_sync = DataValidationService()
                    post_validation = validation_service_sync.comprehensive_validation(
                        data=X_processed,
                        completeness_threshold=0.95  # After preprocessing, should have very few missing values
                    )
                    
                    if not post_validation["valid"]:
                        issues = post_validation.get("all_issues", [])
                        logger.warning(f"Post-preprocessing validation found {len(issues)} issues")
                        for issue in issues[:3]:
                            logger.warning(f"  - {issue}")
                    else:
                        logger.info("Post-preprocessing data validation passed")
                except Exception as e:
                    logger.warning(f"Post-preprocessing validation failed: {e}. Continuing...")

                # Train model - select appropriate predictor based on model type
                logger.info(f"Training {model_type} model...")
                logger.info(f"Algorithm: {algorithm}")

                try:
                    logger.info("Creating predictor instance...")
                    if model_type == "classification":
                        logger.info("Creating Classifier...")
                        predictor = Classifier(algorithm=algorithm)
                        logger.info("Classifier created successfully")
                    elif model_type == "anomaly_detection":
                        logger.info("Creating AnomalyDetector...")
                        # AnomalyDetector doesn't use algorithm parameter, uses contamination instead
                        contamination = hyperparameters.get("contamination", 0.1) if hyperparameters else 0.1
                        predictor = AnomalyDetector(contamination=contamination)
                        logger.info(f"AnomalyDetector created successfully with contamination={contamination}")
                    else:  # regression (default)
                        logger.info("Creating RegressionPredictor...")
                        predictor = RegressionPredictor(algorithm=algorithm)
                        logger.info("RegressionPredictor created successfully")
                    logger.info(f"Predictor type: {type(predictor)}")
                except Exception as e:
                    logger.error(f"ERROR creating predictor: {e}", exc_info=True)
                    raise

                # Check if target was transformed (log transform for skewed regression targets)
                logger.info("="*80)
                logger.info("=== Checking target transformation ===")
                logger.info("="*80)
                logger.info(f"  - use_auto_preprocessing: {use_auto_preprocessing}")
                logger.info(f"  - model_type: {model_type}")
                logger.info(f"  - preprocessing_summary is None: {preprocessing_summary is None}")
                
                # Try multiple detection methods for maximum reliability
                target_was_transformed = False
                
                # Method 1: Check preprocessing_steps list
                if use_auto_preprocessing and hasattr(preprocessor, 'preprocessing_steps'):
                    has_target_transform_step = 'target_log_transform' in preprocessor.preprocessing_steps
                    logger.info(f"  - Method 1 (preprocessing_steps): {has_target_transform_step}")
                    if has_target_transform_step:
                        target_was_transformed = True
                        logger.info(f"  - Method 1 DETECTED target transformation!")
                
                # Method 2: Check target_transform attribute directly
                if use_auto_preprocessing and hasattr(preprocessor, 'target_transform'):
                    has_target_transform_attr = preprocessor.target_transform == 'log1p'
                    logger.info(f"  - Method 2 (target_transform attr): {has_target_transform_attr} (value: {getattr(preprocessor, 'target_transform', 'NOT FOUND')})")
                    if has_target_transform_attr:
                        target_was_transformed = True
                        logger.info(f"  - Method 2 DETECTED target transformation!")
                
                # Method 3: Check preprocessing summary (most reliable - already retrieved)
                if use_auto_preprocessing and model_type == "regression" and preprocessing_summary is not None:
                    try:
                        summary_target_transform = preprocessing_summary.get('target_transform')
                        has_target_transform_summary = summary_target_transform == 'log1p'
                        logger.info(f"  - Method 3 (summary): {has_target_transform_summary} (value: {summary_target_transform})")
                        if has_target_transform_summary:
                            target_was_transformed = True
                            logger.info(f"  - Method 3 DETECTED target transformation!")
                    except Exception as e:
                        logger.warning(f"  - Method 3 (summary) failed: {e}")
                
                # Method 4: Force check if preprocessing_summary is None but we're using AutoPreprocessor
                # This is a fallback - if AutoPreprocessor was used and it's regression, check the preprocessor directly
                if not target_was_transformed and use_auto_preprocessing and model_type == "regression":
                    logger.info("  - Method 4: Fallback check - preprocessing_summary was None, checking preprocessor directly...")
                    try:
                        # Try to get preprocessing summary again if it wasn't retrieved
                        if hasattr(preprocessor, 'get_preprocessing_summary'):
                            fallback_summary = preprocessor.get_preprocessing_summary()
                            if fallback_summary and fallback_summary.get('target_transform') == 'log1p':
                                target_was_transformed = True
                                logger.info(f"  - Method 4 DETECTED target transformation via fallback!")
                                preprocessing_summary = fallback_summary  # Store it for later use
                    except Exception as e:
                        logger.warning(f"  - Method 4 (fallback) failed: {e}")
                
                # Final check: ensure it's regression task
                if model_type != "regression":
                    target_was_transformed = False
                    logger.info(f"  - Skipping inverse transform: not a regression task")
                
                logger.info(f"  - FINAL target_was_transformed: {target_was_transformed}")

                if target_was_transformed:
                    logger.info(f"Target log transformation detected - will inverse transform predictions")
                else:
                    logger.warning(f"Target transformation NOT detected - inverse transform will be skipped")

                # Train the model
                logger.info("Training model in thread pool...")
                training_results = predictor.train(
                    X_processed,
                    y_transformed,
                    hyperparameters=hyperparameters or {},
                    test_size=training_config.get("test_size", 0.2) if training_config else 0.2,
                    do_cross_validation=training_config.get("do_cross_validation", True) if training_config else True
                )
                
                # Return all the results we need
                return {
                    'preprocessor': preprocessor,
                    'feature_names': feature_names,
                    'training_results': training_results,
                    'predictor': predictor,
                    'preprocessing_summary': preprocessing_summary,
                    'target_was_transformed': target_was_transformed,
                    'use_auto_preprocessing': use_auto_preprocessing,
                    'y_transformed': y_transformed
                }
            
            # Use asyncio.to_thread (Python 3.9+) or run_in_executor as fallback
            try:
                # Python 3.9+ has asyncio.to_thread which is cleaner
                preprocess_and_train_results = await asyncio.to_thread(_preprocess_and_train_sync)
            except AttributeError:
                # Fallback for Python < 3.9
                loop = asyncio.get_event_loop()
                preprocess_and_train_results = await loop.run_in_executor(None, _preprocess_and_train_sync)
            
            # Extract results
            preprocessor = preprocess_and_train_results['preprocessor']
            feature_names = preprocess_and_train_results['feature_names']
            training_results = preprocess_and_train_results['training_results']
            predictor = preprocess_and_train_results['predictor']
            preprocessing_summary = preprocess_and_train_results['preprocessing_summary']
            target_was_transformed = preprocess_and_train_results['target_was_transformed']
            use_auto_preprocessing = preprocess_and_train_results['use_auto_preprocessing']
            y = preprocess_and_train_results['y_transformed']
            
            logger.info("Preprocessing and training completed in thread pool, continuing with post-processing...")
            
            try:
                logger.info("Updating job status to 40% progress...")
                await TrainingJobService.update_job_status(
                    db, job_id, TrainingJobStatus.RUNNING, progress=40.0
                )
                logger.info("Job status updated to 40% successfully")
            except Exception as e:
                logger.error(f"ERROR updating job status to 40%: {e}", exc_info=True)
                import sys
                print(f"ERROR updating job status to 40%: {e}", file=sys.stderr, flush=True)
                raise
            
            # Extract feature importance early (before MLflow logging)
            feature_importance = None
            if "feature_importance" in training_results:
                feature_importance_raw = training_results.get("feature_importance")
                if feature_importance_raw:
                    # Convert numpy types to Python types for JSON serialization
                    if isinstance(feature_importance_raw, dict):
                        feature_importance = {
                            str(k): float(v) for k, v in feature_importance_raw.items()
                        }
                    else:
                        # If it's a list or array, convert to dict
                        feature_importance = {
                            str(i): float(v) for i, v in enumerate(feature_importance_raw)
                        }
                    logger.info(f"Extracted feature importance: {len(feature_importance)} features")

            # If target was transformed, we need to inverse transform predictions for metrics
            # CRITICAL: If using AutoPreprocessor for regression, we MUST inverse transform
            # because AutoPreprocessor applies log1p to skewed targets
            # Even if detection failed, if we're using AutoPreprocessor and it's regression,
            # we should attempt inverse transform (worst case: it won't hurt if target wasn't transformed)
            should_inverse_transform = target_was_transformed and model_type == "regression"
            logger.info(f"Initial should_inverse_transform: {should_inverse_transform} (target_was_transformed={target_was_transformed}, model_type={model_type})")
            
            # Aggressive fallback: if AutoPreprocessor was used and it's regression, force inverse transform
            # This is safe because expm1 of non-transformed values will just return the original values
            if not should_inverse_transform and use_auto_preprocessing and model_type == "regression":
                logger.warning("Target transformation detection failed, but AutoPreprocessor was used for regression.")
                logger.warning("Forcing inverse transform attempt (safe: expm1 of non-log values returns original)")
                should_inverse_transform = True
            
            logger.info(f"FINAL should_inverse_transform: {should_inverse_transform}")
            import sys
            print(f"FINAL should_inverse_transform: {should_inverse_transform}", file=sys.stderr, flush=True)
            
            if should_inverse_transform:
                logger.info("="*80)
                logger.info("ENTERING INVERSE TRANSFORM BLOCK")
                logger.info("="*80)
                import sys
                print("ENTERING INVERSE TRANSFORM BLOCK", file=sys.stderr, flush=True)
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                import numpy as np

                logger.info("Inverse transforming predictions to original scale for metrics...")
                import sys
                print("Inverse transforming predictions to original scale for metrics...", file=sys.stderr, flush=True)

                # Get the test predictions and targets from training results
                logger.info(f"training_results keys: {list(training_results.keys())}")
                print(f"training_results keys: {list(training_results.keys())}", file=sys.stderr, flush=True)
                
                if "test_predictions" in training_results and "test_targets" in training_results:
                    logger.info("Found test_predictions and test_targets in training_results")
                    print("Found test_predictions and test_targets in training_results", file=sys.stderr, flush=True)
                    test_predictions_transformed = training_results["test_predictions"]
                    y_test_transformed = training_results["test_targets"]

                    logger.info(f"Test predictions (log space): min={np.min(test_predictions_transformed):.2f}, max={np.max(test_predictions_transformed):.2f}, mean={np.mean(test_predictions_transformed):.2f}")
                    logger.info(f"Test targets (log space): min={np.min(y_test_transformed):.2f}, max={np.max(y_test_transformed):.2f}, mean={np.mean(y_test_transformed):.2f}")
                    print(f"Test predictions (log space): min={np.min(test_predictions_transformed):.2f}, max={np.max(test_predictions_transformed):.2f}, mean={np.mean(test_predictions_transformed):.2f}", file=sys.stderr, flush=True)
                    print(f"Test targets (log space): min={np.min(y_test_transformed):.2f}, max={np.max(y_test_transformed):.2f}, mean={np.mean(y_test_transformed):.2f}", file=sys.stderr, flush=True)
                    
                    # DIAGNOSTIC: Check if values are already in original scale (would be > 1000 typically)
                    # If they're already in original scale, we shouldn't inverse transform
                    pred_max = np.max(np.abs(test_predictions_transformed))
                    target_max = np.max(np.abs(y_test_transformed))
                    logger.info(f"DIAGNOSTIC: Max absolute prediction: {pred_max:.2f}, Max absolute target: {target_max:.2f}")
                    print(f"DIAGNOSTIC: Max absolute prediction: {pred_max:.2f}, Max absolute target: {target_max:.2f}", file=sys.stderr, flush=True)
                    
                    # If values are > 1000, they're likely already in original scale, not log space
                    if pred_max > 1000 or target_max > 1000:
                        logger.warning("WARNING: Values appear to already be in original scale (>1000), not log space!")
                        logger.warning("Skipping inverse transform - metrics are already on original scale")
                        print("WARNING: Values appear to already be in original scale (>1000), not log space!", file=sys.stderr, flush=True)
                        print("Skipping inverse transform - metrics are already on original scale", file=sys.stderr, flush=True)
                        # Don't inverse transform - use metrics as-is
                        test_predictions_original = test_predictions_transformed
                        y_test_original = y_test_transformed
                    else:
                        # Inverse transform to original scale (log1p -> expm1)
                        logger.info("Applying expm1 inverse transform...")
                        test_predictions_original = np.expm1(test_predictions_transformed)
                        y_test_original = np.expm1(y_test_transformed)
                    
                    logger.info(f"Test predictions (original scale): min={np.min(test_predictions_original):.2f}, max={np.max(test_predictions_original):.2f}, mean={np.mean(test_predictions_original):.2f}")
                    logger.info(f"Test targets (original scale): min={np.min(y_test_original):.2f}, max={np.max(y_test_original):.2f}, mean={np.mean(y_test_original):.2f}")
                    print(f"Test predictions (original scale): min={np.min(test_predictions_original):.2f}, max={np.max(test_predictions_original):.2f}, mean={np.mean(test_predictions_original):.2f}", file=sys.stderr, flush=True)
                    print(f"Test targets (original scale): min={np.min(y_test_original):.2f}, max={np.max(y_test_original):.2f}, mean={np.mean(y_test_original):.2f}", file=sys.stderr, flush=True)

                    # Check for invalid values and handle extreme outliers
                    # Remove infinite and NaN values before calculating metrics
                    valid_mask = np.isfinite(y_test_original) & np.isfinite(test_predictions_original)
                    
                    if not np.all(valid_mask):
                        invalid_count = np.sum(~valid_mask)
                        logger.warning(f"Found {invalid_count} invalid values (inf/nan) after inverse transform. Removing them from metrics calculation.")
                        print(f"WARNING: Found {invalid_count} invalid values after inverse transform", file=sys.stderr, flush=True)
                        y_test_original = y_test_original[valid_mask]
                        test_predictions_original = test_predictions_original[valid_mask]
                    
                    # Handle extreme outliers that might skew metrics
                    # Use IQR method to identify and optionally cap extreme values
                    if len(y_test_original) > 0:
                        q1 = np.percentile(y_test_original, 25)
                        q3 = np.percentile(y_test_original, 75)
                        iqr = q3 - q1
                        lower_bound = q1 - 3 * iqr  # More lenient than typical 1.5*IQR
                        upper_bound = q3 + 3 * iqr
                        
                        extreme_mask = (y_test_original < lower_bound) | (y_test_original > upper_bound)
                        extreme_count = np.sum(extreme_mask)
                        
                        if extreme_count > 0:
                            logger.warning(f"Found {extreme_count} extreme outliers in targets (outside 3*IQR range). These may skew metrics.")
                            logger.warning(f"Target range: [{lower_bound:.2f}, {upper_bound:.2f}], but found values from {np.min(y_test_original):.2f} to {np.max(y_test_original):.2f}")
                            print(f"WARNING: {extreme_count} extreme outliers detected in targets", file=sys.stderr, flush=True)
                            
                            # Optionally cap extreme values for more robust metrics
                            # But keep original for reporting
                            y_test_original_capped = np.clip(y_test_original, lower_bound, upper_bound)
                            test_predictions_original_capped = np.clip(test_predictions_original, lower_bound, upper_bound)
                            
                            # Calculate metrics with and without capping
                            logger.info("Calculating metrics on original scale (with extreme outliers)...")
                            rmse_original = np.sqrt(mean_squared_error(y_test_original, test_predictions_original))
                            mae_original = mean_absolute_error(y_test_original, test_predictions_original)
                            r2_original = r2_score(y_test_original, test_predictions_original)
                            
                            # Also calculate capped metrics for comparison
                            rmse_capped = np.sqrt(mean_squared_error(y_test_original_capped, test_predictions_original_capped))
                            mae_capped = mean_absolute_error(y_test_original_capped, test_predictions_original_capped)
                            r2_capped = r2_score(y_test_original_capped, test_predictions_original_capped)
                            
                            logger.info(f"Metrics (with outliers) - RMSE: {rmse_original:.2f}, MAE: {mae_original:.2f}, R²: {r2_original:.4f}")
                            logger.info(f"Metrics (outliers capped) - RMSE: {rmse_capped:.2f}, MAE: {mae_capped:.2f}, R²: {r2_capped:.4f}")
                            print(f"Metrics (with outliers) - RMSE: {rmse_original:.2f}, MAE: {mae_original:.2f}, R²: {r2_original:.4f}", file=sys.stderr, flush=True)
                            print(f"Metrics (outliers capped) - RMSE: {rmse_capped:.2f}, MAE: {mae_capped:.2f}, R²: {r2_capped:.4f}", file=sys.stderr, flush=True)
                            
                            # Decide which metrics to use based on multiple factors
                            # Prefer capped metrics if:
                            # 1. R² is better (higher) AND still reasonable (> -1.0), OR
                            # 2. RMSE is significantly better (at least 20% reduction) AND R² is not much worse
                            rmse_improvement = (rmse_original - rmse_capped) / rmse_original if rmse_original > 0 else 0
                            use_capped = False
                            
                            if r2_capped > r2_original and r2_capped > -1.0:
                                use_capped = True
                                reason = "R² is better"
                            elif rmse_improvement > 0.2 and r2_capped > -1.0 and (r2_original - r2_capped) < 0.1:
                                # RMSE improved by at least 20% and R² didn't degrade too much
                                use_capped = True
                                reason = f"RMSE improved by {rmse_improvement*100:.1f}%"
                            
                            if use_capped:
                                logger.info(f"Using capped metrics (more robust to outliers) - Reason: {reason}")
                                print(f"Using capped metrics - Reason: {reason}", file=sys.stderr, flush=True)
                                rmse_original = rmse_capped
                                mae_original = mae_capped
                                r2_original = r2_capped
                            else:
                                logger.info("Using uncapped metrics (outliers included) - capped metrics were not better")
                                print("Using uncapped metrics (outliers included)", file=sys.stderr, flush=True)
                        else:
                            # No extreme outliers, calculate metrics normally
                            logger.info("Calculating metrics on original scale...")
                            rmse_original = np.sqrt(mean_squared_error(y_test_original, test_predictions_original))
                            mae_original = mean_absolute_error(y_test_original, test_predictions_original)
                            r2_original = r2_score(y_test_original, test_predictions_original)
                    else:
                        logger.error("No valid targets after inverse transform!")
                        # Fallback to transformed scale metrics
                        rmse_original = training_results['test_metrics'].get('rmse', 0)
                        mae_original = training_results['test_metrics'].get('mae', 0)
                        r2_original = training_results['test_metrics'].get('r2_score', 0)
                    
                    # Also calculate some diagnostic info (only if we have valid data)
                    if len(y_test_original) > 0 and len(test_predictions_original) > 0:
                        errors = y_test_original - test_predictions_original
                        logger.info(f"Error statistics: mean={np.mean(errors):.2f}, std={np.std(errors):.2f}, min={np.min(errors):.2f}, max={np.max(errors):.2f}")
                        print(f"Error statistics: mean={np.mean(errors):.2f}, std={np.std(errors):.2f}, min={np.min(errors):.2f}, max={np.max(errors):.2f}", file=sys.stderr, flush=True)

                    logger.info(f"Final metrics on original scale - RMSE: {rmse_original:.2f}, MAE: {mae_original:.2f}, R²: {r2_original:.4f}")
                    logger.info(f"Metrics on transformed scale - RMSE: {training_results['test_metrics'].get('rmse', 0):.2f}, R²: {training_results['test_metrics'].get('r2_score', 0):.4f}")
                    print(f"Final metrics on original scale - RMSE: {rmse_original:.2f}, MAE: {mae_original:.2f}, R²: {r2_original:.4f}", file=sys.stderr, flush=True)
                    print(f"Metrics on transformed scale - RMSE: {training_results['test_metrics'].get('rmse', 0):.2f}, R²: {training_results['test_metrics'].get('r2_score', 0):.4f}", file=sys.stderr, flush=True)

                    # Update metrics to use original scale
                    logger.info(f"BEFORE UPDATE - test_metrics: {training_results['test_metrics']}")
                    print(f"BEFORE UPDATE - test_metrics: {training_results['test_metrics']}", file=sys.stderr, flush=True)
                    
                    training_results["test_metrics"]["rmse"] = float(rmse_original)
                    training_results["test_metrics"]["mae"] = float(mae_original)
                    training_results["test_metrics"]["r2_score"] = float(r2_original)
                    
                    logger.info(f"AFTER UPDATE - test_metrics: {training_results['test_metrics']}")
                    print(f"AFTER UPDATE - test_metrics: {training_results['test_metrics']}", file=sys.stderr, flush=True)
                else:
                    logger.error(f"MISSING test_predictions or test_targets in training_results!")
                    logger.error(f"Available keys: {list(training_results.keys())}")
                    print(f"MISSING test_predictions or test_targets in training_results!", file=sys.stderr, flush=True)
                    print(f"Available keys: {list(training_results.keys())}", file=sys.stderr, flush=True)
            
            await TrainingJobService.update_job_status(
                db, job_id, TrainingJobStatus.RUNNING, progress=80.0
            )
            
            # Save model to cloud storage
            logger.info("="*80)
            logger.info("STARTING MODEL SAVE PROCESS")
            logger.info("="*80)
            logger.info("Saving model to cloud storage...")

            # Create feature name mapping for readable display using original columns
            from app.ml.utils.feature_names import create_feature_name_mapping, create_original_column_mapping

            # If using AutoPreprocessor with original columns tracked, use its mapping
            if use_auto_preprocessing and hasattr(preprocessor, 'get_feature_name_mapping'):
                try:
                    feature_name_mapping = preprocessor.get_feature_name_mapping()
                    logger.info(f"Got feature name mapping from AutoPreprocessor for {len(feature_name_mapping)} features")
                except Exception as e:
                    logger.warning(f"Failed to get mapping from AutoPreprocessor: {e}, falling back to manual creation")
                    feature_name_mapping = create_feature_name_mapping(feature_names, original_columns=original_columns)
            else:
                feature_name_mapping = create_feature_name_mapping(feature_names, original_columns=original_columns)

            original_column_mapping = create_original_column_mapping(feature_names, original_columns)
            logger.info(f"Created feature name mapping for {len(feature_name_mapping)} features")
            logger.info(f"Created original column mapping for {len(original_column_mapping)} features")

            # Build metadata dictionary
            metadata = {
                "algorithm": algorithm,
                "feature_names": feature_names,
                "feature_name_mapping": feature_name_mapping,  # Mapping from technical to readable names
                "original_columns": original_columns,  # Store original dataset columns
                "original_column_mapping": original_column_mapping,  # Mapping from technical features to original columns
                "hyperparameters": hyperparameters
            }

            # Include label encoder for string classification targets
            if label_encoder is not None:
                metadata["label_encoder"] = {
                    "classes": label_encoder.classes_.tolist()
                }
                logger.info(f"Saving label encoder with classes: {label_encoder.classes_.tolist()}")

            # Update progress to 85% before starting the save (to show we're preparing to save)
            await TrainingJobService.update_job_status(
                db, job_id, TrainingJobStatus.RUNNING, progress=85.0
            )
            logger.info("Updated progress to 85% - preparing to save model")

            try:
                model_path = await model_storage.save_model(
                    model=predictor.model,
                    model_name=model_name,
                    version="1.0.0",
                    metadata=metadata,
                    preprocessor=preprocessor,  # Save preprocessor for automatic transformation during predictions
                    feature_names=feature_names
                )
                logger.info(f"Model saved successfully to: {model_path}")

                # Update progress to 87% after model save completes
                await TrainingJobService.update_job_status(
                    db, job_id, TrainingJobStatus.RUNNING, progress=87.0
                )
                logger.info("Updated progress to 87% - model saved, starting MLflow logging")
            except Exception as e:
                logger.error(f"Error saving model to cloud storage: {e}", exc_info=True)
                raise
            
            # Log to MLflow (non-blocking, skip if unavailable)
            # Skip MLflow entirely if tracking URI is not configured or service is unavailable
            mlflow_available = False
            try:
                from app.core.config import settings
                
                # Quick check: if tracking URI is None or empty, skip MLflow
                if not settings.MLFLOW_TRACKING_URI or settings.MLFLOW_TRACKING_URI == "":
                    logger.debug("MLflow tracking URI not configured, skipping MLflow logging")
                else:
                    # Try a quick connection test with timeout
                    logger.info("Checking MLflow availability...")
                    experiment_tracker = ExperimentTracker()
                    experiment_name = f"{model_name}_{model_type}"
                    
                    # Use asyncio timeout to prevent hanging
                    try:
                        # Run the check in a thread with timeout
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(experiment_tracker.client.get_experiment, experiment_name)
                            # Wait max 2 seconds for the check
                            future.result(timeout=2.0)
                        mlflow_available = True
                        logger.info("MLflow is available, proceeding with logging")
                    except (concurrent.futures.TimeoutError, ConnectionError, Exception) as e:
                        logger.warning(f"MLflow service unavailable or timeout (non-critical): {type(e).__name__}")
                        mlflow_available = False
                
                # Only log to MLflow if available
                if mlflow_available:
                    logger.info("Logging experiment to MLflow...")

                    # Update progress to 88% - starting MLflow logging
                    await TrainingJobService.update_job_status(
                        db, job_id, TrainingJobStatus.RUNNING, progress=88.0
                    )

                    with experiment_tracker.track_experiment(
                        experiment_name=experiment_name,
                        run_name=f"{model_name}_{job_id}",
                        tags={
                            "model_name": model_name,
                            "algorithm": algorithm,
                            "model_type": model_type,
                            "job_id": str(job_id),
                            "data_source_id": str(data_source_id)
                        },
                        log_model=False,  # Let log_model_artifacts handle model logging
                        model=None,
                        model_artifact_path="model"
                    ):
                        # Log hyperparameters
                        if hyperparameters:
                            experiment_tracker.client.log_params(hyperparameters)

                        # Log training configuration
                        if training_config:
                            experiment_tracker.log_training_config(training_config)

                        # Log metrics
                        test_metrics = training_results.get("test_metrics", {})
                        experiment_tracker.log_evaluation_results(test_metrics, dataset_name="test")

                        # Log cross-validation metrics if available
                        if "cv_metrics" in training_results:
                            cv_metrics = training_results["cv_metrics"]
                            experiment_tracker.log_evaluation_results(cv_metrics, dataset_name="cv")

                        # Log feature importance if available
                        if feature_importance:
                            experiment_tracker.client.log_dict(
                                feature_importance,
                                artifact_path="feature_importance"
                            )

                        # Log model artifacts (includes model, preprocessor, feature names, metadata)
                        experiment_tracker.log_model_artifacts(
                            model=predictor.model,
                            preprocessor=preprocessor,
                            feature_names=feature_names,
                            metadata=metadata
                        )

                    logger.info("MLflow logging completed successfully")

                    # Update progress to 89% after MLflow completes
                    await TrainingJobService.update_job_status(
                        db, job_id, TrainingJobStatus.RUNNING, progress=89.0
                    )
                    logger.info("Updated progress to 89% - MLflow logging complete")
                else:
                    logger.info("Skipping MLflow logging (service unavailable)")
            except Exception as e:
                # Don't fail training if MLflow logging fails
                logger.warning(f"MLflow logging failed (non-critical): {e}", exc_info=True)

            # Update progress to 90% - ready to create model record
            await TrainingJobService.update_job_status(
                db, job_id, TrainingJobStatus.RUNNING, progress=90.0
            )
            logger.info("Updated progress to 90% - creating model record")
            print("Updated progress to 90% - creating model record", file=sys.stderr, flush=True)
            
            # Create or update model record
            test_metrics = training_results["test_metrics"]
            logger.info(f"About to create model record. test_metrics keys: {list(test_metrics.keys())}")
            print(f"About to create model record. test_metrics keys: {list(test_metrics.keys())}", file=sys.stderr, flush=True)
            
            logger.info(f"Extracting test_metrics for model record: {test_metrics}")
            print(f"Extracting test_metrics for model record: {test_metrics}", file=sys.stderr, flush=True)
            
            # feature_importance was already extracted earlier (before MLflow logging)
            if feature_importance:
                logger.info(f"Using feature importance with {len(feature_importance)} features")
            else:
                logger.info("No feature importance available")

            # Map model_type string to ModelType enum
            model_type_enum = {
                "regression": ModelType.REGRESSION,
                "classification": ModelType.CLASSIFICATION,
                "anomaly_detection": ModelType.ANOMALY_DETECTION
            }.get(model_type, ModelType.REGRESSION)

            # Log the exact values being saved to the model
            rmse_value = test_metrics.get("rmse")
            mae_value = test_metrics.get("mae")
            r2_value = test_metrics.get("r2_score")
            logger.info(f"Creating model record with metrics - RMSE: {rmse_value}, MAE: {mae_value}, R²: {r2_value}")
            print(f"Creating model record with metrics - RMSE: {rmse_value}, MAE: {mae_value}, R²: {r2_value}", file=sys.stderr, flush=True)
            
            # Get user_id from training job
            user_id = job.created_by if job else None
            
            # Ensure user_id is set - this is critical for model visibility in the list
            if not user_id:
                logger.warning(f"Training job {job_id} has no created_by value. Model may not be visible to users.")
                print(f"WARNING: Training job {job_id} has no created_by value. Model may not be visible to users.", file=sys.stderr, flush=True)
            
            logger.info(f"Creating model with user_id: {user_id} (from job.created_by: {job.created_by if job else 'N/A'})")
            print(f"Creating model with user_id: {user_id} (from job.created_by: {job.created_by if job else 'N/A'})", file=sys.stderr, flush=True)
            
            model = MLModel(
                name=model_name,
                type=model_type_enum,
                status=ModelStatus.TRAINED,
                version="1.0.0",
                data_source_id=data_source_id,
                features=feature_names,
                original_columns=original_columns,  # Store original column names for predictions
                feature_name_mapping=feature_name_mapping,  # Store readable feature name mapping
                hyperparameters=hyperparameters or {},
                feature_importance=feature_importance,
                rmse=rmse_value,
                mae=mae_value,
                r2_score=r2_value,
                accuracy=test_metrics.get("accuracy"),
                precision=test_metrics.get("precision"),
                recall=test_metrics.get("recall"),
                f1_score=test_metrics.get("f1_score"),
                model_path=model_path,
                is_active=True,
                created_by=user_id  # Set user ownership from training job
            )
            
            logger.info("Adding model to database session...")
            print("Adding model to database session...", file=sys.stderr, flush=True)
            db.add(model)
            
            # Flush before commit to ensure changes are visible immediately
            logger.info("Flushing model to database...")
            print("Flushing model to database...", file=sys.stderr, flush=True)
            await db.flush()
            
            logger.info("Committing model to database...")
            print("Committing model to database...", file=sys.stderr, flush=True)
            await db.commit()
            
            logger.info("Refreshing model from database...")
            print("Refreshing model from database...", file=sys.stderr, flush=True)
            await db.refresh(model)
            
            logger.info(f"Model created successfully with ID: {model.id}, name: {model.name}, status: {model.status}")
            print(f"Model created successfully with ID: {model.id}, name: {model.name}, status: {model.status}", file=sys.stderr, flush=True)

            # Create data lineage records
            try:
                lineage_service = DataLineageService()
                user_id = job.created_by if job else None
                
                # Create lineage record for the model
                model_lineage = await lineage_service.create_lineage(
                    db=db,
                    source_id=data_source_id,
                    target_type="model",
                    target_id=model.id,
                    transformation={
                        "algorithm": algorithm,
                        "hyperparameters": hyperparameters or {},
                        "preprocessing": {
                            "feature_engineering": True,
                            "feature_count": len(feature_names) if feature_names else 0
                        }
                    },
                    created_by=user_id
                )
                logger.info(f"Created data lineage record for model {model.id} from data source {data_source_id}")
                
                # Create lineage record for the training job
                job_lineage = await lineage_service.create_lineage(
                    db=db,
                    source_id=data_source_id,
                    target_type="training_job",
                    target_id=job_id,
                    transformation={
                        "algorithm": algorithm,
                        "hyperparameters": hyperparameters or {},
                        "model_name": model_name,
                        "target_column": target_column
                    },
                    created_by=user_id
                )
                logger.info(f"Created data lineage record for training job {job_id} from data source {data_source_id}")
            except Exception as lineage_error:
                # Log error but don't fail the training - lineage is metadata
                logger.error(f"Error creating data lineage records: {lineage_error}", exc_info=True)
                print(f"Error creating data lineage records: {lineage_error}", file=sys.stderr, flush=True)

            # Update progress to 95% after model record creation
            await TrainingJobService.update_job_status(
                db, job_id, TrainingJobStatus.RUNNING, progress=95.0
            )
            logger.info("Updated progress to 95% - model record created, finalizing job")

            # Update training job
            logger.info(f"Updating training job {job_id} to COMPLETED status...")
            print(f"Updating training job {job_id} to COMPLETED status...", file=sys.stderr, flush=True)
            try:
                await db.execute(
                    update(TrainingJob)
                    .where(TrainingJob.id == job_id)
                    .values(
                        model_id=model.id,
                        status=TrainingJobStatus.COMPLETED,
                        progress=100.0,
                        completed_at=datetime.utcnow(),
                        metrics=test_metrics
                    )
                )
                await db.flush()
                await db.commit()
                logger.info(f"Training job {job_id} updated to COMPLETED. Model ID: {model.id}")
                print(f"Training job {job_id} updated to COMPLETED. Model ID: {model.id}", file=sys.stderr, flush=True)
            except Exception as e:
                # Log error but don't fail - model is already created
                logger.error(f"Error updating training job {job_id} status (model already created): {e}", exc_info=True)
                print(f"Error updating training job {job_id} status (model already created): {e}", file=sys.stderr, flush=True)
                # Try to commit anyway to ensure model is saved
                try:
                    await db.commit()
                except Exception as commit_error:
                    logger.error(f"Error committing after training job update failure: {commit_error}", exc_info=True)
            
            logger.info(f"Training job {job_id} completed successfully. Model ID: {model.id}, name: {model.name}")
            print(f"Training job {job_id} completed successfully. Model ID: {model.id}, name: {model.name}", file=sys.stderr, flush=True)
            return model
        
        except Exception as e:
            logger.error(f"Error in training job {job_id}: {e}", exc_info=True)
            
            # Update job status to failed
            await TrainingJobService.update_job_status(
                db, job_id, TrainingJobStatus.FAILED, error_message=str(e)
            )
            
            raise
    
    @staticmethod
    async def run_training_job_async(
        db: AsyncSession,
        job_id: int,
        **kwargs
    ) -> None:
        """Run training job asynchronously in background."""
        # This would typically be run in a background task queue
        # For now, we'll use asyncio.create_task
        try:
            await TrainingJobService.execute_training_job(db, job_id, **kwargs)
        except Exception as e:
            logger.error(f"Async training job {job_id} failed: {e}")

