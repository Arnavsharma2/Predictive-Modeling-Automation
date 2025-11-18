"""
Batch prediction job service.
"""
import asyncio
import io
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from app.models.database.batch_jobs import BatchPredictionJob, BatchJobStatus
from app.models.database.ml_models import MLModel, ModelStatus, ModelType
from app.models.database.data_sources import DataSource
from app.models.database.data_points import DataPoint
from app.ml.storage.model_storage import model_storage
from app.ml.models.predictor import RegressionPredictor
from app.ml.models.classifier import Classifier
from app.ml.models.anomaly_detector import AnomalyDetector
from app.storage.cloud_storage import cloud_storage
from app.core.logging import get_logger
from app.core.websocket import manager as ws_manager

logger = get_logger(__name__)


class BatchPredictionJobService:
    """Service for managing batch prediction jobs."""
    
    @staticmethod
    async def create_batch_job(
        db: AsyncSession,
        model_id: int,
        input_type: str = "data_source",
        data_source_id: Optional[int] = None,
        input_config: Optional[Dict[str, Any]] = None,
        job_name: Optional[str] = None,
        result_format: str = "csv",
        scheduled_at: Optional[datetime] = None,
        created_by: Optional[int] = None
    ) -> BatchPredictionJob:
        """
        Create a new batch prediction job.
        
        Args:
            db: Database session
            model_id: ID of model to use for predictions
            input_type: Type of input (data_source, file, query)
            data_source_id: ID of data source (if input_type is data_source)
            input_config: Input configuration (file path, query, etc.)
            job_name: Optional name for the job
            result_format: Format for results (csv, json, parquet)
            scheduled_at: When to run the job (None for immediate)
            created_by: ID of user creating the job
            
        Returns:
            Created batch prediction job
        """
        # Validate model exists and is ready
        model = await db.get(MLModel, model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        if model.status not in [ModelStatus.TRAINED, ModelStatus.DEPLOYED]:
            raise ValueError(f"Model {model_id} is not ready for predictions (status: {model.status})")
        
        # Validate data source if provided
        if input_type == "data_source":
            if not data_source_id:
                raise ValueError("data_source_id is required for data_source input type")
            data_source = await db.get(DataSource, data_source_id)
            if not data_source:
                raise ValueError(f"Data source {data_source_id} not found")
        
        # Validate input_config for file input type
        if input_type == "file":
            if not input_config or not input_config.get("file_path"):
                raise ValueError("file_path is required in input_config for file input type")
        
        # Create job
        job = BatchPredictionJob(
            model_id=model_id,
            data_source_id=data_source_id,
            status=BatchJobStatus.PENDING if not scheduled_at else BatchJobStatus.QUEUED,
            job_name=job_name or f"Batch prediction for model {model_id}",
            input_type=input_type,
            input_config=input_config or {},
            result_format=result_format,
            scheduled_at=scheduled_at,
            created_by=created_by
        )
        
        db.add(job)
        await db.commit()
        await db.refresh(job)
        
        logger.info(f"Created batch prediction job {job.id} for model {model_id}")
        return job
    
    @staticmethod
    async def get_batch_job(db: AsyncSession, job_id: int) -> Optional[BatchPredictionJob]:
        """Get batch prediction job by ID."""
        return await db.get(BatchPredictionJob, job_id)
    
    @staticmethod
    async def update_job_status(
        db: AsyncSession,
        job_id: int,
        status: BatchJobStatus,
        progress: Optional[float] = None,
        total_records: Optional[int] = None,
        processed_records: Optional[int] = None,
        failed_records: Optional[int] = None,
        error_message: Optional[str] = None,
        result_path: Optional[str] = None
    ) -> None:
        """
        Update batch prediction job status and broadcast to WebSocket clients.
        
        Args:
            db: Database session
            job_id: Batch job ID
            status: New status
            progress: Progress percentage (0-100)
            total_records: Total records to process
            processed_records: Records processed so far
            failed_records: Records that failed
            error_message: Error message if failed
            result_path: Path to results file
        """
        update_data = {
            "status": status,
            "updated_at": datetime.now(timezone.utc)
        }
        
        if progress is not None:
            update_data["progress"] = progress
        
        if total_records is not None:
            update_data["total_records"] = total_records
        
        if processed_records is not None:
            update_data["processed_records"] = processed_records
        
        if failed_records is not None:
            update_data["failed_records"] = failed_records
        
        if error_message:
            update_data["error_message"] = error_message
        
        if result_path:
            update_data["result_path"] = result_path
        
        if status == BatchJobStatus.RUNNING and not update_data.get("started_at"):
            update_data["started_at"] = datetime.now(timezone.utc)
        elif status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED]:
            update_data["completed_at"] = datetime.now(timezone.utc)
        
        await db.execute(
            update(BatchPredictionJob).where(BatchPredictionJob.id == job_id).values(**update_data)
        )
        await db.flush()
        await db.commit()
        
        # Broadcast update to WebSocket clients
        await ws_manager.broadcast_job_update(
            f"batch_{job_id}",
            {
                "type": "batch_job_update",
                "job_id": job_id,
                "status": status.value,
                "progress": progress,
                "total_records": total_records,
                "processed_records": processed_records,
                "failed_records": failed_records,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        )
    
    @staticmethod
    async def cancel_job(db: AsyncSession, job_id: int) -> bool:
        """
        Cancel a batch prediction job.
        
        Args:
            db: Database session
            job_id: Batch job ID
            
        Returns:
            True if cancelled, False if job not found or already completed
        """
        job = await db.get(BatchPredictionJob, job_id)
        if not job:
            return False
        
        if job.status in [BatchJobStatus.COMPLETED, BatchJobStatus.FAILED, BatchJobStatus.CANCELLED]:
            return False
        
        await BatchPredictionJobService.update_job_status(
            db, job_id, BatchJobStatus.CANCELLED
        )
        
        logger.info(f"Cancelled batch prediction job {job_id}")
        return True
    
    @staticmethod
    async def load_input_data(
        db: AsyncSession,
        job: BatchPredictionJob,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load input data for batch prediction.
        
        Args:
            db: Database session
            job: Batch prediction job
            limit: Maximum number of records to load
            
        Returns:
            DataFrame with input data
        """
        if job.input_type == "data_source":
            if not job.data_source_id:
                raise ValueError("Data source ID is required for data_source input type")
            
            # Query data points
            query = select(DataPoint).where(DataPoint.source_id == job.data_source_id)
            
            if limit:
                query = query.limit(limit)
            
            result = await db.execute(query)
            data_points = result.scalars().all()
            
            if not data_points:
                raise ValueError(f"No data points found for source {job.data_source_id}")
            
            # Convert to DataFrame
            records = []
            for point in data_points:
                record = point.data.copy()
                record['timestamp'] = point.timestamp
                records.append(record)
            
            return pd.DataFrame(records)
        
        elif job.input_type == "file":
            # Load from file path in input_config
            if not job.input_config:
                raise ValueError("input_config is required for file input type")
            file_path = job.input_config.get("file_path")
            if not file_path:
                raise ValueError("file_path is required in input_config for file input type")
            
            # Download from cloud storage or read from local path
            try:
                file_content = await cloud_storage.download_file(file_path)
                # Try to read as CSV first, then JSON
                try:
                    return pd.read_csv(io.BytesIO(file_content))
                except (pd.errors.ParserError, ValueError):
                    import json
                    data = json.loads(file_content.decode('utf-8'))
                    return pd.DataFrame(data)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                raise
        
        else:
            raise ValueError(f"Unsupported input type: {job.input_type}")
    
    @staticmethod
    async def execute_batch_prediction(
        db: AsyncSession,
        job_id: int,
        batch_size: int = 1000
    ) -> BatchPredictionJob:
        """
        Execute a batch prediction job.
        
        Args:
            db: Database session
            job_id: Batch job ID
            batch_size: Number of records to process per batch
            
        Returns:
            Updated batch prediction job
        """
        job = await BatchPredictionJobService.get_batch_job(db, job_id)
        if not job:
            raise ValueError(f"Batch prediction job {job_id} not found")
        
        # Update status to running
        await BatchPredictionJobService.update_job_status(
            db, job_id, BatchJobStatus.RUNNING, progress=0.0
        )
        
        try:
            # Load model
            model = await db.get(MLModel, job.model_id)
            if not model or not model.model_path:
                raise ValueError(f"Model {job.model_id} not found or has no model path")
            
            model_package = await model_storage.load_model(model.model_path, return_package=True)
            
            # Extract components
            if isinstance(model_package, dict):
                ml_model = model_package.get("model")
                preprocessor = model_package.get("preprocessor")
            else:
                ml_model = model_package
                preprocessor = None
            
            # Load input data
            logger.info(f"Loading input data for batch job {job_id}...")
            input_df = await BatchPredictionJobService.load_input_data(db, job)
            total_records = len(input_df)
            
            # Log expected columns if available
            if model.original_columns:
                logger.info(f"Model expects {len(model.original_columns)} original columns: {model.original_columns[:10]}...")
                logger.info(f"Input data has {len(input_df.columns)} columns: {list(input_df.columns)[:10]}...")
                missing_cols = [col for col in model.original_columns if col not in input_df.columns]
                if missing_cols:
                    logger.warning(f"Input file is missing {len(missing_cols)} expected columns: {missing_cols[:10]}...")
                    logger.warning("Missing columns will be filled with defaults, which may affect prediction accuracy.")
            
            await BatchPredictionJobService.update_job_status(
                db, job_id, BatchJobStatus.RUNNING,
                progress=5.0,
                total_records=total_records,
                processed_records=0
            )
            
            logger.info(f"Processing {total_records} records in batches of {batch_size}")
            
            # Prepare predictor based on model type
            if model.type == ModelType.CLASSIFICATION:
                predictor = Classifier()
            elif model.type == ModelType.ANOMALY_DETECTION:
                contamination = model.hyperparameters.get("contamination", 0.1) if model.hyperparameters else 0.1
                predictor = AnomalyDetector(contamination=contamination)
            else:  # REGRESSION
                predictor = RegressionPredictor()
            
            predictor.model = ml_model
            predictor.is_trained = True
            predictor.feature_names = model.features
            
            # Process in batches
            all_predictions = []
            all_inputs = []
            processed = 0
            failed = 0
            
            for i in range(0, total_records, batch_size):
                batch_df = input_df.iloc[i:i+batch_size].copy()
                
                try:
                    # Prepare features
                    if preprocessor:
                        # Use original columns if available
                        if model.original_columns:
                            # Check for missing columns and fill them with appropriate defaults
                            missing_cols = [col for col in model.original_columns if col not in batch_df.columns]
                            if missing_cols:
                                logger.warning(f"Missing columns in batch: {missing_cols}. Filling with defaults.")
                                
                                # Fill missing columns with appropriate defaults
                                for col in missing_cols:
                                    # For timestamp/date columns, use NaT (Not a Time)
                                    if any(keyword in col.lower() for keyword in ['timestamp', 'time', 'date']):
                                        batch_df[col] = pd.NaT
                                    # For numeric-looking columns, use NaN
                                    elif any(c.isdigit() for c in col) or col.startswith('num_') or col.endswith('_num'):
                                        batch_df[col] = np.nan
                                    # For categorical/string columns, use empty string
                                    else:
                                        batch_df[col] = ''
                            
                            # Reorder to match training order
                            batch_df = batch_df[model.original_columns]
                        
                        # Infer and convert data types (numeric columns to float, categorical stay as object/string)
                        for col in batch_df.columns:
                            try:
                                batch_df[col] = pd.to_numeric(batch_df[col], errors='ignore')
                            except (ValueError, TypeError):
                                pass  # Keep as string/object for categorical columns
                        
                        # Apply preprocessing
                        batch_df_processed = preprocessor.transform(batch_df)
                    else:
                        # Use transformed features
                        if model.features:
                            batch_df_processed = batch_df[model.features]
                        else:
                            batch_df_processed = batch_df
                    
                    # Make predictions
                    if model.type == ModelType.CLASSIFICATION:
                        results = predictor.predict_with_proba(batch_df_processed)
                        predictions = results["predictions"]
                        probabilities = results["probabilities"]
                        
                        # Format predictions
                        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
                            all_predictions.append({
                                "prediction": str(pred),
                                "probabilities": {str(k): float(v) for k, v in zip(results["classes"], prob)} if results.get("classes") else {}
                            })
                    elif model.type == ModelType.ANOMALY_DETECTION:
                        predictions = predictor.predict(batch_df_processed)
                        scores = predictor.predict_scores(batch_df_processed) if hasattr(predictor, 'predict_scores') else None
                        
                        for idx, pred in enumerate(predictions):
                            result = {
                                "is_anomaly": bool(pred),
                                "anomaly_score": float(scores[idx]) if scores is not None else None
                            }
                            all_predictions.append(result)
                    else:  # REGRESSION
                        predictions = predictor.predict(batch_df_processed)
                        for pred in predictions:
                            all_predictions.append({"prediction": float(pred)})
                    
                    # Store input data for results
                    all_inputs.append(batch_df)
                    processed += len(batch_df)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i}-{i+batch_size}: {e}", exc_info=True)
                    failed += len(batch_df)
                
                # Update progress
                progress = min(90.0, 5.0 + (processed / total_records) * 85.0)
                await BatchPredictionJobService.update_job_status(
                    db, job_id, BatchJobStatus.RUNNING,
                    progress=progress,
                    processed_records=processed,
                    failed_records=failed
                )
            
            # Combine results
            logger.info(f"Combining results: {len(all_predictions)} predictions, {processed} processed, {failed} failed")
            
            # Create results DataFrame
            results_df = pd.DataFrame(all_inputs[0]) if all_inputs else pd.DataFrame()
            if len(all_inputs) > 1:
                results_df = pd.concat(all_inputs, ignore_index=True)
            
            # Add predictions column
            if model.type == ModelType.CLASSIFICATION:
                results_df['predicted_class'] = [p['prediction'] for p in all_predictions]
                results_df['probabilities'] = [p['probabilities'] for p in all_predictions]
            elif model.type == ModelType.ANOMALY_DETECTION:
                results_df['is_anomaly'] = [p['is_anomaly'] for p in all_predictions]
                if all_predictions[0].get('anomaly_score') is not None:
                    results_df['anomaly_score'] = [p['anomaly_score'] for p in all_predictions]
            else:  # REGRESSION
                results_df['prediction'] = [p['prediction'] for p in all_predictions]
            
            # Save results to cloud storage
            logger.info("Saving results to cloud storage...")
            result_filename = f"batch_predictions_{job_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            if job.result_format == "csv":
                result_content = results_df.to_csv(index=False).encode('utf-8')
                result_path = f"batch_results/{result_filename}.csv"
                content_type = "text/csv"
            elif job.result_format == "json":
                result_content = results_df.to_json(orient='records').encode('utf-8')
                result_path = f"batch_results/{result_filename}.json"
                content_type = "application/json"
            elif job.result_format == "parquet":
                buffer = io.BytesIO()
                results_df.to_parquet(buffer, index=False)
                result_content = buffer.getvalue()
                result_path = f"batch_results/{result_filename}.parquet"
                content_type = "application/octet-stream"
            else:
                raise ValueError(f"Unsupported result format: {job.result_format}")
            
            await cloud_storage.upload_file(
                file_content=result_content,
                file_path=result_path,
                content_type=content_type
            )
            
            logger.info(f"Results saved to: {result_path}")
            
            # Update job status
            await BatchPredictionJobService.update_job_status(
                db, job_id, BatchJobStatus.COMPLETED,
                progress=100.0,
                processed_records=processed,
                failed_records=failed,
                result_path=result_path
            )
            
            # Refresh job
            await db.refresh(job)
            return job
        
        except Exception as e:
            logger.error(f"Error executing batch prediction job {job_id}: {e}", exc_info=True)
            await BatchPredictionJobService.update_job_status(
                db, job_id, BatchJobStatus.FAILED,
                error_message=str(e)
            )
            raise

