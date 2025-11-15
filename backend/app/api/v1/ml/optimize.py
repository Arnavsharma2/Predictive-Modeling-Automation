"""
Hyperparameter optimization API endpoints.
"""
from typing import Optional, Annotated
import asyncio
import sys
import optuna
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.core.database import get_db, get_db_session
from app.core.logging import get_logger
from app.core.config import settings
from app.core.permissions import has_resource_access
from app.middleware.auth import get_current_user
from app.models.schemas.optimization import (
    OptimizationRequest,
    OptimizationResponse,
    StudySummaryResponse,
)
from app.ml.optimization.hyperparameter_search import HyperparameterSearch
from app.models.database.ml_models import MLModel
from app.models.database.data_sources import DataSource
from app.models.database.training_jobs import TrainingJob, TrainingJobStatus
from app.services.training_job import TrainingJobService
from app.models.database.users import User

logger = get_logger(__name__)

router = APIRouter(prefix="/optimize", tags=["Optimization"])


async def _run_optimization_in_background(
    study_name: str,
    model_id: Optional[int],
    data_source_id: Optional[int],
    target_column: Optional[str],
    algorithm: str,
    search_space: dict,
    metric_name: str,
    direction: str,
    n_trials: int,
    timeout: Optional[float]
):
    """
    Run hyperparameter optimization in background with its own database session.
    """
    logger.info("="*100)
    logger.info(f"BACKGROUND OPTIMIZATION TASK STARTED - STUDY: {study_name}")
    logger.info("="*100)
    
    try:
        async with get_db_session() as db:
            # Load training data
            if model_id:
                # Load from existing model
                model = await db.get(MLModel, model_id)
                if not model:
                    raise ValueError(f"Model {model_id} not found")
                if not model.data_source_id:
                    raise ValueError(f"Model {model_id} does not have associated data source")
                
                data_source_id = model.data_source_id
                # Try to get target column from training job's training_config or meta_data
                if not target_column:
                    # First, try to find training job by model_id (for retrained models)
                    training_job_query = select(TrainingJob).where(
                        TrainingJob.model_id == model_id
                    ).order_by(desc(TrainingJob.created_at)).limit(1)
                    result = await db.execute(training_job_query)
                    training_job = result.scalar_one_or_none()
                    
                    # If not found by model_id, try to find by data_source_id
                    # (since model_id might not be set when training job is created)
                    if not training_job and model.data_source_id:
                        training_job_query = select(TrainingJob).where(
                            TrainingJob.data_source_id == model.data_source_id
                        ).order_by(desc(TrainingJob.created_at)).limit(1)
                        result = await db.execute(training_job_query)
                        training_job = result.scalar_one_or_none()
                    
                    if training_job:
                        # Try training_config first, then meta_data
                        if training_job.training_config:
                            target_column = training_job.training_config.get("target_column")
                        if not target_column and training_job.meta_data:
                            target_column = training_job.meta_data.get("target_column")
                
                if not target_column:
                    raise ValueError("Target column is required when optimizing existing model. Please provide target_column in the request or ensure the model has an associated training job with target_column in training_config.")
            
            if not data_source_id or not target_column:
                raise ValueError("Both data_source_id and target_column are required")
            
            # Load training data
            logger.info(f"Loading training data from source {data_source_id}, target: {target_column}")
            X, y = await TrainingJobService.load_training_data(
                db, data_source_id, target_column
            )
            
            logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")
            
            # Determine problem type from target
            if y.dtype == 'object' or y.dtype.name == 'category':
                problem_type = "classification"
            else:
                problem_type = "regression"
            
            logger.info(f"Detected problem type: {problem_type}")
            
            # Preprocess data using AutoPreprocessor (same as training pipeline)
            logger.info("Preprocessing data using AutoPreprocessor...")
            from app.ml.preprocessing.auto_preprocessor import AutoPreprocessor
            import pandas as pd
            
            # Combine X and y for preprocessing
            data_df = X.copy()
            data_df[target_column] = y
            
            # Create and fit preprocessor
            preprocessor = AutoPreprocessor(
                target_column=target_column,
                task=problem_type,
                enable_feature_engineering=True,
                enable_outlier_handling=True,
                enable_feature_selection=True,
                verbose=False  # Reduce verbosity for optimization
            )
            
            # Fit and transform
            result = preprocessor.fit_transform(data_df, y)
            if isinstance(result, tuple):
                X_processed, y_processed = result
            else:
                X_processed = result
                y_processed = y
            
            # Convert to numpy arrays for sklearn compatibility
            if isinstance(X_processed, pd.DataFrame):
                X_processed = X_processed.values
            if isinstance(y_processed, pd.Series):
                y_processed = y_processed.values
            
            logger.info(f"Preprocessed data: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
            
            # Update X and y to use preprocessed versions
            X = X_processed
            y = y_processed
            
            # Initialize optimization service
            search_service = HyperparameterSearch(
                study_name=study_name,
                n_trials=n_trials,
                timeout=timeout
            )
            
            # Get search space
            if not search_space:
                search_space = search_service.get_search_space(
                    algorithm=algorithm,
                    custom_space=None
                )
            
            if not search_space:
                raise ValueError(f"No search space defined for algorithm '{algorithm}'")
            
            # Create objective function
            from app.ml.trainers.regression_trainer import RegressionTrainer
            from app.ml.trainers.classification_trainer import ClassificationTrainer
            from sklearn.model_selection import cross_val_score
            
            def objective_func(trial, hyperparameters: dict):
                """Objective function for optimization."""
                try:
                    # Create trainer
                    if problem_type == "regression":
                        trainer = RegressionTrainer(algorithm=algorithm)
                    else:
                        trainer = ClassificationTrainer(algorithm=algorithm)
                    
                    # Create model with hyperparameters
                    model = trainer._create_model(hyperparameters=hyperparameters)
                    
                    # Cross-validation with appropriate metric
                    # Use n_jobs=1 to avoid resource contention and deadlocks
                    # when multiple trials are running
                    if problem_type == "regression":
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        if metric_name == "rmse":
                            scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error', n_jobs=1)
                            mean_score = -scores.mean()  # Negate to get positive RMSE
                        elif metric_name == "mae":
                            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
                            mean_score = -scores.mean()  # Negate to get positive MAE
                        elif metric_name == "r2_score":
                            scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=1)
                            mean_score = scores.mean()  # R² is already positive
                        else:
                            # Default to RMSE
                            scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error', n_jobs=1)
                            mean_score = -scores.mean()
                    else:  # classification
                        from sklearn.metrics import accuracy_score, f1_score
                        if metric_name == "accuracy":
                            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=1)
                            mean_score = scores.mean()
                        elif metric_name == "f1_score":
                            scores = cross_val_score(model, X, y, cv=5, scoring='f1', n_jobs=1)
                            mean_score = scores.mean()
                        else:
                            # Default to accuracy
                            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=1)
                            mean_score = scores.mean()
                    
                    # Optuna's direction is set when creating the study
                    # For "minimize", Optuna will minimize the returned value
                    # For "maximize", Optuna will maximize the returned value
                    # So we just return the metric value as-is
                    return mean_score
                    
                except Exception as e:
                    logger.error(f"Error in optimization trial: {e}", exc_info=True)
                    # Return bad score based on direction
                    # For minimize: return large positive value
                    # For maximize: return large negative value
                    return float('inf') if direction == "minimize" else float('-inf')
            
            # Create objective wrapper for Optuna
            def optuna_objective(trial):
                """Optuna objective function wrapper."""
                # Suggest hyperparameters
                params = search_service.optimizer.suggest_hyperparameters(trial, search_space)
                # Call our objective function
                return objective_func(trial, params)
            
            # Set direction and create study
            search_service.optimizer.direction = direction
            study = search_service.optimizer.create_study(study_name=study_name)
            
            # Run optimization with error handling
            logger.info(f"Starting optimization with {n_trials} trials...")
            try:
                study = search_service.optimizer.optimize(
                    objective_func=optuna_objective,
                    n_trials=n_trials,
                    timeout=timeout
                )
            except (KeyboardInterrupt, Exception) as e:
                logger.warning(f"Optimization interrupted or failed: {e}. Using best trial so far.")
                # Continue with best trial found so far if study exists
                if search_service.optimizer.study is None:
                    raise
                study = search_service.optimizer.study
                
                # Check if we have any completed trials
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if not completed_trials:
                    logger.error("No completed trials found. Optimization failed.")
                    raise RuntimeError(f"Hyperparameter optimization failed: {str(e)}")
            
            # Get results
            best_params = search_service.optimizer.get_best_params()
            best_value = search_service.optimizer.get_best_value()
            best_trial = search_service.optimizer.get_best_trial()
            
            results = {
                "best_params": best_params,
                "best_value": best_value,
                "best_trial_number": best_trial.number,
                "n_trials": len(study.trials),
                "study_summary": search_service.optimizer.get_study_summary()
            }
            
            logger.info(f"Optimization completed. Best value: {results['best_value']:.4f}")
            logger.info(f"Best parameters: {results['best_params']}")
            
            return results
            
    except Exception as e:
        logger.error(f"Background optimization task failed: {e}", exc_info=True)
        print(f"Background optimization task failed: {e}", file=sys.stderr, flush=True)
        raise


@router.post("", response_model=OptimizationResponse, status_code=status.HTTP_201_CREATED)
async def optimize_hyperparameters(
    request: OptimizationRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Start hyperparameter optimization for a model.
    
    Runs optimization as a background job and returns immediately with study name.
    """
    try:
        # Validate inputs
        if request.n_trials < 1 or request.n_trials > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="n_trials must be between 1 and 1000"
            )
        
        if request.timeout and request.timeout < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="timeout must be non-negative"
            )
        
        # Validate algorithm
        valid_algorithms = ["random_forest", "xgboost", "lightgbm", "catboost", "gradient_boosting"]
        if request.algorithm not in valid_algorithms:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid algorithm. Must be one of: {', '.join(valid_algorithms)}"
            )
        
        # Validate direction
        if request.direction not in ["minimize", "maximize"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="direction must be 'minimize' or 'maximize'"
            )
        
        # Get model or data source
        data_source_id = None
        target_column = None
        
        if request.model_id:
            # Validate model exists and user has access
            model = await db.get(MLModel, request.model_id)
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {request.model_id} not found"
                )
            
            # Check user has access
            if not has_resource_access(current_user, model.created_by, model.shared_with):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have access to this model"
                )
            
            if model.data_source_id:
                data_source_id = model.data_source_id
                # Try to get target column from training job's training_config or meta_data
                if not target_column:
                    # First, try to find training job by model_id (for retrained models)
                    training_job_query = select(TrainingJob).where(
                        TrainingJob.model_id == request.model_id
                    ).order_by(desc(TrainingJob.created_at)).limit(1)
                    result = await db.execute(training_job_query)
                    training_job = result.scalar_one_or_none()
                    
                    # If not found by model_id, try to find by data_source_id
                    # (since model_id might not be set when training job is created)
                    if not training_job and model.data_source_id:
                        training_job_query = select(TrainingJob).where(
                            TrainingJob.data_source_id == model.data_source_id
                        ).order_by(desc(TrainingJob.created_at)).limit(1)
                        result = await db.execute(training_job_query)
                        training_job = result.scalar_one_or_none()
                    
                    if training_job:
                        # Try training_config first, then meta_data
                        if training_job.training_config:
                            target_column = training_job.training_config.get("target_column")
                        if not target_column and training_job.meta_data:
                            target_column = training_job.meta_data.get("target_column")
        else:
            # For new model optimization, we need data_source_id and target_column
            # But these aren't in the request schema, so we'll need to add them
            # For now, require model_id
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="model_id is required. Optimization for new models requires data_source_id and target_column (not yet supported)"
            )
        
        if not data_source_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model must have associated data source"
            )
        
        # Validate data source access
        data_source = await db.get(DataSource, data_source_id)
        if not data_source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data source {data_source_id} not found"
            )
        
        if not has_resource_access(current_user, data_source.created_by, data_source.shared_with):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this data source"
            )
        
        # Generate study name if not provided
        study_name = request.study_name or f"{request.algorithm}_optimization_{current_user.id}_{int(asyncio.get_event_loop().time())}"
        
        # Get search space
        search_service = HyperparameterSearch(
            study_name=study_name,
            n_trials=request.n_trials,
            timeout=request.timeout
        )
        
        search_space = search_service.get_search_space(
            algorithm=request.algorithm,
            custom_space=request.search_space
        )
        
        if not search_space:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No search space defined for algorithm '{request.algorithm}'. Please provide search_space."
            )
        
        # Start optimization in background
        logger.info(f"Scheduling background optimization task for study '{study_name}'...")
        
        async def _run_with_error_handling():
            """Wrapper to ensure exceptions are properly logged."""
            try:
                await _run_optimization_in_background(
                    study_name=study_name,
                    model_id=request.model_id,
                    data_source_id=data_source_id,
                    target_column=target_column,
                    algorithm=request.algorithm,
                    search_space=search_space,
                    metric_name=request.metric_name,
                    direction=request.direction,
                    n_trials=request.n_trials,
                    timeout=request.timeout
                )
            except Exception as e:
                logger.error(f"Unhandled exception in background optimization task: {e}", exc_info=True)
                print(f"CRITICAL: Unhandled exception in background optimization task: {e}", file=sys.stderr, flush=True)
                raise
        
        task = asyncio.create_task(_run_with_error_handling())
        logger.info(f"Background optimization task created: {task}, study: '{study_name}'")
        
        return OptimizationResponse(
            study_name=study_name,
            best_params={},
            best_value=0.0,
            best_trial_number=0,
            n_trials=0,
            status="running"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting optimization: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting optimization: {str(e)}"
        )


@router.get("/study/{study_name}", response_model=StudySummaryResponse)
@router.get("/{study_name}/summary", response_model=StudySummaryResponse)
async def get_study_summary(
    study_name: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Get summary of an optimization study.
    """
    try:
        if not study_name or not study_name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Study name is required"
            )
        
        search_service = HyperparameterSearch(study_name=study_name)
        
        try:
            # Try to load existing study
            study = search_service.optimizer.create_study(study_name=study_name)
            
            # Check if study exists and has trials
            if len(study.trials) == 0:
                # Study exists but has no trials yet (still running)
                return StudySummaryResponse(
                    study_name=study_name,
                    n_trials=0,
                    best_value=0.0,
                    best_params={},
                    best_trial_number=0,
                    direction=search_service.optimizer.direction or "minimize"
                )
            
            summary = search_service.optimizer.get_study_summary()
            
            return StudySummaryResponse(
                study_name=summary["study_name"],
                n_trials=summary["n_trials"],
                best_value=summary["best_value"],
                best_params=summary["best_params"],
                best_trial_number=summary["best_trial_number"],
                direction=summary["direction"]
            )
        except Exception as e:
            # Study might not exist yet
            logger.warning(f"Study '{study_name}' not found or not accessible: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Study '{study_name}' not found. It may still be running or may not exist."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting study summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting study summary: {str(e)}"
        )

