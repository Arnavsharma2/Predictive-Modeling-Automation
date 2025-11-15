"""
Experiment tracking API endpoints.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.schemas.experiments import (
    ExperimentCreate,
    ExperimentResponse,
    RunCreate,
    RunResponse,
    RunListResponse,
    LogMetricsRequest,
    LogParamsRequest,
    RunSearchRequest,
    RunComparisonResponse,
)
from app.ml.tracking.experiment_tracker import ExperimentTracker

logger = get_logger(__name__)

router = APIRouter(prefix="/experiments", tags=["Experiments"])

# Global experiment tracker instance
experiment_tracker = ExperimentTracker()


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    experiment: ExperimentCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new MLflow experiment.
    """
    try:
        experiment_id = experiment_tracker.client.create_experiment(
            experiment_name=experiment.name,
            tags=experiment.tags or {}
        )
        
        mlflow_experiment = experiment_tracker.client.get_experiment(experiment.name)
        
        return ExperimentResponse(
            experiment_id=experiment_id,
            name=experiment.name,
            tags=experiment.tags or {},
            artifact_location=mlflow_experiment.artifact_location if mlflow_experiment else None
        )
    except ConnectionError as e:
        logger.warning(f"MLflow service unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLflow service is not available. Please ensure the MLflow service is running."
        )
    except Exception as e:
        logger.error(f"Error creating experiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating experiment: {str(e)}"
        )


@router.get("", response_model=List[ExperimentResponse])
async def list_experiments(
    db: AsyncSession = Depends(get_db)
):
    """
    List all experiments.
    """
    try:
        import mlflow
        
        # Get all experiments
        experiments = mlflow.search_experiments()
        
        results = []
        for exp in experiments:
            results.append(ExperimentResponse(
                experiment_id=exp.experiment_id,
                name=exp.name,
                tags=exp.tags,
                artifact_location=exp.artifact_location
            ))
        
        return results
    except ConnectionError as e:
        logger.warning(f"MLflow service unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLflow service is not available. Please ensure the MLflow service is running."
        )
    except Exception as e:
        logger.error(f"Error listing experiments: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing experiments: {str(e)}"
        )


@router.get("/{experiment_name}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_name: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get experiment by name.
    """
    try:
        experiment = experiment_tracker.client.get_experiment(experiment_name)
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment '{experiment_name}' not found"
            )
        
        return ExperimentResponse(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            tags=experiment.tags,
            artifact_location=experiment.artifact_location
        )
    except HTTPException:
        raise
    except ConnectionError as e:
        logger.warning(f"MLflow service unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLflow service is not available. Please ensure the MLflow service is running."
        )
    except Exception as e:
        logger.error(f"Error getting experiment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting experiment: {str(e)}"
        )


@router.post("/runs", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
async def create_run(
    run: RunCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new run in an experiment.
    """
    try:
        mlflow_run = experiment_tracker.client.start_run(
            experiment_name=run.experiment_name,
            run_name=run.run_name,
            tags=run.tags or {}
        )
        
        run_info = mlflow_run.info
        
        return RunResponse(
            run_id=run_info.run_id,
            run_name=run_info.run_name,
            experiment_id=run_info.experiment_id,
            status=run_info.status,
            start_time=run_info.start_time,
            end_time=run_info.end_time,
            metrics={},
            params={},
            tags=run.tags or {}
        )
    except ConnectionError as e:
        logger.warning(f"MLflow service unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLflow service is not available. Please ensure the MLflow service is running."
        )
    except Exception as e:
        logger.error(f"Error creating run: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating run: {str(e)}"
        )


@router.post("/runs/search", response_model=RunListResponse)
async def search_runs(
    request: RunSearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Search for runs in an experiment.
    """
    try:
        experiment = experiment_tracker.client.get_experiment(request.experiment_name)
        
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment '{request.experiment_name}' not found"
            )
        
        runs = experiment_tracker.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=request.filter_string,
            max_results=request.max_results
        )
        
        run_responses = []
        for run in runs:
            run_responses.append(RunResponse(
                run_id=run.info.run_id,
                run_name=run.info.run_name,
                experiment_id=run.info.experiment_id,
                status=run.info.status,
                start_time=run.info.start_time,
                end_time=run.info.end_time,
                metrics=run.data.metrics,
                params=run.data.params,
                tags=run.data.tags
            ))
        
        return RunListResponse(
            runs=run_responses,
            total=len(run_responses)
        )
    except HTTPException:
        raise
    except ConnectionError as e:
        logger.warning(f"MLflow service unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLflow service is not available. Please ensure the MLflow service is running."
        )
    except Exception as e:
        logger.error(f"Error searching runs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching runs: {str(e)}"
        )


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get run by ID.
    """
    try:
        mlflow_run = experiment_tracker.client.get_run(run_id)
        
        if not mlflow_run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found"
            )
        
        return RunResponse(
            run_id=mlflow_run.info.run_id,
            run_name=mlflow_run.info.run_name,
            experiment_id=mlflow_run.info.experiment_id,
            status=mlflow_run.info.status,
            start_time=mlflow_run.info.start_time,
            end_time=mlflow_run.info.end_time,
            metrics=mlflow_run.data.metrics,
            params=mlflow_run.data.params,
            tags=mlflow_run.data.tags
        )
    except HTTPException:
        raise
    except ConnectionError as e:
        logger.warning(f"MLflow service unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLflow service is not available. Please ensure the MLflow service is running."
        )
    except Exception as e:
        logger.error(f"Error getting run: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting run: {str(e)}"
        )


@router.post("/runs/{run_id}/metrics", status_code=status.HTTP_200_OK)
async def log_metrics(
    run_id: str,
    request: LogMetricsRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Log metrics to a run.
    """
    try:
        from mlflow.tracking import MlflowClient
        
        # Verify run exists
        run = experiment_tracker.client.get_run(run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found"
            )
        
        # Log metrics using MlflowClient
        client = MlflowClient(tracking_uri=experiment_tracker.client.tracking_uri)
        for metric in request.metrics:
            client.log_metric(
                run_id=run_id,
                key=metric.name,
                value=metric.value,
                step=metric.step
            )
        
        return {"message": f"Logged {len(request.metrics)} metrics to run {run_id}"}
    except HTTPException:
        raise
    except ConnectionError as e:
        logger.warning(f"MLflow service unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLflow service is not available. Please ensure the MLflow service is running."
        )
    except Exception as e:
        logger.error(f"Error logging metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error logging metrics: {str(e)}"
        )


@router.post("/runs/{run_id}/params", status_code=status.HTTP_200_OK)
async def log_params(
    run_id: str,
    request: LogParamsRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Log parameters to a run.
    """
    try:
        from mlflow.tracking import MlflowClient
        
        # Verify run exists
        run = experiment_tracker.client.get_run(run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found"
            )
        
        # Log parameters using MlflowClient
        client = MlflowClient(tracking_uri=experiment_tracker.client.tracking_uri)
        for param in request.params:
            client.log_param(
                run_id=run_id,
                key=param.name,
                value=param.value
            )
        
        return {"message": f"Logged {len(request.params)} parameters to run {run_id}"}
    except HTTPException:
        raise
    except ConnectionError as e:
        logger.warning(f"MLflow service unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLflow service is not available. Please ensure the MLflow service is running."
        )
    except Exception as e:
        logger.error(f"Error logging parameters: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error logging parameters: {str(e)}"
        )


@router.post("/runs/{run_id}/end", status_code=status.HTTP_200_OK)
async def end_run(
    run_id: str,
    status: str = Query("FINISHED", pattern="^(FINISHED|FAILED|KILLED)$"),
    db: AsyncSession = Depends(get_db)
):
    """
    End a run.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        # Verify run exists
        run = experiment_tracker.client.get_run(run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run '{run_id}' not found"
            )
        
        # End the run using MlflowClient
        client = MlflowClient(tracking_uri=experiment_tracker.client.tracking_uri)
        client.set_terminated(run_id=run_id, status=status)
        return {"message": f"Ended run {run_id} with status {status}"}
    except HTTPException:
        raise
    except ConnectionError as e:
        logger.warning(f"MLflow service unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLflow service is not available. Please ensure the MLflow service is running."
        )
    except Exception as e:
        logger.error(f"Error ending run: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ending run: {str(e)}"
        )


@router.get("/{experiment_name}/compare", response_model=RunComparisonResponse)
async def compare_runs(
    experiment_name: str,
    metric_name: str = Query(..., description="Metric to compare"),
    top_n: int = Query(10, ge=1, le=100, description="Number of top runs"),
    ascending: bool = Query(True, description="Sort ascending (True for RMSE, False for accuracy/R²)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Compare runs in an experiment.
    """
    try:
        runs = experiment_tracker.compare_runs(
            experiment_name=experiment_name,
            metric_name=metric_name,
            top_n=top_n,
            ascending=ascending
        )
        
        run_responses = []
        best_run_id = None
        best_metric_value = None
        
        for run_data in runs:
            run_responses.append(RunResponse(
                run_id=run_data["run_id"],
                run_name=run_data["run_name"],
                experiment_id="",  # Not in comparison data
                status=run_data["status"],
                start_time=None,  # Not in comparison data
                end_time=None,
                metrics=run_data["metrics"],
                params=run_data["params"],
                tags=run_data["tags"]
            ))
            
            if best_run_id is None:
                best_run_id = run_data["run_id"]
                best_metric_value = run_data["metric_value"]
        
        return RunComparisonResponse(
            runs=run_responses,
            metric_name=metric_name,
            best_run_id=best_run_id,
            best_metric_value=best_metric_value
        )
    except ConnectionError as e:
        logger.warning(f"MLflow service unavailable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLflow service is not available. Please ensure the MLflow service is running."
        )
    except Exception as e:
        logger.error(f"Error comparing runs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing runs: {str(e)}"
        )

