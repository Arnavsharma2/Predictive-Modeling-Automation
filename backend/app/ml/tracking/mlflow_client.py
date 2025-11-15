"""
MLflow client wrapper for experiment tracking.
"""
import mlflow
import mlflow.sklearn
from typing import Optional, Dict, Any, List
from pathlib import Path
import os

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Import MLflowException for better error handling
try:
    from mlflow.exceptions import MlflowException
except ImportError:
    # Fallback if MLflowException is not available
    MlflowException = Exception


class MLflowClient:
    """
    Wrapper for MLflow client operations.
    
    Provides a simplified interface for MLflow experiment tracking,
    run management, and artifact logging.
    """
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow client.
        
        Args:
            tracking_uri: MLflow tracking URI (defaults to settings)
        """
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")
        except Exception as e:
            logger.warning(f"Failed to set MLflow tracking URI: {e}. MLflow features will be disabled.")
            self.tracking_uri = None
    
    def check_server_health(self) -> bool:
        """
        Check if MLflow server is accessible and healthy.
        
        Returns:
            True if server is accessible, False otherwise
        """
        if not self.tracking_uri:
            return False
        
        try:
            # Try to access MLflow's health endpoint or list experiments
            experiments = mlflow.search_experiments(max_results=1)
            return True
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["connection", "refused", "timeout", "unreachable", "name or service not known"]):
                logger.debug(f"MLflow server health check failed: {e}")
                return False
            # Other errors might still mean server is up (e.g., no experiments yet)
            return True
    
    def create_experiment(
        self,
        experiment_name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a new MLflow experiment.
        
        Args:
            experiment_name: Name of the experiment
            tags: Optional tags for the experiment
            
        Returns:
            Experiment ID
        """
        if not self.tracking_uri:
            raise ConnectionError("MLflow tracking URI not configured")
        
        try:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                tags=tags or {}
            )
            logger.info(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
            return experiment_id
        except Exception as e:
            # Experiment might already exist
            if "already exists" in str(e).lower():
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    logger.info(f"Experiment '{experiment_name}' already exists with ID: {experiment.experiment_id}")
                    return experiment.experiment_id
            # Connection errors - log but don't fail
            if "connection" in str(e).lower() or "name or service not known" in str(e).lower():
                logger.warning(f"MLflow connection error (service may not be available): {e}")
                raise ConnectionError(f"MLflow service unavailable: {e}")
            raise
    
    def get_experiment(self, experiment_name: str) -> Optional[mlflow.entities.Experiment]:
        """
        Get experiment by name.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Experiment object or None if not found
        """
        if not self.tracking_uri:
            return None
        
        try:
            return mlflow.get_experiment_by_name(experiment_name)
        except Exception as e:
            if "connection" in str(e).lower() or "name or service not known" in str(e).lower():
                logger.warning(f"MLflow connection error: {e}")
            else:
                logger.error(f"Error getting experiment '{experiment_name}': {e}")
            return None
    
    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Active run object
        """
        if not self.tracking_uri:
            raise ConnectionError("MLflow tracking URI not configured")
        
        try:
            # Get or create experiment
            experiment = self.get_experiment(experiment_name)
            if not experiment:
                self.create_experiment(experiment_name)
            
            mlflow.set_experiment(experiment_name)
            
            run = mlflow.start_run(run_name=run_name, tags=tags or {})
            logger.info(f"Started MLflow run: {run.info.run_id}")
            return run
        except ConnectionError:
            raise
        except Exception as e:
            if "connection" in str(e).lower() or "name or service not known" in str(e).lower():
                logger.warning(f"MLflow connection error: {e}")
                raise ConnectionError(f"MLflow service unavailable: {e}")
            raise
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for metrics
        """
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ) -> None:
        """
        Log a model as an artifact.
        
        Supports multiple model types: sklearn, XGBoost, LightGBM, CatBoost.
        
        Args:
            model: Model object to log
            artifact_path: Path within artifacts directory
            registered_model_name: Optional name for model registry
        """
        if not self.tracking_uri:
            logger.warning("MLflow tracking URI not configured, skipping model logging")
            return
        
        # Verify MLflow server is accessible before attempting to log
        if not self.check_server_health():
            logger.warning(
                "MLflow server not accessible. "
                "Model logging skipped, but training will continue."
            )
            return
        
        model_type = type(model).__name__
        model_module = type(model).__module__
        
        # Determine model flavor based on type
        # Import mlflow modules at function level to avoid scoping issues
        try:
            # Suppress deprecation warnings by using artifact_path (still valid for log_model)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*artifact_path.*")
                
                if "xgboost" in model_module.lower() or "XGB" in model_type:
                    import mlflow.xgboost as mlflow_xgboost
                    mlflow_xgboost.log_model(
                        model,
                        artifact_path=artifact_path,
                        registered_model_name=registered_model_name
                    )
                    logger.info(f"Logged XGBoost model to artifact path: {artifact_path}")
                elif "lightgbm" in model_module.lower() or "LGBM" in model_type:
                    import mlflow.lightgbm as mlflow_lightgbm
                    mlflow_lightgbm.log_model(
                        model,
                        artifact_path=artifact_path,
                        registered_model_name=registered_model_name
                    )
                    logger.info(f"Logged LightGBM model to artifact path: {artifact_path}")
                elif "catboost" in model_module.lower() or "CatBoost" in model_type:
                    import mlflow.catboost as mlflow_catboost
                    mlflow_catboost.log_model(
                        model,
                        artifact_path=artifact_path,
                        registered_model_name=registered_model_name
                    )
                    logger.info(f"Logged CatBoost model to artifact path: {artifact_path}")
                else:
                    # Default to sklearn (works for RandomForest, etc.)
                    # Use the already imported mlflow.sklearn
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path=artifact_path,
                        registered_model_name=registered_model_name
                    )
                    logger.info(f"Logged sklearn model to artifact path: {artifact_path}")
        except MlflowException as e:
            # Handle MLflow-specific errors gracefully
            error_msg = str(e)
            error_lower = error_msg.lower()
            
            # Check for HTTP errors (404, 500, etc.)
            if "404" in error_msg or "logged-models" in error_lower or "not found" in error_lower:
                logger.warning(
                    f"MLflow logged-models endpoint not available (may be version mismatch or server issue): {error_msg}. "
                    "Model logging skipped, but training will continue."
                )
                return
            elif "connection" in error_lower or "name or service not known" in error_lower or "refused" in error_lower:
                logger.warning(
                    f"MLflow connection error: {error_msg}. "
                    "Model logging skipped, but training will continue."
                )
                return
            elif "500" in error_msg or "internal server error" in error_lower:
                logger.warning(
                    f"MLflow server error: {error_msg}. "
                    "Model logging skipped, but training will continue."
                )
                return
            else:
                # For other MLflow errors, log and don't raise (non-critical)
                logger.warning(f"MLflow model logging failed (non-critical): {e}")
                return
        except Exception as e:
            # Handle other unexpected errors - don't fail training
            error_msg = str(e)
            error_lower = error_msg.lower()
            
            # Check for common connection/HTTP errors
            if any(keyword in error_lower for keyword in ["404", "connection", "refused", "timeout", "unreachable"]):
                logger.warning(
                    f"MLflow model logging failed (non-critical): {error_msg}. "
                    "Training will continue."
                )
            else:
                logger.warning(f"Unexpected error logging model to MLflow (non-critical): {e}")
            return
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log a file or directory as an artifact.
        
        Args:
            local_path: Path to local file or directory
            artifact_path: Optional path within artifacts directory
        """
        mlflow.log_artifacts(local_path, artifact_path=artifact_path)
        logger.debug(f"Logged artifact from: {local_path}")
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_path: str) -> None:
        """
        Log a dictionary as a JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            artifact_path: Path for the artifact
        """
        import json
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dictionary, f, indent=2)
            temp_path = f.name
        
        try:
            mlflow.log_artifact(temp_path, artifact_path=artifact_path)
        finally:
            os.unlink(temp_path)
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags for the current run.
        
        Args:
            tags: Dictionary of tags
        """
        mlflow.set_tags(tags)
        logger.debug(f"Set {len(tags)} tags")
    
    def end_run(self, status: str = "FINISHED", run_id: Optional[str] = None) -> None:
        """
        End a run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
            run_id: Optional run ID to end. If None, ends the current active run.
        """
        if run_id:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=self.tracking_uri)
            client.set_terminated(run_id=run_id, status=status)
            logger.info(f"Ended MLflow run {run_id} with status: {status}")
        else:
            mlflow.end_run(status=status)
            logger.info(f"Ended current MLflow run with status: {status}")
    
    def search_runs(
        self,
        experiment_ids: Optional[List[str]] = None,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[mlflow.entities.Run]:
        """
        Search for runs.
        
        Args:
            experiment_ids: List of experiment IDs to search
            filter_string: Optional filter string (e.g., "metrics.rmse < 0.5")
            max_results: Maximum number of results
            
        Returns:
            List of run objects
        """
        return mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            output_format="list"
        )
    
    def get_run(self, run_id: str) -> mlflow.entities.Run:
        """
        Get a run by ID.
        
        Args:
            run_id: Run ID
            
        Returns:
            Run object
        """
        return mlflow.get_run(run_id)
    
    def get_best_run(
        self,
        experiment_name: str,
        metric_name: str,
        ascending: bool = True
    ) -> Optional[mlflow.entities.Run]:
        """
        Get the best run from an experiment based on a metric.
        
        Args:
            experiment_name: Name of the experiment
            metric_name: Name of the metric to optimize
            ascending: True to minimize, False to maximize
            
        Returns:
            Best run object or None
        """
        experiment = self.get_experiment(experiment_name)
        if not experiment:
            return None
        
        runs = self.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=1000
        )
        
        if not runs:
            return None
        
        # Filter runs that have the metric
        valid_runs = [
            run for run in runs
            if metric_name in run.data.metrics
        ]
        
        if not valid_runs:
            return None
        
        # Sort by metric value
        valid_runs.sort(
            key=lambda r: r.data.metrics[metric_name],
            reverse=not ascending
        )
        
        return valid_runs[0]

