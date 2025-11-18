"""
A/B test management service.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from app.core.logging import get_logger
from app.models.database.ab_tests import ABTest, ABTestPrediction, ABTestStatus
from app.models.database.model_versions import ModelVersion
from app.services.ab_testing.traffic_router import TrafficRouter
from app.services.ab_testing.statistics import ABTestStatistics

logger = get_logger(__name__)


class ABTestManager:
    """Manage A/B tests."""
    
    @staticmethod
    async def create_test(
        db: AsyncSession,
        name: str,
        model_id: int,
        control_version_id: int,
        treatment_version_id: int,
        control_traffic_percentage: float = 50.0,
        treatment_traffic_percentage: float = 50.0,
        routing_strategy: str = "random",
        min_samples: int = 1000,
        description: Optional[str] = None
    ) -> ABTest:
        """
        Create A/B test.
        
        Args:
            db: Database session
            name: Test name
            model_id: Model ID
            control_version_id: Control version ID
            treatment_version_id: Treatment version ID
            control_traffic_percentage: Control traffic percentage
            treatment_traffic_percentage: Treatment traffic percentage
            routing_strategy: Routing strategy
            min_samples: Minimum samples for significance
            description: Description
            
        Returns:
            Created A/B test
        """
        # Validate versions exist
        control_version = await db.get(ModelVersion, control_version_id)
        treatment_version = await db.get(ModelVersion, treatment_version_id)
        
        if not control_version or not treatment_version:
            raise ValueError("Control or treatment version not found")
        
        # Validate traffic percentages
        total = control_traffic_percentage + treatment_traffic_percentage
        if abs(total - 100.0) > 0.01:
            raise ValueError("Traffic percentages must sum to 100")
        
        test = ABTest(
            name=name,
            description=description,
            model_id=model_id,
            control_version_id=control_version_id,
            treatment_version_id=treatment_version_id,
            control_traffic_percentage=control_traffic_percentage,
            treatment_traffic_percentage=treatment_traffic_percentage,
            routing_strategy=routing_strategy,
            min_samples=min_samples
        )
        
        db.add(test)
        await db.commit()
        await db.refresh(test)
        
        logger.info(f"Created A/B test: {name} (ID: {test.id})")
        return test
    
    @staticmethod
    async def start_test(
        db: AsyncSession,
        test_id: int
    ) -> ABTest:
        """
        Start A/B test.
        
        Args:
            db: Database session
            test_id: Test ID
            
        Returns:
            Updated test
        """
        test = await db.get(ABTest, test_id)
        if not test:
            raise ValueError(f"A/B test {test_id} not found")
        
        test.status = ABTestStatus.RUNNING
        test.start_date = datetime.now(timezone.utc)
        
        await db.commit()
        await db.refresh(test)
        
        logger.info(f"Started A/B test: {test_id}")
        return test
    
    @staticmethod
    async def stop_test(
        db: AsyncSession,
        test_id: int
    ) -> ABTest:
        """
        Stop A/B test.
        
        Args:
            db: Database session
            test_id: Test ID
            
        Returns:
            Updated test
        """
        test = await db.get(ABTest, test_id)
        if not test:
            raise ValueError(f"A/B test {test_id} not found")
        
        test.status = ABTestStatus.COMPLETED
        test.end_date = datetime.now(timezone.utc)
        
        # Calculate results
        await ABTestManager._calculate_results(db, test)
        
        await db.commit()
        await db.refresh(test)
        
        logger.info(f"Stopped A/B test: {test_id}")
        return test
    
    @staticmethod
    async def route_prediction(
        db: AsyncSession,
        test_id: int,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Route prediction request to appropriate variant.
        
        Args:
            db: Database session
            test_id: Test ID
            user_id: User ID
            session_id: Session ID
            
        Returns:
            Dictionary with variant info or None if test not active
        """
        test = await db.get(ABTest, test_id)
        if not test or test.status != ABTestStatus.RUNNING:
            return None
        
        # Route to variant
        variant = TrafficRouter.route(
            test_id=test_id,
            control_percentage=test.control_traffic_percentage,
            treatment_percentage=test.treatment_traffic_percentage,
            routing_strategy=test.routing_strategy,
            user_id=user_id,
            session_id=session_id
        )
        
        # Get version ID
        version_id = (
            test.control_version_id if variant == "control"
            else test.treatment_version_id
        )
        
        return {
            "variant": variant,
            "version_id": version_id,
            "test_id": test_id
        }
    
    @staticmethod
    async def record_prediction(
        db: AsyncSession,
        test_id: int,
        version_id: int,
        variant: str,
        input_data: Dict[str, Any],
        prediction: Optional[Dict[str, Any]] = None,
        actual_value: Optional[Any] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None
    ) -> ABTestPrediction:
        """
        Record prediction made during A/B test.
        
        Args:
            db: Database session
            test_id: Test ID
            version_id: Version ID used
            variant: Variant (control or treatment)
            input_data: Input data
            prediction: Prediction result
            actual_value: Ground truth (if available)
            user_id: User ID
            session_id: Session ID
            latency_ms: Latency in milliseconds
            error: Error message (if any)
            
        Returns:
            Created prediction record
        """
        pred_record = ABTestPrediction(
            test_id=test_id,
            version_id=version_id,
            variant=variant,
            input_data=input_data,
            prediction=prediction,
            actual_value=actual_value,
            user_id=user_id,
            session_id=session_id,
            latency_ms=latency_ms,
            error=error
        )
        
        db.add(pred_record)
        await db.commit()
        await db.refresh(pred_record)
        
        return pred_record
    
    @staticmethod
    async def get_results(
        db: AsyncSession,
        test_id: int,
        metric_name: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Get A/B test results.
        
        Args:
            db: Database session
            test_id: Test ID
            metric_name: Metric to compare
            
        Returns:
            Test results
        """
        test = await db.get(ABTest, test_id)
        if not test:
            raise ValueError(f"A/B test {test_id} not found")
        
        # Get predictions for control and treatment
        control_query = select(ABTestPrediction).where(
            and_(
                ABTestPrediction.test_id == test_id,
                ABTestPrediction.variant == "control"
            )
        )
        control_result = await db.execute(control_query)
        control_predictions = control_result.scalars().all()
        
        treatment_query = select(ABTestPrediction).where(
            and_(
                ABTestPrediction.test_id == test_id,
                ABTestPrediction.variant == "treatment"
            )
        )
        treatment_result = await db.execute(treatment_query)
        treatment_predictions = treatment_result.scalars().all()
        
        # Convert to dict format
        control_data = [
            {
                "prediction": p.prediction,
                "actual_value": p.actual_value
            }
            for p in control_predictions
        ]
        
        treatment_data = [
            {
                "prediction": p.prediction,
                "actual_value": p.actual_value
            }
            for p in treatment_predictions
        ]
        
        # Calculate metrics
        # Determine metric type from model
        from app.models.database.ml_models import MLModel
        model = await db.get(MLModel, test.model_id)
        metric_type = "regression" if model and "regression" in model.type.value.lower() else "classification"
        
        control_metrics = ABTestStatistics.calculate_metrics(control_data, metric_type)
        treatment_metrics = ABTestStatistics.calculate_metrics(treatment_data, metric_type)
        
        # Perform statistical test
        t_test_result = ABTestStatistics.t_test(
            control_metrics,
            treatment_metrics,
            metric_name
        )
        
        # Determine winner
        winner = ABTestStatistics.determine_winner(
            control_metrics,
            treatment_metrics,
            metric_name,
            t_test_result.get("p_value")
        )
        
        return {
            "test_id": test_id,
            "test_name": test.name,
            "status": test.status.value,
            "control_metrics": control_metrics,
            "treatment_metrics": treatment_metrics,
            "statistical_test": t_test_result,
            "winner": winner,
            "control_samples": len(control_predictions),
            "treatment_samples": len(treatment_predictions)
        }
    
    @staticmethod
    async def _calculate_results(
        db: AsyncSession,
        test: ABTest
    ):
        """Calculate and update test results."""
        results = await ABTestManager.get_results(db, test.id)
        
        test.control_metrics = results["control_metrics"]
        test.treatment_metrics = results["treatment_metrics"]
        test.statistical_significance = results["statistical_test"].get("p_value")
        test.winner = results["winner"]
        
        await db.commit()

