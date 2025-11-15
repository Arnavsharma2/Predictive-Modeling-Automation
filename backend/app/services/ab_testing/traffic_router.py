"""
Traffic routing for A/B testing.
"""
from typing import Optional, Literal
import hashlib
import random

from app.core.logging import get_logger

logger = get_logger(__name__)


class TrafficRouter:
    """Route traffic between control and treatment variants."""
    
    @staticmethod
    def route(
        test_id: int,
        control_percentage: float,
        treatment_percentage: float,
        routing_strategy: str = "random",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Literal["control", "treatment"]:
        """
        Route request to control or treatment variant.
        
        Args:
            test_id: A/B test ID
            control_percentage: Control traffic percentage (0-100)
            treatment_percentage: Treatment traffic percentage (0-100)
            routing_strategy: Routing strategy (random, user_id, session_id)
            user_id: User ID for consistent routing
            session_id: Session ID for consistent routing
            
        Returns:
            "control" or "treatment"
        """
        # Normalize percentages
        total = control_percentage + treatment_percentage
        if total == 0:
            return "control"  # Default
        
        control_pct = control_percentage / total
        treatment_pct = treatment_percentage / total
        
        if routing_strategy == "random":
            # Random routing
            rand = random.random()
            if rand < control_pct:
                return "control"
            else:
                return "treatment"
        
        elif routing_strategy == "user_id":
            # Consistent routing by user ID
            if not user_id:
                # Fallback to random if no user_id
                return TrafficRouter.route(
                    test_id, control_percentage, treatment_percentage,
                    routing_strategy="random"
                )
            
            # Hash user_id + test_id for consistent assignment
            hash_input = f"{test_id}:{user_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            hash_pct = (hash_value % 10000) / 10000.0
            
            if hash_pct < control_pct:
                return "control"
            else:
                return "treatment"
        
        elif routing_strategy == "session_id":
            # Consistent routing by session ID
            if not session_id:
                # Fallback to random if no session_id
                return TrafficRouter.route(
                    test_id, control_percentage, treatment_percentage,
                    routing_strategy="random"
                )
            
            # Hash session_id + test_id for consistent assignment
            hash_input = f"{test_id}:{session_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            hash_pct = (hash_value % 10000) / 10000.0
            
            if hash_pct < control_pct:
                return "control"
            else:
                return "treatment"
        
        else:
            # Unknown strategy, default to random
            logger.warning(f"Unknown routing strategy: {routing_strategy}, using random")
            return TrafficRouter.route(
                test_id, control_percentage, treatment_percentage,
                routing_strategy="random"
            )

