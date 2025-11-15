# API v1 package
from app.api.v1.ingestion import router as ingestion_router
from app.api.v1.data import router as data_router  # data is a package (directory)
from app.api.v1.ml import router as ml_router  # ml is a package (directory)
from app.api.v1.alerts import router as alerts_router
from app.api.v1.ab_test import router as ab_test_router

__all__ = [
    "ingestion_router",
    "data_router",
    "ml_router",
    "alerts_router",
    "ab_test_router",
]

