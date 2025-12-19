"""
FastAPI application for video recommendations.
"""

from .app import app
from .schemas import (
    VideoInput,
    RecommendRequest,
    RecommendResponse,
    SimilarityRequest,
    SimilarityResponse,
    HealthResponse
)

__all__ = [
    "app",
    "VideoInput",
    "RecommendRequest",
    "RecommendResponse",
    "SimilarityRequest",
    "SimilarityResponse",
    "HealthResponse"
]
