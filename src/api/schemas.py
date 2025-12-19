"""
Pydantic Schemas for API request/response validation.

Pydantic automatically validates incoming JSON and provides clear error messages.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ============ Request Schemas ============

class VideoInput(BaseModel):
    """Input for encoding a single video."""
    video_id: str = Field(..., description="Unique identifier for the video")
    video_path: Optional[str] = Field(None, description="Path to video file")
    title: str = Field(..., description="Video title")
    description: Optional[str] = Field("", description="Video description")

    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "abc123",
                "video_path": "data/raw/videos/abc123.mp4",
                "title": "How to make pasta",
                "description": "Easy 10-minute recipe"
            }
        }


class RecommendRequest(BaseModel):
    """Request for video recommendations."""
    video_id: str = Field(..., description="ID of video to find recommendations for")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations")

    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "abc123",
                "top_k": 10
            }
        }


class SimilarityRequest(BaseModel):
    """Request to compute similarity between two videos."""
    video_id_1: str
    video_id_2: str


# ============ Response Schemas ============

class VideoEmbedding(BaseModel):
    """Response containing video embedding."""
    video_id: str
    embedding: list[float]
    dimension: int


class Recommendation(BaseModel):
    """A single recommendation."""
    video_id: str
    score: float = Field(..., description="Similarity score (0-1)")
    title: Optional[str] = None


class RecommendResponse(BaseModel):
    """Response with list of recommendations."""
    query_video_id: str
    recommendations: list[Recommendation]
    total: int


class SimilarityResponse(BaseModel):
    """Response with similarity score."""
    video_id_1: str
    video_id_2: str
    similarity: float = Field(..., description="Cosine similarity (-1 to 1)")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    index_size: int = Field(..., description="Number of videos in index")


class IndexResponse(BaseModel):
    """Response after indexing a video."""
    video_id: str
    indexed: bool
    message: str
