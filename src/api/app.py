"""
FastAPI Application for Video Recommendations

This provides a REST API for:
- Encoding videos into embeddings
- Finding similar videos
- Managing the video index
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    VideoInput,
    RecommendRequest,
    RecommendResponse,
    Recommendation,
    SimilarityRequest,
    SimilarityResponse,
    HealthResponse,
    IndexResponse,
    VideoEmbedding
)


# ============ App Setup ============

app = FastAPI(
    title="Multimodal Video Recommender",
    description="API for video recommendations using multimodal embeddings",
    version="1.0.0"
)

# Allow cross-origin requests (for web frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Global State ============

class ModelState:
    """Holds the model and video index."""

    def __init__(self):
        self.model = None
        self.device = None
        self.video_index = {}  # video_id -> embedding
        self.video_metadata = {}  # video_id -> {title, description, ...}
        self.embeddings_matrix = None  # For fast similarity search
        self.video_ids = []  # Ordered list matching embeddings_matrix rows

    def is_loaded(self) -> bool:
        return self.model is not None

    def index_size(self) -> int:
        return len(self.video_index)

    def rebuild_matrix(self):
        """Rebuild the embeddings matrix for fast search."""
        if self.video_index:
            self.video_ids = list(self.video_index.keys())
            embeddings = [self.video_index[vid] for vid in self.video_ids]
            self.embeddings_matrix = torch.stack(embeddings)
        else:
            self.embeddings_matrix = None
            self.video_ids = []


state = ModelState()


# ============ Startup/Shutdown ============

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    print("Loading model...")

    # Set device (supports CUDA, MPS, and CPU)
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.device import get_device
    state.device = get_device()

    try:
        # Import model
        from models.two_tower import TwoTowerModel

        # Initialize model
        # Note: pretrained=False avoids SSL download issues on macOS
        # For production, fix SSL certs and use pretrained=True
        state.model = TwoTowerModel(
            feature_dim=512,
            share_towers=True,
            freeze_encoders=True,
            pretrained=False  # Set to True once SSL is fixed
        )
        state.model.to(state.device)
        state.model.eval()

        # Load checkpoint if exists
        checkpoint_path = Path("checkpoints/best.pt")
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=state.device)
            state.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("No checkpoint found, using untrained model")

        print("Model loaded successfully!")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will run but recommendations won't work")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    print("Shutting down...")


# ============ Endpoints ============

@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "name": "Multimodal Video Recommender API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=state.is_loaded(),
        index_size=state.index_size()
    )


@app.post("/index", response_model=IndexResponse, tags=["Index"])
async def index_video(video: VideoInput):
    """
    Add a video to the search index.

    This encodes the video and stores its embedding for future searches.
    """
    if not state.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # For now, we'll create a dummy embedding from the text
        # In production, you'd load the actual video file
        with torch.no_grad():
            # Encode text (simplified - real version would use video/audio too)
            text = f"{video.title} {video.description}"

            # Create dummy frames and audio for demonstration
            dummy_frames = torch.zeros(1, 8, 3, 224, 224).to(state.device)
            dummy_audio = torch.zeros(1, 16000).to(state.device)

            embedding = state.model.encode_video(
                dummy_frames,
                dummy_audio,
                [text]
            )

            # Store in index
            state.video_index[video.video_id] = embedding.squeeze(0).cpu()
            state.video_metadata[video.video_id] = {
                "title": video.title,
                "description": video.description,
                "video_path": video.video_path
            }

            # Rebuild search matrix
            state.rebuild_matrix()

        return IndexResponse(
            video_id=video.video_id,
            indexed=True,
            message=f"Video indexed successfully. Index size: {state.index_size()}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode", response_model=VideoEmbedding, tags=["Encoding"])
async def encode_video(video: VideoInput):
    """
    Get the embedding for a video without indexing it.

    Useful for debugging or one-off comparisons.
    """
    if not state.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        with torch.no_grad():
            text = f"{video.title} {video.description}"
            dummy_frames = torch.zeros(1, 8, 3, 224, 224).to(state.device)
            dummy_audio = torch.zeros(1, 16000).to(state.device)

            embedding = state.model.encode_video(
                dummy_frames,
                dummy_audio,
                [text]
            )

            embedding_list = embedding.squeeze(0).cpu().tolist()

        return VideoEmbedding(
            video_id=video.video_id,
            embedding=embedding_list,
            dimension=len(embedding_list)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendRequest):
    """
    Get video recommendations similar to the query video.

    The query video must be in the index.
    """
    if not state.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.video_id not in state.video_index:
        raise HTTPException(
            status_code=404,
            detail=f"Video '{request.video_id}' not found in index"
        )

    if state.index_size() < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 videos in index for recommendations"
        )

    try:
        # Get query embedding
        query_emb = state.video_index[request.video_id]

        # Compute similarities with all videos
        similarities = query_emb @ state.embeddings_matrix.T

        # Get top-k (excluding the query itself)
        top_k = min(request.top_k + 1, len(state.video_ids))
        scores, indices = torch.topk(similarities, k=top_k)

        # Build recommendations (skip if it's the query video)
        recommendations = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            vid = state.video_ids[idx]
            if vid != request.video_id:
                metadata = state.video_metadata.get(vid, {})
                recommendations.append(Recommendation(
                    video_id=vid,
                    score=score,
                    title=metadata.get("title")
                ))
                if len(recommendations) >= request.top_k:
                    break

        return RecommendResponse(
            query_video_id=request.video_id,
            recommendations=recommendations,
            total=len(recommendations)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity", response_model=SimilarityResponse, tags=["Similarity"])
async def compute_similarity(request: SimilarityRequest):
    """
    Compute similarity between two indexed videos.
    """
    if request.video_id_1 not in state.video_index:
        raise HTTPException(
            status_code=404,
            detail=f"Video '{request.video_id_1}' not found in index"
        )

    if request.video_id_2 not in state.video_index:
        raise HTTPException(
            status_code=404,
            detail=f"Video '{request.video_id_2}' not found in index"
        )

    emb1 = state.video_index[request.video_id_1]
    emb2 = state.video_index[request.video_id_2]

    similarity = (emb1 @ emb2).item()

    return SimilarityResponse(
        video_id_1=request.video_id_1,
        video_id_2=request.video_id_2,
        similarity=similarity
    )


@app.get("/index/list", tags=["Index"])
async def list_indexed_videos():
    """List all videos in the index."""
    return {
        "videos": list(state.video_metadata.keys()),
        "total": state.index_size()
    }


@app.delete("/index/{video_id}", tags=["Index"])
async def remove_from_index(video_id: str):
    """Remove a video from the index."""
    if video_id not in state.video_index:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")

    del state.video_index[video_id]
    del state.video_metadata[video_id]
    state.rebuild_matrix()

    return {"message": f"Video '{video_id}' removed from index"}


# ============ Run Server ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
