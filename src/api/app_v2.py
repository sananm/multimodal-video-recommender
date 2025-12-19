"""
FastAPI Application v2 - With Real Video Processing

This version processes actual video files from MicroLens dataset.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.device import get_device
from data.processors import VideoProcessor, AudioProcessor


# ============ Schemas ============

class VideoInput(BaseModel):
    video_id: str
    video_path: Optional[str] = None
    title: str = ""

class RecommendRequest(BaseModel):
    video_id: str
    top_k: int = Field(default=5, ge=1, le=50)

class IndexMicroLensRequest(BaseModel):
    max_videos: int = Field(default=10, ge=1, le=100)


# ============ App Setup ============

app = FastAPI(
    title="Multimodal Video Recommender v2",
    description="Real video processing with MicroLens dataset",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Global State ============

class ModelState:
    def __init__(self):
        self.model = None
        self.device = None
        self.video_index = {}
        self.video_metadata = {}
        self.embeddings_matrix = None
        self.video_ids = []
        self.video_processor = None
        self.audio_processor = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def index_size(self) -> int:
        return len(self.video_index)

    def rebuild_matrix(self):
        if self.video_index:
            self.video_ids = list(self.video_index.keys())
            embeddings = [self.video_index[vid] for vid in self.video_ids]
            self.embeddings_matrix = torch.stack(embeddings)
        else:
            self.embeddings_matrix = None
            self.video_ids = []


state = ModelState()


# ============ Startup ============

@app.on_event("startup")
async def load_model():
    print("Loading model...")

    state.device = get_device()

    # Initialize processors
    state.video_processor = VideoProcessor(num_frames=8, frame_size=(224, 224))
    state.audio_processor = AudioProcessor(sample_rate=16000, duration=5.0)

    try:
        from models.two_tower import TwoTowerModel

        state.model = TwoTowerModel(
            feature_dim=512,
            share_towers=True,
            freeze_encoders=True,
            pretrained=False
        )
        state.model.to(state.device)
        state.model.eval()

        print("Model loaded successfully on", state.device)

    except Exception as e:
        print(f"Error loading model: {e}")


# ============ Endpoints ============

@app.get("/")
async def root():
    return {"name": "Multimodal Video Recommender v2", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state.is_loaded(),
        "index_size": state.index_size(),
        "device": str(state.device)
    }


@app.post("/index/microlens")
async def index_microlens(request: IndexMicroLensRequest):
    """
    Index videos from MicroLens dataset.
    Processes real video files and creates embeddings.
    """
    if not state.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    import pandas as pd

    data_dir = Path("data/raw/microlens")
    video_dir = data_dir / "videos"

    # Load titles
    titles_df = pd.read_csv(data_dir / "MicroLens-50k_titles.csv")

    # Get available videos
    available = {int(p.stem) for p in video_dir.glob("*.mp4")}
    titles_df = titles_df[titles_df['item'].isin(available)].head(request.max_videos)

    indexed = 0
    for _, row in titles_df.iterrows():
        video_id = str(row['item'])
        title = row['title']
        video_path = video_dir / f"{video_id}.mp4"

        try:
            # Extract frames and audio
            frames = state.video_processor.extract_frames(video_path)
            audio = state.audio_processor.extract_audio(video_path)

            # Move to device and add batch dimension
            frames = frames.unsqueeze(0).to(state.device)
            audio = audio.unsqueeze(0).to(state.device)

            # Encode
            with torch.no_grad():
                embedding = state.model.encode_video(frames, audio, [title])

            # Store
            state.video_index[video_id] = embedding.squeeze(0).cpu()
            state.video_metadata[video_id] = {"title": title, "path": str(video_path)}
            indexed += 1

            print(f"Indexed: {video_id} - {title[:40]}...")

        except Exception as e:
            print(f"Failed to index {video_id}: {e}")

    state.rebuild_matrix()

    return {
        "indexed": indexed,
        "total_in_index": state.index_size(),
        "message": f"Indexed {indexed} real videos with multimodal embeddings"
    }


@app.post("/index")
async def index_video(video: VideoInput):
    """Index a single video by path."""
    if not state.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    video_path = Path(video.video_path) if video.video_path else None

    if video_path and video_path.exists():
        # Process real video
        frames = state.video_processor.extract_frames(video_path)
        audio = state.audio_processor.extract_audio(video_path)

        frames = frames.unsqueeze(0).to(state.device)
        audio = audio.unsqueeze(0).to(state.device)

        with torch.no_grad():
            embedding = state.model.encode_video(frames, audio, [video.title])
    else:
        # Fallback to text-only
        dummy_frames = torch.zeros(1, 8, 3, 224, 224).to(state.device)
        dummy_audio = torch.zeros(1, 80000).to(state.device)

        with torch.no_grad():
            embedding = state.model.encode_video(dummy_frames, dummy_audio, [video.title])

    state.video_index[video.video_id] = embedding.squeeze(0).cpu()
    state.video_metadata[video.video_id] = {"title": video.title}
    state.rebuild_matrix()

    return {"video_id": video.video_id, "indexed": True, "index_size": state.index_size()}


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """Get recommendations for a video."""
    if request.video_id not in state.video_index:
        raise HTTPException(status_code=404, detail=f"Video '{request.video_id}' not found")

    if state.index_size() < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 videos")

    query_emb = state.video_index[request.video_id]
    similarities = query_emb @ state.embeddings_matrix.T

    top_k = min(request.top_k + 1, len(state.video_ids))
    scores, indices = torch.topk(similarities, k=top_k)

    recommendations = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        vid = state.video_ids[idx]
        if vid != request.video_id:
            meta = state.video_metadata.get(vid, {})
            recommendations.append({
                "video_id": vid,
                "score": round(score, 4),
                "title": meta.get("title", "")[:60]
            })
            if len(recommendations) >= request.top_k:
                break

    return {
        "query_video_id": request.video_id,
        "query_title": state.video_metadata.get(request.video_id, {}).get("title", "")[:60],
        "recommendations": recommendations
    }


@app.get("/index/list")
async def list_videos():
    """List indexed videos."""
    return {
        "videos": [
            {"video_id": vid, "title": state.video_metadata.get(vid, {}).get("title", "")[:50]}
            for vid in state.video_ids
        ],
        "total": state.index_size()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
