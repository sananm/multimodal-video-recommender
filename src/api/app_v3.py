"""
FastAPI Application v3 - With PostgreSQL + pgvector

This version stores embeddings in PostgreSQL for persistence and scalability.
"""

import os
import torch
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.device import get_device
from data.processors import VideoProcessor, AudioProcessor
from db.database import get_database, get_db_session, init_database
from db.models import Base, Video
from db.repository import VideoRepository


# ============ Dependency ============
def get_db():
    """FastAPI dependency for database sessions."""
    db = get_database()
    session = db.SessionLocal()
    try:
        yield session
    finally:
        session.close()


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
    title="Multimodal Video Recommender v3",
    description="With PostgreSQL + pgvector for persistent storage",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Global State (Model Only) ============

class ModelState:
    def __init__(self):
        self.model = None
        self.device = None
        self.video_processor = None
        self.audio_processor = None

    def is_loaded(self) -> bool:
        return self.model is not None


state = ModelState()


# ============ Startup ============

@app.on_event("startup")
async def load_model():
    print("Initializing database...")
    try:
        init_database()
        print("Database initialized!")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        print("Make sure PostgreSQL is running: docker-compose up -d db")

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

        # Load trained weights if available
        checkpoint_path = Path("checkpoints/best.pt")
        if checkpoint_path.exists():
            print(f"Loading trained weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=state.device)
            if 'model_state_dict' in checkpoint:
                state.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                state.model.load_state_dict(checkpoint)
            print("Trained weights loaded!")

        state.model.to(state.device)
        state.model.eval()

        print(f"Model loaded successfully on {state.device}")

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()


# ============ Endpoints ============

@app.get("/")
async def root():
    return {
        "name": "Multimodal Video Recommender v3",
        "version": "3.0.0",
        "storage": "PostgreSQL + pgvector"
    }


@app.get("/health")
async def health(db: Session = Depends(get_db)):
    repo = VideoRepository(db)
    return {
        "status": "ok",
        "model_loaded": state.is_loaded(),
        "index_size": repo.count(),
        "device": str(state.device),
        "storage": "postgresql"
    }


@app.post("/index/microlens")
async def index_microlens(request: IndexMicroLensRequest, db: Session = Depends(get_db)):
    """
    Index videos from MicroLens dataset into PostgreSQL.
    """
    if not state.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    import pandas as pd

    repo = VideoRepository(db)
    data_dir = Path("data/raw/microlens")
    video_dir = data_dir / "videos"

    # Load titles
    titles_df = pd.read_csv(data_dir / "MicroLens-50k_titles.csv")

    # Get available videos
    available = {int(p.stem) for p in video_dir.glob("*.mp4")}
    titles_df = titles_df[titles_df['item'].isin(available)].head(request.max_videos)

    indexed = 0
    skipped = 0

    for _, row in titles_df.iterrows():
        video_id = str(row['item'])
        title = row['title']
        video_path = video_dir / f"{video_id}.mp4"

        # Check if already indexed
        existing = repo.get_by_video_id(video_id)
        if existing and existing.embedding is not None:
            skipped += 1
            continue

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

            # Convert to list for PostgreSQL
            embedding_list = embedding.squeeze(0).cpu().tolist()

            # Store in database
            if existing:
                repo.update_embedding(video_id, embedding_list)
            else:
                repo.create(
                    video_id=video_id,
                    title=title,
                    video_path=str(video_path),
                    embedding=embedding_list
                )

            indexed += 1
            print(f"Indexed: {video_id} - {title[:40]}...")

        except Exception as e:
            print(f"Failed to index {video_id}: {e}")

    return {
        "indexed": indexed,
        "skipped": skipped,
        "total_in_index": repo.count(),
        "message": f"Indexed {indexed} videos, skipped {skipped} (already in DB)"
    }


@app.post("/index")
async def index_video(video: VideoInput, db: Session = Depends(get_db)):
    """Index a single video by path."""
    if not state.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    repo = VideoRepository(db)
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

    embedding_list = embedding.squeeze(0).cpu().tolist()

    # Check if exists
    existing = repo.get_by_video_id(video.video_id)
    if existing:
        repo.update_embedding(video.video_id, embedding_list)
    else:
        repo.create(
            video_id=video.video_id,
            title=video.title,
            video_path=str(video_path) if video_path else None,
            embedding=embedding_list
        )

    return {
        "video_id": video.video_id,
        "indexed": True,
        "index_size": repo.count()
    }


@app.post("/recommend")
async def recommend(request: RecommendRequest, db: Session = Depends(get_db)):
    """Get recommendations for a video using pgvector similarity search."""
    repo = VideoRepository(db)

    # Get query video
    query_video = repo.get_by_video_id(request.video_id)
    if not query_video:
        raise HTTPException(status_code=404, detail=f"Video '{request.video_id}' not found")

    if query_video.embedding is None:
        raise HTTPException(status_code=400, detail=f"Video '{request.video_id}' has no embedding")

    if repo.count() < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 videos in index")

    # Find similar videos using pgvector
    try:
        similar_videos = repo.find_similar(
            embedding=query_video.embedding,
            limit=request.top_k,
            exclude_video_id=request.video_id
        )
    except Exception as e:
        # Fallback to Python-based similarity if pgvector fails
        print(f"pgvector search failed, using fallback: {e}")
        similar_videos = repo.find_similar_fallback(
            embedding=query_video.embedding,
            limit=request.top_k,
            exclude_video_id=request.video_id
        )

    recommendations = [
        {
            "video_id": video.video_id,
            "score": round(score, 4),
            "title": video.title[:60] if video.title else ""
        }
        for video, score in similar_videos
    ]

    return {
        "query_video_id": request.video_id,
        "query_title": query_video.title[:60] if query_video.title else "",
        "recommendations": recommendations,
        "storage": "postgresql"
    }


@app.get("/index/list")
async def list_videos(db: Session = Depends(get_db)):
    """List indexed videos from database."""
    repo = VideoRepository(db)
    videos = repo.get_all(limit=100)

    return {
        "videos": [
            {
                "video_id": v.video_id,
                "title": v.title[:50] if v.title else "",
                "has_embedding": v.embedding is not None
            }
            for v in videos
        ],
        "total": repo.count()
    }


@app.delete("/index/{video_id}")
async def delete_video(video_id: str, db: Session = Depends(get_db)):
    """Delete a video from the index."""
    repo = VideoRepository(db)

    if repo.delete(video_id):
        return {"deleted": True, "video_id": video_id}
    else:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")


@app.delete("/index")
async def clear_index(db: Session = Depends(get_db)):
    """Clear all videos from the index."""
    repo = VideoRepository(db)
    count = repo.count()

    for video in repo.get_all(limit=10000):
        repo.delete(video.video_id)

    return {"cleared": count, "message": f"Deleted {count} videos from index"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
