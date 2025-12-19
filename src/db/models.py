"""
SQLAlchemy Database Models

These define the structure of our database tables.
SQLAlchemy automatically creates tables and handles SQL generation.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Text, DateTime, Integer, Float, Index
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Try to import pgvector, fall back to ARRAY if not available
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    print("pgvector not installed. Using ARRAY for embeddings.")


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Video(Base):
    """
    Video metadata and embeddings.

    This table stores all information about indexed videos,
    including their computed embeddings for similarity search.
    """

    __tablename__ = "videos"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Video identifier (from external source)
    video_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    # Metadata
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    video_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Tags stored as array
    tags: Mapped[Optional[list]] = mapped_column(ARRAY(String), nullable=True)

    # Embedding (512-dimensional vector)
    # Using pgvector for efficient similarity search
    if HAS_PGVECTOR:
        embedding = Column(Vector(512), nullable=True)
    else:
        embedding = Column(ARRAY(Float), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    def __repr__(self):
        return f"<Video(video_id='{self.video_id}', title='{self.title[:30]}...')>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "video_id": self.video_id,
            "title": self.title,
            "description": self.description,
            "video_path": self.video_path,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class UserInteraction(Base):
    """
    User interactions with videos (for future user-based recommendations).

    Stores events like: views, likes, shares, watch time.
    """

    __tablename__ = "user_interactions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    video_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Interaction type: 'view', 'like', 'share', 'complete'
    interaction_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Additional data (e.g., watch_time in seconds)
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # When the interaction happened
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    def __repr__(self):
        return f"<Interaction(user='{self.user_id}', video='{self.video_id}', type='{self.interaction_type}')>"


# Create index for fast similarity search with pgvector
if HAS_PGVECTOR:
    # IVFFlat index for approximate nearest neighbor search
    # This makes similarity queries much faster on large datasets
    video_embedding_index = Index(
        'ix_video_embedding',
        Video.embedding,
        postgresql_using='ivfflat',
        postgresql_with={'lists': 100},  # Number of clusters
        postgresql_ops={'embedding': 'vector_cosine_ops'}
    )
