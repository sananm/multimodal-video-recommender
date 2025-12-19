"""
Repository Pattern for Database Operations

Repositories provide a clean interface for database operations,
hiding the SQL complexity from the rest of the application.
"""

from typing import Optional
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .models import Video, UserInteraction


class VideoRepository:
    """
    Repository for Video operations.

    Provides methods for CRUD operations and similarity search.
    """

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        video_id: str,
        title: str,
        description: str = None,
        video_path: str = None,
        tags: list[str] = None,
        embedding: list[float] = None
    ) -> Video:
        """
        Create a new video record.

        Args:
            video_id: Unique identifier
            title: Video title
            description: Video description
            video_path: Path to video file
            tags: List of tags
            embedding: 512-dimensional embedding vector

        Returns:
            Created Video object
        """
        video = Video(
            video_id=video_id,
            title=title,
            description=description,
            video_path=video_path,
            tags=tags,
            embedding=embedding
        )

        self.session.add(video)
        self.session.commit()
        self.session.refresh(video)

        return video

    def get_by_video_id(self, video_id: str) -> Optional[Video]:
        """Get video by its external ID."""
        stmt = select(Video).where(Video.video_id == video_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_all(self, limit: int = 100, offset: int = 0) -> list[Video]:
        """Get all videos with pagination."""
        stmt = select(Video).offset(offset).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def update_embedding(self, video_id: str, embedding: list[float]) -> Optional[Video]:
        """Update the embedding for a video."""
        video = self.get_by_video_id(video_id)
        if video:
            video.embedding = embedding
            self.session.commit()
            self.session.refresh(video)
        return video

    def delete(self, video_id: str) -> bool:
        """Delete a video by ID."""
        video = self.get_by_video_id(video_id)
        if video:
            self.session.delete(video)
            self.session.commit()
            return True
        return False

    def count(self) -> int:
        """Count total videos."""
        stmt = select(Video)
        return len(list(self.session.execute(stmt).scalars().all()))

    def find_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        exclude_video_id: str = None
    ) -> list[tuple[Video, float]]:
        """
        Find similar videos using cosine similarity.

        This uses pgvector's efficient similarity search.

        Args:
            embedding: Query embedding vector
            limit: Number of results to return
            exclude_video_id: Video ID to exclude from results

        Returns:
            List of (Video, similarity_score) tuples
        """
        # Convert embedding to string format for pgvector
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        # Build query based on whether we have pgvector
        # Using cosine distance: 1 - cosine_similarity
        # So we order by distance ASC (smaller = more similar)
        if exclude_video_id:
            query = text("""
                SELECT *, 1 - (embedding <=> :embedding) as similarity
                FROM videos
                WHERE video_id != :exclude_id AND embedding IS NOT NULL
                ORDER BY embedding <=> :embedding
                LIMIT :limit
            """)
            result = self.session.execute(
                query,
                {"embedding": embedding_str, "exclude_id": exclude_video_id, "limit": limit}
            )
        else:
            query = text("""
                SELECT *, 1 - (embedding <=> :embedding) as similarity
                FROM videos
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> :embedding
                LIMIT :limit
            """)
            result = self.session.execute(
                query,
                {"embedding": embedding_str, "limit": limit}
            )

        # Convert results
        videos_with_scores = []
        for row in result:
            video = Video(
                id=row.id,
                video_id=row.video_id,
                title=row.title,
                description=row.description,
                video_path=row.video_path,
                tags=row.tags,
                embedding=row.embedding,
                created_at=row.created_at,
                updated_at=row.updated_at
            )
            videos_with_scores.append((video, row.similarity))

        return videos_with_scores

    def find_similar_fallback(
        self,
        embedding: list[float],
        limit: int = 10,
        exclude_video_id: str = None
    ) -> list[tuple[Video, float]]:
        """
        Find similar videos without pgvector (slower, for testing).

        Computes cosine similarity in Python.
        """
        import numpy as np

        query_emb = np.array(embedding)

        all_videos = self.get_all(limit=1000)

        scores = []
        for video in all_videos:
            if video.embedding is None:
                continue
            if exclude_video_id and video.video_id == exclude_video_id:
                continue

            video_emb = np.array(video.embedding)

            # Cosine similarity
            similarity = np.dot(query_emb, video_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(video_emb)
            )

            scores.append((video, float(similarity)))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:limit]


class InteractionRepository:
    """Repository for user interaction operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        user_id: str,
        video_id: str,
        interaction_type: str,
        value: float = None
    ) -> UserInteraction:
        """Record a user interaction."""
        interaction = UserInteraction(
            user_id=user_id,
            video_id=video_id,
            interaction_type=interaction_type,
            value=value
        )

        self.session.add(interaction)
        self.session.commit()
        self.session.refresh(interaction)

        return interaction

    def get_user_history(
        self,
        user_id: str,
        limit: int = 100
    ) -> list[UserInteraction]:
        """Get a user's interaction history."""
        stmt = (
            select(UserInteraction)
            .where(UserInteraction.user_id == user_id)
            .order_by(UserInteraction.timestamp.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_video_interactions(
        self,
        video_id: str,
        limit: int = 100
    ) -> list[UserInteraction]:
        """Get all interactions for a video."""
        stmt = (
            select(UserInteraction)
            .where(UserInteraction.video_id == video_id)
            .order_by(UserInteraction.timestamp.desc())
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())
