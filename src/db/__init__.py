"""
Database layer for video recommendation system.
"""

from .config import DatabaseConfig
from .models import Base, Video, UserInteraction
from .database import Database, get_database, init_database, get_db_session
from .repository import VideoRepository, InteractionRepository

__all__ = [
    "DatabaseConfig",
    "Base",
    "Video",
    "UserInteraction",
    "Database",
    "get_database",
    "init_database",
    "get_db_session",
    "VideoRepository",
    "InteractionRepository"
]
