"""
Database Configuration

Handles connection settings and database URL construction.
Uses environment variables for sensitive data.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DatabaseConfig:
    """Database configuration from environment variables."""

    # PostgreSQL connection settings
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "video_recommender")

    # Connection pool settings
    POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))

    @classmethod
    def get_database_url(cls, async_driver: bool = False) -> str:
        """
        Build the database connection URL.

        Args:
            async_driver: Use asyncpg instead of psycopg2

        Returns:
            PostgreSQL connection URL
        """
        driver = "postgresql+asyncpg" if async_driver else "postgresql+psycopg2"

        return (
            f"{driver}://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}"
            f"@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
        )

    @classmethod
    def print_config(cls):
        """Print current configuration (hide password)."""
        print(f"Database Configuration:")
        print(f"  Host: {cls.POSTGRES_HOST}")
        print(f"  Port: {cls.POSTGRES_PORT}")
        print(f"  User: {cls.POSTGRES_USER}")
        print(f"  Database: {cls.POSTGRES_DB}")
        print(f"  Pool Size: {cls.POOL_SIZE}")


# Create default .env template if it doesn't exist
ENV_TEMPLATE = """# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here
POSTGRES_DB=video_recommender

# Connection Pool
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
"""


def create_env_template():
    """Create a .env.template file for reference."""
    env_path = Path(__file__).parent.parent.parent / ".env.template"
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(ENV_TEMPLATE)
        print(f"Created {env_path}")


if __name__ == "__main__":
    DatabaseConfig.print_config()
    print(f"\nDatabase URL: {DatabaseConfig.get_database_url()}")
