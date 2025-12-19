-- Database Initialization Script
-- This runs automatically when PostgreSQL container starts

-- Enable pgvector extension for similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create videos table
CREATE TABLE IF NOT EXISTS videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    video_path VARCHAR(1000),
    tags TEXT[],
    embedding vector(512),  -- 512-dimensional vector
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for fast similarity search
CREATE INDEX IF NOT EXISTS ix_video_embedding
ON videos
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index on video_id for fast lookups
CREATE INDEX IF NOT EXISTS ix_video_id ON videos(video_id);

-- Create user interactions table
CREATE TABLE IF NOT EXISTS user_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    video_id VARCHAR(255) NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,
    value FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for user interactions
CREATE INDEX IF NOT EXISTS ix_user_id ON user_interactions(user_id);
CREATE INDEX IF NOT EXISTS ix_interaction_video_id ON user_interactions(video_id);
CREATE INDEX IF NOT EXISTS ix_timestamp ON user_interactions(timestamp);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at
CREATE TRIGGER update_videos_updated_at
    BEFORE UPDATE ON videos
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Log success
DO $$
BEGIN
    RAISE NOTICE 'Database initialized successfully!';
END $$;
