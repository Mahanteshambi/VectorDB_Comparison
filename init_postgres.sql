-- Initialize PostgreSQL with pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for vector storage
CREATE TABLE IF NOT EXISTS vectors (
    id SERIAL PRIMARY KEY,
    vector_id INTEGER NOT NULL,
    embedding vector(1024),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS vectors_embedding_idx ON vectors 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create index for vector_id lookups
CREATE INDEX IF NOT EXISTS vectors_vector_id_idx ON vectors (vector_id);
