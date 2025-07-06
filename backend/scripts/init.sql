-- PostgreSQL initialization script for RAG Platform
-- This script sets up the initial database structure

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search
CREATE EXTENSION IF NOT EXISTS "pgcrypto"; -- For password hashing

-- Create database user with appropriate permissions (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'rag_user') THEN
        CREATE USER rag_user WITH PASSWORD 'rag_password';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE rag_platform TO rag_user;
GRANT ALL ON SCHEMA public TO rag_user;

-- Create initial schemas
CREATE SCHEMA IF NOT EXISTS rag_platform;
GRANT ALL ON SCHEMA rag_platform TO rag_user;

-- Set default schema for user
ALTER USER rag_user SET search_path = rag_platform, public;

-- Create initial tables (will be managed by Alembic migrations)
-- These are basic structures that will be refined by migrations

-- Document storage table
CREATE TABLE IF NOT EXISTS rag_platform.documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    file_size INTEGER NOT NULL,
    file_hash VARCHAR(64),
    document_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    total_chunks INTEGER DEFAULT 0,
    processed_chunks INTEGER DEFAULT 0,
    processing_time FLOAT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Document chunks table
CREATE TABLE IF NOT EXISTS rag_platform.document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES rag_platform.documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    chunk_size INTEGER NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    metadata JSONB,
    vector_id VARCHAR(255),
    embedding_model VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);

-- Search queries table
CREATE TABLE IF NOT EXISTS rag_platform.search_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    search_type VARCHAR(50) NOT NULL,
    limit_val INTEGER NOT NULL DEFAULT 10,
    min_score FLOAT,
    filters JSONB,
    total_results INTEGER NOT NULL DEFAULT 0,
    search_time FLOAT NOT NULL,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    ip_address VARCHAR(45),
    user_agent VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Search results table
CREATE TABLE IF NOT EXISTS rag_platform.search_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID REFERENCES rag_platform.search_queries(id) ON DELETE CASCADE,
    document_id UUID REFERENCES rag_platform.documents(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES rag_platform.document_chunks(id) ON DELETE SET NULL,
    score FLOAT NOT NULL,
    rank INTEGER NOT NULL,
    result_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    title VARCHAR(255) NOT NULL,
    source VARCHAR(500) NOT NULL,
    context TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Embedding cache table
CREATE TABLE IF NOT EXISTS rag_platform.embedding_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_hash VARCHAR(64) NOT NULL UNIQUE,
    content TEXT NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,
    embedding_dimensions INTEGER NOT NULL,
    embedding_provider VARCHAR(50) NOT NULL,
    hit_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System metrics table
CREATE TABLE IF NOT EXISTS rag_platform.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    value FLOAT NOT NULL,
    labels JSONB,
    unit VARCHAR(20),
    description VARCHAR(255),
    component VARCHAR(50),
    environment VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Job status table
CREATE TABLE IF NOT EXISTS rag_platform.job_status (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(255) NOT NULL UNIQUE,
    job_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    task_name VARCHAR(100) NOT NULL,
    args JSONB,
    kwargs JSONB,
    progress FLOAT DEFAULT 0.0,
    total_items INTEGER,
    processed_items INTEGER DEFAULT 0,
    result JSONB,
    error_message TEXT,
    traceback TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_status ON rag_platform.documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON rag_platform.documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON rag_platform.documents(created_at);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON rag_platform.document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON rag_platform.document_chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_vector_id ON rag_platform.document_chunks(vector_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm ON rag_platform.document_chunks USING gin(content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_search_queries_time ON rag_platform.search_queries(created_at);
CREATE INDEX IF NOT EXISTS idx_search_queries_user ON rag_platform.search_queries(user_id);
CREATE INDEX IF NOT EXISTS idx_search_queries_session ON rag_platform.search_queries(session_id);
CREATE INDEX IF NOT EXISTS idx_search_results_query ON rag_platform.search_results(query_id);
CREATE INDEX IF NOT EXISTS idx_search_results_document ON rag_platform.search_results(document_id);
CREATE INDEX IF NOT EXISTS idx_search_results_score ON rag_platform.search_results(score);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_hash ON rag_platform.embedding_cache(content_hash);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON rag_platform.system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON rag_platform.system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_component ON rag_platform.system_metrics(component);
CREATE INDEX IF NOT EXISTS idx_job_status_job_id ON rag_platform.job_status(job_id);
CREATE INDEX IF NOT EXISTS idx_job_status_status ON rag_platform.job_status(status);
CREATE INDEX IF NOT EXISTS idx_job_status_created_at ON rag_platform.job_status(created_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION rag_platform.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON rag_platform.documents 
    FOR EACH ROW EXECUTE FUNCTION rag_platform.update_updated_at_column();

-- Insert initial configuration data
INSERT INTO rag_platform.documents (id, title, filename, original_filename, file_size, document_type, status) 
VALUES (
    '00000000-0000-0000-0000-000000000000',
    'System Information',
    'system_info.txt',
    'System Information',
    25,
    'text',
    'completed'
) ON CONFLICT (id) DO NOTHING;

-- Create view for document statistics
CREATE OR REPLACE VIEW rag_platform.document_stats AS
SELECT 
    COUNT(*) as total_documents,
    COUNT(*) FILTER (WHERE status = 'completed') as processed_documents,
    COUNT(*) FILTER (WHERE status = 'pending') as pending_documents,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_documents,
    SUM(file_size) as total_file_size,
    AVG(file_size) as avg_file_size
FROM rag_platform.documents;

-- Grant permissions on all objects
GRANT ALL ON ALL TABLES IN SCHEMA rag_platform TO rag_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA rag_platform TO rag_user;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA rag_platform TO rag_user;

-- Set up logging
\echo 'PostgreSQL initialization completed successfully'
