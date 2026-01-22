"""
Configuration management for the RAG system.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_provider: str = "openai"
    
    # Vector Database
    vector_db_type: str = "pinecone"
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "rag-qa-index"
    
    # Embeddings
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    embedding_provider: str = "openai"
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    confidence_threshold: float = 0.7
    max_retries: int = 3
    temperature: float = 0.0
    
    # GCP Configuration
    gcp_project_id: Optional[str] = None
    gcp_bucket_name: Optional[str] = None
    use_gcp_storage: bool = False
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    
    # Security
    secret_key: str = "change-this-secret-key"
    algorithm: str = "HS256"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
