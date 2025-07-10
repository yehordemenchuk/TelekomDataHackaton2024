from pydantic import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Claude API Configuration
    anthropic_api_key: str = "your_claude_api_key_here"
    
    # Database Configuration
    database_url: str = "postgresql://user:password@localhost:5432/legal_ai_db"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    
    # FAISS Configuration
    faiss_index_path: str = "./faiss_index"
    metadata_storage_path: str = "./document_metadata"
    
    # Application Configuration
    secret_key: str = "your_secret_key_here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # MCP Configuration
    mcp_server_port: int = 8001
    mcp_client_timeout: int = 30
    
    # Legal Document Storage
    document_storage_path: str = "./legal_documents"
    template_storage_path: str = "./templates"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # Embedding Model Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Claude Model Configuration
    claude_model: str = "claude-3-sonnet-20240229"
    max_tokens: int = 4000
    temperature: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Ensure required directories exist
os.makedirs(settings.document_storage_path, exist_ok=True)
os.makedirs(settings.template_storage_path, exist_ok=True)
os.makedirs(settings.faiss_index_path, exist_ok=True)
os.makedirs(settings.metadata_storage_path, exist_ok=True) 