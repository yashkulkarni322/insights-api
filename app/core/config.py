from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Qdrant Configuration
    QDRANT_URL: str = "http://192.168.1.13:6333"
    COLLECTION_NAME: str = "finance_docs"
    
    # LLM Configuration
    LLM_URL: str = "http://192.168.1.11:8078/v1/chat/completions"
    EMBEDDING_URL: str = "http://192.168.1.11:8074/encode_text"
    
    # Token Limits
    MAX_TOKENS_BEFORE_MAP_REDUCE: int = 120000
    MAX_TOKENS_PER_MAP_CHUNK: int = 50000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()
