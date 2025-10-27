"""Application configuration and settings"""

# API URLs
QDRANT_URL = "http://192.168.1.13:6333"
COLLECTION_NAME = "finance_docs"
LLM_URL = "http://192.168.1.11:8077/v1/chat/completions"
EMBEDDING_URL = "http://192.168.1.11:8074/encode_text"

# Token configuration
MAX_TOKENS_BEFORE_SUMMARIZATION = 120000
CHUNK_SIZE_FOR_SUMMARIZATION = 50000
MEGA_SUMMARY_TARGET = 5000
INSIGHTS_MAX_TOKENS = 2000

# LLM configuration
LLM_MODEL = "RedHatAI/gemma-3-27b-it-quantized.w4a16"
LLM_TEMPERATURE = 0.3
LLM_TIMEOUT = 300.0

# Embedding configuration
EMBEDDING_TIMEOUT = 60.0