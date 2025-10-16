# Insights API

A FastAPI-based service for generating investigative insights from forensic data using LLM and vector database storage.

## Features

- Automatic insights generation using LLM
- Map-reduce approach for large datasets (>100k tokens)
- Vector embeddings storage in Qdrant
- Flexible case type support
- Dense and sparse embeddings

## Architecture

- **FastAPI** for REST API
- **Qdrant** for vector storage
- **LangChain** for map-reduce processing
- **FastEmbed** for sparse embeddings
- **Custom LLM** integration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables (copy .env.example to .env):
```bash
QDRANT_URL=http://192.168.1.13:6333
COLLECTION_NAME=finance_docs
LLM_URL=http://192.168.1.11:8078/v1/chat/completions
EMBEDDING_URL=http://192.168.1.11:8074/encode_text
MAX_TOKENS_BEFORE_MAP_REDUCE=100000
MAX_TOKENS_PER_MAP_CHUNK=50000
```

3. Run the API:
```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8081
```

## API Endpoints

### POST /generate-insights
Generate or retrieve insights for a file.

**Request:**
```json
{
  "case_id": "string",
  "file_id": "string",
  "case_type": "string",
  "data_source": "audio|video|image|ufed_extraction|others"
}
```

**Response:**
```json
{
  "case_id": "string",
  "file_id": "string",
  "case_type": "string",
  "data_source": "string",
  "insights": "string",
  "source": "existing|generated",
  "chunk_count": 0,
  "total_tokens": 0,
  "used_map_reduce": false
}
```

### GET /health
Health check endpoint.

### GET /
API information and available endpoints.

## Token Thresholds

- **Direct approach**: < 100k tokens
- **Map-reduce approach**: ≥ 100k tokens
- **Map chunk size**: 50k tokens per chunk

## Supported Data Sources

- audio
- video
- image
- ufed_extraction
- others