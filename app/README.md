# Insights API

A FastAPI-based service for generating investigative insights from forensic case documents using LLM-powered summarization and analysis.

## Features

- **Automatic Insight Generation**: Generates structured investigative insights from case documents
- **Smart Summarization**: Automatically uses summarization for large documents (>120k tokens)
- **Vector Storage**: Stores insights with embeddings in Qdrant for retrieval
- **Caching**: Retrieves existing insights without regeneration
- **Modular Architecture**: Clean separation of concerns for maintainability

## Architecture
```
insights_api/
├── app/
│   ├── api/              # API routes and endpoints
│   ├── config/           # Configuration and prompts
│   ├── models/           # Pydantic schemas
│   ├── services/         # Business logic
│   └── utils/            # Utility functions
└── requirements.txt
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd insights_api
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure settings**

Edit `app/config/settings.py` to match your environment:
```python
QDRANT_URL = "http://192.168.1.13:6333"
LLM_URL = "http://192.168.1.11:8078/v1/chat/completions"
EMBEDDING_URL = "http://192.168.1.11:8074/encode_text"
```

## Usage

### Starting the Server
```bash
# From the insights_api directory
python -m app.main

# Or using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8081 --reload
```

The API will be available at `http://localhost:8081`

### API Endpoints

#### 1. **Generate Insights**
```bash
POST /generate-insights
```

**Request Body:**
```json
{
  "case_id": "case123",
  "file_id": "file234",
  "case_type": "Drug Trafficking and Substance Abuse",
  "data_source": "ufed_extraction"
}
```

**Response:**
```json
{
  "case_id": "case123",
  "file_id": "file234",
  "case_type": "Drug Trafficking and Substance Abuse",
  "data_source": "ufed_extraction",
  "insights": "Summary: ...\nCriminal Activities: ...",
  "source": "generated",
  "chunk_count": 1570,
  "total_tokens": 1135956,
  "used_summarization": true,
  "num_summary_chunks": 23
}
```

**Example with cURL:**
```bash
curl -X POST "http://localhost:8081/generate-insights" \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "case123",
    "file_id": "file234",
    "case_type": "Financial Crime Investigation",
    "data_source": "audio"
  }'
```

#### 2. **Health Check**
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "qdrant": "connected",
  "collections": 5
}
```

#### 3. **Root**
```bash
GET /
```

Returns API information, supported case types, and data sources.

## Supported Data Sources

- `audio` - Audio recordings
- `video` - Video recordings
- `image` - Image files
- `ufed_extraction` - UFED mobile device extractions
- `others` - Other data sources

## Common Case Types

- Drug Trafficking and Substance Abuse
- Arms Trafficking
- Cyber Crime
- Terrorism
- Murder and Homicide
- Suicide
- General

**Note:** Any case type string is accepted, not limited to the above.

## How It Works

### 1. **Check for Existing Insights**
The API first checks if insights already exist for the given `file_id`, `case_type`, and `data_source` combination.

### 2. **Retrieve Document Chunks**
If no existing insights are found, all document chunks are retrieved from Qdrant.

### 3. **Token Counting**
The total token count is calculated to determine the processing approach.

### 4. **Processing Logic**

#### **For documents < 120k tokens:**
- Insights are generated directly from the concatenated content

#### **For documents > 120k tokens:**
- Content is split into 50k token chunks
- Each chunk is summarized individually
- All summaries are merged into a ~5k token mega summary
- Final insights are generated from the mega summary

### 5. **Storage**
Generated insights are stored in Qdrant with:
- Dense embeddings (from custom embedding service)
- Sparse embeddings (BM25)
- Metadata matching the original chunks
- `content_type: "insights"` for filtering

## Configuration

### Token Limits
Edit `app/config/settings.py`:
```python
MAX_TOKENS_BEFORE_SUMMARIZATION = 120000  # Threshold for summarization
CHUNK_SIZE_FOR_SUMMARIZATION = 50000      # Size of each summary chunk
MEGA_SUMMARY_TARGET = 5000                # Target tokens for mega summary
INSIGHTS_MAX_TOKENS = 2000                # Max tokens in final insights
```

### LLM Settings
```python
LLM_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0.3
LLM_TIMEOUT = 300.0  # seconds
```

### Custom Prompts
Edit prompts in `app/config/prompts.py` to customize the insight generation format.

## Logging

Logs are written to:
- Console (stdout)
- `insights_api.log` file

Log levels can be configured in `app/main.py`:
```python
logging.basicConfig(level=logging.INFO)
```

## Project Structure Details
```
app/
├── main.py                    # FastAPI application entry point
├── api/
│   ├── __init__.py
│   └── routes.py              # API endpoint handlers
├── config/
│   ├── __init__.py
│   ├── settings.py            # Configuration constants
│   └── prompts.py             # LLM prompt templates
├── models/
│   ├── __init__.py
│   └── schemas.py             # Pydantic request/response models
├── services/
│   ├── __init__.py
│   ├── insights_service.py    # Main insights generation logic
│   ├── llm_service.py         # LLM API interactions
│   └── qdrant_service.py      # Vector database operations
└── utils/
    ├── __init__.py
    └── token_utils.py         # Token counting and text splitting
```

## Error Handling

The API provides detailed error responses:

- `400` - Invalid input (missing case_type, invalid data_source)
- `404` - No chunks found for the given file_id
- `500` - Internal server errors (LLM failures, Qdrant issues, etc.)

Example error response:
```json
{
  "detail": "No chunks found for file_id=file234"
}
```

## Performance Considerations

### Large Documents
For documents with 1M+ tokens:
- Processing time: ~10-15 minutes
- Number of LLM calls: ~25-30 (for 50k chunk summarization)
- Recommended: Use batch processing or background tasks for very large documents

### Optimization Tips
1. Adjust `CHUNK_SIZE_FOR_SUMMARIZATION` to reduce LLM calls
2. Implement parallel chunk summarization (future enhancement)
3. Use caching - insights are automatically cached in Qdrant

## Development

### Running in Development Mode
```bash
uvicorn app.main:app --reload --log-level debug
```

### Testing
```bash
# Test health endpoint
curl http://localhost:8081/health

# Test insight generation with small document
curl -X POST http://localhost:8081/generate-insights \
  -H "Content-Type: application/json" \
  -d '{"case_id": "test", "file_id": "small_file", "case_type": "General", "data_source": "others"}'
```

## Troubleshooting

### Common Issues

**1. Connection refused to Qdrant**
- Verify Qdrant is running: `curl http://192.168.1.13:6333/collections`
- Check `QDRANT_URL` in `app/config/settings.py`

**2. LLM timeout errors**
- Increase `LLM_TIMEOUT` in settings
- Check LLM service health: `curl http://192.168.1.11:8078/health`

**3. Embedding API errors**
- Verify embedding service: `curl http://192.168.1.11:8074/health`
- Check request format matches expected form-data structure

**4. Out of memory errors**
- Reduce `CHUNK_SIZE_FOR_SUMMARIZATION`
- Process documents in batches