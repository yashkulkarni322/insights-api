from fastapi import FastAPI, HTTPException
from app.models.schemas import InsightsRequest, InsightsResponse
from app.models.enums import CaseType, DataSource
from app.services.insights_service import InsightsService
from app.core.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(title="Insights API", version="2.0.0")
insights_service = InsightsService()

@app.get("/")
async def root():
    return {
        "message": "Insights API is running",
        "version": "2.0.0",
        "features": {
            "map_reduce": True,
            "token_threshold": settings.MAX_TOKENS_BEFORE_MAP_REDUCE,
            "flexible_case_types": True
        },
        "endpoints": {
            "/generate-insights": "POST - Generate or retrieve insights for a file",
            "/health": "GET - Health check endpoint"
        },
        "common_case_types": [ct.value for ct in CaseType],
        "note": "Any case type string is now accepted",
        "supported_data_sources": [ds.value for ds in DataSource]
    }

@app.post("/generate-insights", response_model=InsightsResponse)
async def generate_insights(request: InsightsRequest):
    """Main endpoint to generate or retrieve insights"""
    try:
        logger.info(f"Processing: case_id={request.case_id}, file_id={request.file_id}")
        
        if not request.case_type or not request.case_type.strip():
            raise HTTPException(status_code=400, detail="case_type cannot be empty")
        
        if request.data_source not in [ds.value for ds in DataSource]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data_source. Must be one of: {[ds.value for ds in DataSource]}"
            )
        
        result = await insights_service.get_or_generate_insights(
            case_id=request.case_id,
            file_id=request.file_id,
            case_type=request.case_type,
            data_source=request.data_source
        )
        
        return InsightsResponse(
            case_id=request.case_id,
            file_id=request.file_id,
            case_type=request.case_type,
            data_source=request.data_source,
            **result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        collections = insights_service.qdrant.client.get_collections()
        return {
            "status": "healthy",
            "qdrant": "connected",
            "collections": len(collections.collections),
            "features": {
                "map_reduce_enabled": True,
                "token_threshold": settings.MAX_TOKENS_BEFORE_MAP_REDUCE,
                "flexible_case_types": True
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}