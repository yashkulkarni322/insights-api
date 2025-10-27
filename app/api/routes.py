"""API route handlers"""
from fastapi import APIRouter, HTTPException
import logging

from app.models.schemas import InsightsRequest, InsightsResponse, CaseType, DataSource
from app.services.qdrant_service import get_points_by_file_id, store_insights_in_qdrant
from app.services.insights_service import (
    generate_insights_with_summarization,
    generate_insights_direct,
    should_use_summarization
)
from app.services.llm_service import generate_dense_embedding
from app.utils.token_utils import count_tokens
from app.config.settings import COLLECTION_NAME

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    return {
        "message": "Insights API is running",
        "version": "3.0.0",
        "endpoints": {
            "/generate-insights": "POST - Generate or retrieve insights for a file",
            "/health": "GET - Health check endpoint"
        },
        "common_case_types": [ct.value for ct in CaseType],
        "supported_data_sources": [ds.value for ds in DataSource]
    }


@router.post("/generate-insights", response_model=InsightsResponse)
async def generate_insights(request: InsightsRequest):
    """Main endpoint to generate or retrieve insights for a given file"""
    try:
        logger.info("=" * 80)
        logger.info(f"Processing request: case_id={request.case_id}, file_id={request.file_id}")
        logger.info("=" * 80)
        
        # Validate inputs
        if not request.case_type or not request.case_type.strip():
            raise HTTPException(status_code=400, detail="case_type cannot be empty")
        
        if request.data_source not in [ds.value for ds in DataSource]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data_source. Must be one of: {[ds.value for ds in DataSource]}"
            )
        
        # Step 1: Check if insights already exist
        logger.info("Step 1: Checking for existing insights")
        existing_insights = get_points_by_file_id(
            file_id=request.file_id,
            content_type="insights",
            case_type=request.case_type,
            data_source=request.data_source
        )
        
        if existing_insights:
            logger.info(f"Found existing insights for file_id={request.file_id}")
            insights_text = existing_insights[0].payload.get("page_content", "")
            
            return InsightsResponse(
                case_id=request.case_id,
                file_id=request.file_id,
                case_type=request.case_type,
                data_source=request.data_source,
                insights=insights_text,
                source="existing"
            )
        
        # Step 2: Retrieve all chunks
        logger.info("Step 2: Retrieving chunks from Qdrant")
        chunks_points = get_points_by_file_id(
            file_id=request.file_id,
            data_source=request.data_source
        )
        
        chunks_points = [p for p in chunks_points if p.payload.get("content_type") != "insights"]
        
        if not chunks_points:
            logger.info("Falling back to all chunks without data_source filter")
            chunks_points = get_points_by_file_id(file_id=request.file_id)
            chunks_points = [p for p in chunks_points if p.payload.get("content_type") != "insights"]
            
            if not chunks_points:
                raise HTTPException(status_code=404, detail=f"No chunks found for file_id={request.file_id}")
        
        logger.info(f"Found {len(chunks_points)} chunks")
        
        # Step 3: Concatenate page_content
        logger.info("Step 3: Concatenating page_content")
        chunks_text = [point.payload.get("page_content", "") for point in chunks_points]
        chunks_text = [text for text in chunks_text if text]
        
        if not chunks_text:
            raise HTTPException(status_code=400, detail="No valid page_content found")
        
        concatenated_text = "\n\n".join(chunks_text)
        total_tokens = count_tokens(concatenated_text)
        
        logger.info(f"Total tokens: {total_tokens}")
        
        # Step 4: Generate insights
        used_summarization = should_use_summarization(total_tokens)
        num_summary_chunks = None
        
        if used_summarization:
            logger.info("Using summarization approach")
            insights_text, num_summary_chunks = await generate_insights_with_summarization(
                concatenated_text,
                request.case_type,
                request.data_source
            )
        else:
            logger.info("Using direct approach")
            insights_text = await generate_insights_direct(
                concatenated_text,
                request.case_type,
                request.data_source
            )
        
        # Step 5: Generate embeddings and store
        logger.info("Step 5: Storing insights in Qdrant")
        dense_embedding = await generate_dense_embedding(
            insights_text,
            request.case_id,
            request.file_id
        )
        
        await store_insights_in_qdrant(
            insights=insights_text,
            file_id=request.file_id,
            case_id=request.case_id,
            sample_metadata=chunks_points[0].payload,
            case_type=request.case_type,
            data_source=request.data_source,
            dense_embedding=dense_embedding
        )
        
        logger.info("=" * 80)
        logger.info("Successfully generated and stored insights")
        logger.info("=" * 80)
        
        return InsightsResponse(
            case_id=request.case_id,
            file_id=request.file_id,
            case_type=request.case_type,
            data_source=request.data_source,
            insights=insights_text,
            source="generated",
            chunk_count=len(chunks_text),
            total_tokens=total_tokens,
            used_summarization=used_summarization,
            num_summary_chunks=num_summary_chunks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        from app.services.qdrant_service import qdrant_client
        collections = qdrant_client.get_collections()
        return {
            "status": "healthy",
            "qdrant": "connected",
            "collections": len(collections.collections)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }