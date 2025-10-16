from typing import List, Dict, Any
from fastapi import HTTPException
from app.services.qdrant_service import QdrantService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.core.config import settings
from app.utils.logger import setup_logger
from app.utils.tokenizer import count_tokens

logger = setup_logger(__name__)

class InsightsService:
    def __init__(self):
        self.qdrant = QdrantService()
        self.embedding = EmbeddingService()
        self.llm = LLMService()
    
    async def generate_insights_with_llm(
        self,
        chunks: List[str],
        case_type: str,
        data_source: str
    ) -> tuple[str, bool]:
        """Generate insights with automatic map-reduce for large datasets"""
        concatenated_data = "\n\n".join(chunks)
        total_tokens = count_tokens(concatenated_data)
        
        logger.info(f"Total tokens in concatenated chunks: {total_tokens}")
        
        if total_tokens > settings.MAX_TOKENS_BEFORE_MAP_REDUCE:
            logger.info(f"Using map-reduce (tokens: {total_tokens})")
            insights = await self.llm.generate_insights_map_reduce(
                chunks, case_type, data_source
            )
            return insights, True
        else:
            logger.info(f"Using direct approach (tokens: {total_tokens})")
            insights = await self.llm.generate_insights_direct(
                chunks, case_type, data_source
            )
            return insights, False
    
    async def store_insights_in_qdrant(
        self,
        insights: str,
        file_id: str,
        case_id: str,
        sample_metadata: Dict[str, Any],
        case_type: str,
        data_source: str
    ) -> str:
        """Store generated insights in Qdrant"""
        dense_embedding = await self.embedding.generate_dense_embedding(
            insights, case_id, file_id
        )
        sparse_embedding = self.embedding.generate_sparse_embedding(insights)
        
        payload = {
            "page_content": insights,
            "file_id": file_id,
            "case_id": case_id,
            "content_type": "insights",
            "case_type": case_type,
            "data_source": data_source
        }
        
        for key, value in sample_metadata.items():
            if key not in ["page_content", "file_id", "case_id", "content_type"]:
                payload[key] = value
        
        point_id = self.qdrant.store_point(payload, dense_embedding, sparse_embedding)
        logger.info(f"Stored insights with point_id={point_id}")
        return point_id
    
    async def get_or_generate_insights(
        self,
        case_id: str,
        file_id: str,
        case_type: str,
        data_source: str
    ) -> Dict[str, Any]:
        """Main orchestration method"""
        # Check for existing insights
        existing_insights = self.qdrant.get_points_by_file_id(
            file_id=file_id,
            content_type="insights",
            case_type=case_type,
            data_source=data_source
        )
        
        if existing_insights:
            logger.info(f"Found existing insights for file_id={file_id}")
            return {
                "insights": existing_insights[0].payload.get("page_content", ""),
                "source": "existing",
                "chunk_count": None,
                "total_tokens": None,
                "used_map_reduce": False
            }
        
        # Retrieve chunks
        logger.info(f"Retrieving chunks for file_id={file_id}")
        chunks_points = self.qdrant.get_points_by_file_id(
            file_id=file_id,
            content_type=None,
            case_type=None,
            data_source=data_source
        )
        
        chunks_points = [p for p in chunks_points 
                        if p.payload.get("content_type") != "insights"]
        
        if not chunks_points:
            chunks_points = self.qdrant.get_points_by_file_id(
                file_id=file_id,
                content_type=None,
                case_type=None,
                data_source=None
            )
            chunks_points = [p for p in chunks_points 
                           if p.payload.get("content_type") != "insights"]
            
            if not chunks_points:
                raise HTTPException(
                    status_code=404,
                    detail=f"No chunks found for file_id={file_id}"
                )
        
        logger.info(f"Found {len(chunks_points)} chunks")
        
        # Extract text
        chunks_text = [p.payload.get("page_content", "") for p in chunks_points]
        chunks_text = [text for text in chunks_text if text]
        
        if not chunks_text:
            raise HTTPException(
                status_code=400,
                detail="No valid page_content found in chunks"
            )
        
        # Count tokens
        total_tokens = count_tokens("\n\n".join(chunks_text))
        
        # Generate insights
        insights_text, used_map_reduce = await self.generate_insights_with_llm(
            chunks_text, case_type, data_source
        )
        
        # Store insights
        await self.store_insights_in_qdrant(
            insights=insights_text,
            file_id=file_id,
            case_id=case_id,
            sample_metadata=chunks_points[0].payload,
            case_type=case_type,
            data_source=data_source
        )
        
        return {
            "insights": insights_text,
            "source": "generated",
            "chunk_count": len(chunks_text),
            "total_tokens": total_tokens,
            "used_map_reduce": used_map_reduce
        }