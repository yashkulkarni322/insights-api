from qdrant_client import QdrantClient #type:ignore
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct #type:ignore
from typing import Optional, List, Any, Dict
import uuid
from app.core.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.collection_name = settings.COLLECTION_NAME
    
    def get_points_by_file_id(
        self,
        file_id: str,
        content_type: Optional[str] = None,
        case_type: Optional[str] = None,
        data_source: Optional[str] = None
    ) -> List[Any]:
        """Retrieve points from Qdrant by file_id and optional filters"""
        filter_conditions = [
            FieldCondition(key="file_id", match=MatchValue(value=file_id))
        ]
        
        if content_type:
            filter_conditions.append(
                FieldCondition(key="content_type", match=MatchValue(value=content_type))
            )
        if case_type:
            filter_conditions.append(
                FieldCondition(key="case_type", match=MatchValue(value=case_type))
            )
        if data_source:
            filter_conditions.append(
                FieldCondition(key="data_source", match=MatchValue(value=data_source))
            )
        
        payload_filter = Filter(must=filter_conditions)
        
        points = []
        offset = None
        limit = 100
        
        while True:
            points_batch, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=payload_filter,
                offset=offset,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            points.extend(points_batch)
            
            if next_offset is None or len(points_batch) < limit:
                break
            offset = next_offset
        
        logger.info(f"Retrieved {len(points)} points for file_id={file_id}")
        return points
    
    def store_point(
        self,
        payload: Dict[str, Any],
        dense_embedding: List[float],
        sparse_embedding: Dict[str, List]
    ) -> str:
        """Store a point in Qdrant"""
        point_id = str(uuid.uuid4())
        
        point = PointStruct(
            id=point_id,
            vector={
                "dense-embed": dense_embedding,
                "sparse-embed": {
                    "indices": sparse_embedding["indices"],
                    "values": sparse_embedding["values"]
                }
            },
            payload=payload
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        logger.info(f"Stored point with id={point_id}")
        return point_id