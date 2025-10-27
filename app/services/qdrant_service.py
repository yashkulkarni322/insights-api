"""Qdrant database service for vector operations"""
from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import (  # type: ignore
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct
)
from fastembed import SparseTextEmbedding  # type: ignore
from typing import List, Dict, Any, Optional
import uuid
import logging

from app.config.settings import QDRANT_URL, COLLECTION_NAME

logger = logging.getLogger(__name__)

# Initialize clients
qdrant_client = QdrantClient(url=QDRANT_URL)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def get_points_by_file_id(
    file_id: str,
    collection_name: str = COLLECTION_NAME,
    content_type: Optional[str] = None,
    case_type: Optional[str] = None,
    data_source: Optional[str] = None
) -> List[Any]:
    """Retrieve points from Qdrant by file_id and optional filters"""
    filter_conditions = [
        FieldCondition(
            key="file_id",
            match=MatchValue(value=file_id)
        )
    ]
    
    if content_type:
        filter_conditions.append(
            FieldCondition(
                key="content_type",
                match=MatchValue(value=content_type)
            )
        )
    
    if case_type:
        filter_conditions.append(
            FieldCondition(
                key="case_type",
                match=MatchValue(value=case_type)
            )
        )
    
    if data_source:
        filter_conditions.append(
            FieldCondition(
                key="data_source",
                match=MatchValue(value=data_source)
            )
        )
    
    payload_filter = Filter(must=filter_conditions)
    
    points = []
    offset = None
    limit = 100
    
    while True:
        points_batch, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
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
    
    logger.info(f"Retrieved {len(points)} points for file_id={file_id}, content_type={content_type}")
    return points


def generate_sparse_embedding(text: str) -> Dict[str, List]:
    """Generate sparse embeddings using FastEmbed BM25"""
    embeddings = list(sparse_model.embed([text]))
    if embeddings:
        return {
            "indices": embeddings[0].indices.tolist(),
            "values": embeddings[0].values.tolist()
        }
    return {"indices": [], "values": []}


async def store_insights_in_qdrant(
    insights: str,
    file_id: str,
    case_id: str,
    sample_metadata: Dict[str, Any],
    case_type: str,
    data_source: str,
    dense_embedding: List[float]
) -> str:
    """Store generated insights in Qdrant with embeddings"""
    logger.info(f"Storing insights in Qdrant for file_id={file_id}")
    
    # Generate sparse embedding
    sparse_embedding = generate_sparse_embedding(insights)
    
    # Create payload
    payload = {
        "page_content": insights,
        "file_id": file_id,
        "case_id": case_id,
        "content_type": "insights",
        "case_type": case_type,
        "data_source": data_source
    }
    
    # Copy other metadata fields
    for key, value in sample_metadata.items():
        if key not in ["page_content", "file_id", "case_id", "content_type"]:
            payload[key] = value
    
    # Generate unique point ID
    point_id = str(uuid.uuid4())
    
    # Create point with vectors
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
    
    # Upsert point to Qdrant
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )
    
    logger.info(f"Stored insights with point_id={point_id}")
    return point_id