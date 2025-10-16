from fastembed import SparseTextEmbedding #type:ignore
import httpx
from typing import List, Dict
from app.core.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingService:
    def __init__(self):
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    async def generate_dense_embedding(
        self,
        text: str,
        case_id: str,
        file_id: str
    ) -> List[float]:
        """Generate dense embeddings using custom embedding API"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            form_payload = {
                "text": text,
                "ingest": "false",
                "case_id": case_id,
                "file_id": file_id
            }
            response = await client.post(
                settings.EMBEDDING_URL,
                data=form_payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded"
                }
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") != "success":
                raise Exception("Embedding generation failed")
            
            return result["embeddings"]
    
    def generate_sparse_embedding(self, text: str) -> Dict[str, List]:
        """Generate sparse embeddings using FastEmbed BM25"""
        embeddings = list(self.sparse_model.embed([text]))
        if embeddings:
            return {
                "indices": embeddings[0].indices.tolist(),
                "values": embeddings[0].values.tolist()
            }
        return {"indices": [], "values": []}