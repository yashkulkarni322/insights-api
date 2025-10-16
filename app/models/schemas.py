from pydantic import BaseModel
from typing import Optional

class InsightsRequest(BaseModel):
    case_id: str
    file_id: str
    case_type: str
    data_source: str

class InsightsResponse(BaseModel):
    case_id: str
    file_id: str
    case_type: str
    data_source: str
    insights: str
    source: str
    chunk_count: Optional[int] = None
    total_tokens: Optional[int] = None
    used_map_reduce: Optional[bool] = False