"""Pydantic models for request/response validation"""
from pydantic import BaseModel
from typing import Optional
from enum import Enum


class CaseType(str, Enum):
    DRUG_TRAFFICKING = "Drug Trafficking and Substance Abuse"
    ARMS_TRAFFICKING = "Arms Trafficking"
    CYBER_CRIME = "Cyber Crime"
    TERRORISM = "Terrorism"
    MURDER = "Murder and Homicide"
    SUICIDE = "Suicide"
    GENERAL = "General"


class DataSource(str, Enum):
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    UFED = "ufed_extraction"
    OTHERS = "others"


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
    source: str  # "existing" or "generated"
    chunk_count: Optional[int] = None
    total_tokens: Optional[int] = None
    used_summarization: Optional[bool] = False
    num_summary_chunks: Optional[int] = None