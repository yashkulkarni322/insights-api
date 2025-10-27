"""Main insights generation service"""
import logging
from typing import Tuple

from app.config.settings import (
    MAX_TOKENS_BEFORE_SUMMARIZATION,
    CHUNK_SIZE_FOR_SUMMARIZATION,
    MEGA_SUMMARY_TARGET,
    INSIGHTS_MAX_TOKENS
)
from app.utils.token_utils import count_tokens, split_into_token_chunks
from app.services.llm_service import (
    summarize_chunk,
    create_mega_summary,
    generate_insights_from_text
)

logger = logging.getLogger(__name__)


async def generate_insights_with_summarization(
    concatenated_text: str,
    case_type: str,
    data_source: str
) -> Tuple[str, int]:
    """
    Generate insights using summarization approach for large datasets
    
    Returns:
        tuple: (insights_text, num_summary_chunks)
    """
    logger.info("Using summarization approach for large dataset")
    
    # Step 1: Split into 50k token chunks
    token_chunks = split_into_token_chunks(concatenated_text, CHUNK_SIZE_FOR_SUMMARIZATION)
    
    # Step 2: Summarize each chunk
    chunk_summaries = []
    for idx, chunk in enumerate(token_chunks):
        summary = await summarize_chunk(chunk, idx)
        chunk_summaries.append(summary)
    
    # Step 3: Create mega summary
    mega_summary = await create_mega_summary(chunk_summaries, MEGA_SUMMARY_TARGET)
    
    # Step 4: Generate insights from mega summary
    insights = await generate_insights_from_text(
        mega_summary,
        case_type,
        data_source,
        INSIGHTS_MAX_TOKENS
    )
    
    return insights, len(token_chunks)


async def generate_insights_direct(
    concatenated_text: str,
    case_type: str,
    data_source: str
) -> str:
    """Generate insights directly without summarization for smaller datasets"""
    logger.info("Using direct approach for dataset")
    
    insights = await generate_insights_from_text(
        concatenated_text,
        case_type,
        data_source,
        INSIGHTS_MAX_TOKENS
    )
    
    return insights


def should_use_summarization(total_tokens: int) -> bool:
    """Determine if summarization approach should be used"""
    return total_tokens > MAX_TOKENS_BEFORE_SUMMARIZATION