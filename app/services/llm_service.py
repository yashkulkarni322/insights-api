"""LLM service for text generation"""
import httpx
from typing import Optional
import logging
from fastapi import HTTPException

from app.config.settings import (
    LLM_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    EMBEDDING_URL,
    EMBEDDING_TIMEOUT
)
from app.config.prompt import SUMMARIZATION_PROMPT, MEGA_SUMMARY_PROMPT, PROMPT_TEMPLATE
from app.utils.token_utils import count_tokens

logger = logging.getLogger(__name__)


async def call_llm(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: float = LLM_TEMPERATURE
) -> str:
    """Call the LLM API with the given prompt"""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }
    
    if max_tokens:
        payload["max_tokens"] = max_tokens
    
    try:
        logger.info(f"Calling LLM API at {LLM_URL}")
        logger.info(f"Prompt length: {len(prompt)} characters, {count_tokens(prompt)} tokens")
        
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(LLM_URL, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            logger.info(f"LLM response received: {len(content)} characters, {count_tokens(content)} tokens")
            return content
            
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM API call failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error calling LLM: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


async def summarize_chunk(chunk: str, chunk_idx: int) -> str:
    """Summarize a single chunk"""
    logger.info(f"Summarizing chunk {chunk_idx + 1}")
    
    prompt = SUMMARIZATION_PROMPT.format(text=chunk)
    summary = await call_llm(prompt)
    
    logger.info(f"Chunk {chunk_idx + 1} summarized successfully")
    return summary


async def create_mega_summary(chunk_summaries: list, target_tokens: int) -> str:
    """Create a mega summary from all chunk summaries"""
    logger.info("Creating mega summary from all chunk summaries")
    
    combined_summaries = "\n\n".join([
        f"Summary {i+1}:\n{summary}"
        for i, summary in enumerate(chunk_summaries)
    ])
    
    prompt = MEGA_SUMMARY_PROMPT.format(
        target_tokens=target_tokens,
        summaries=combined_summaries
    )
    
    mega_summary = await call_llm(prompt, max_tokens=target_tokens + 500)
    mega_summary_tokens = count_tokens(mega_summary)
    
    logger.info(f"Mega summary created: {mega_summary_tokens} tokens")
    return mega_summary


async def generate_insights_from_text(text: str, case_type: str, data_source: str, max_tokens: int) -> str:
    """Generate insights from text using the prompt template"""
    logger.info("Generating insights from text")
    
    formatted_prompt = PROMPT_TEMPLATE.format(
        case_type=case_type,
        data_source=data_source
    )
    
    full_prompt = f"""{formatted_prompt}

Content to analyze:
{text}"""
    
    insights = await call_llm(full_prompt, max_tokens=max_tokens)
    insights_tokens = count_tokens(insights)
    
    logger.info(f"Insights generated: {insights_tokens} tokens")
    return insights


async def generate_dense_embedding(text: str, case_id: str, file_id: str) -> list:
    """Generate dense embeddings using custom embedding API"""
    async with httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT) as client:
        form_payload = {
            "text": text,
            "ingest": "false",
            "case_id": case_id,
            "file_id": file_id
        }
        try:
            response = await client.post(
                EMBEDDING_URL,
                data=form_payload,
                headers={"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as http_err:
            status = http_err.response.status_code if http_err.response is not None else None
            body = http_err.response.text if http_err.response is not None else ""
            logger.error(f"Embedding API error {status}: {body}")
            raise HTTPException(status_code=500, detail=f"Embedding API error: {body}")
        
        result = response.json()
        
        if result.get("status") != "success":
            raise HTTPException(
                status_code=500,
                detail="Embedding generation failed"
            )
        
        return result["embeddings"]