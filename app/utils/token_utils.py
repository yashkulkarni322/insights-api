"""Utility functions for token counting and text splitting"""
import tiktoken
from typing import List
import logging

logger = logging.getLogger(__name__)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    return len(tokenizer.encode(text))


def split_into_token_chunks(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks of specified token size"""
    logger.info(f"Splitting text into chunks of {chunk_size} tokens")
    
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    chunks = []
    
    for i in range(0, total_tokens, chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        logger.info(f"Created chunk {len(chunks)}: {len(chunk_tokens)} tokens")
    
    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks