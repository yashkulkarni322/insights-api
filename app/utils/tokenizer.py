import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    return len(tokenizer.encode(text))