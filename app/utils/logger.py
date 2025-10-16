import logging
from app.core.config import settings

def setup_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=settings.LOG_LEVEL)
    return logging.getLogger(name)