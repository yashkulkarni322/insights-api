"""FastAPI application entry point"""
from fastapi import FastAPI
import logging

from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('insights_api.log'),
        logging.StreamHandler()
    ]
)

app = FastAPI(title="Insights API", version="3.0.0")

# Include API routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)