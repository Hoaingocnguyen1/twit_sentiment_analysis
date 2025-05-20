import os
import sys
import argparse
import logging
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.dependencies import ModelStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure app/ is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Startup and shutdown event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("API startup: initializing resources")
    yield
    # Shutdown
    logger.info("API shutdown: cleaning up resources")


model_store = ModelStore()

# Create FastAPI app with lifespan manager
api = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis returning raw label indices",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with specific origins for better security
api.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Import router after app is created to avoid circular imports
from app.router import router
api.include_router(router, prefix="/api/v1")

@api.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

@api.post("/reload-model")
def reload_model():
    model_store.reload()
    return {"message": "Champion model reloaded."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, port=8000)
