

import os
import sys
import argparse
import logging
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, APIRouter, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.dependencies import model_store, get_model_store
from app.models import PredictInput, ErrorResponse, ModelInfo
from mlflow.pyfunc import PyFuncModel
from typing import List, Dict, Any

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
async def reload_model():
    model_store.reload()
    return {"message": "Champion model reloaded."}

@api.post(
    "/api/v1/predict",
    response_model=List[int],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict_sentiment(
    input_data: PredictInput,
    model: PyFuncModel = Depends(get_model_store)
) -> List[int]:
    """
    Predict sentiment indices for single or multiple texts.
    Returns a list of integers: 0=NEGATIVE, 1=NEUTRAL, 2=POSITIVE.
    If a single text was provided, the list will contain one element.
    """
    try:
        texts = input_data.texts  # populated via validator
        if not texts:
            raise HTTPException(status_code=400, detail="Empty text list provided")
            
        logger.debug(f"Received {len(texts)} texts for prediction")
        predictions = model.predict(texts)
        return predictions
    
    except Exception as e:
        logger.exception("Error in /predict endpoint")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, port=8000)
