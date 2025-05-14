import uvicorn
import argparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import your modules
from app.router import router
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using MLflow models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    """Tasks to run on application startup"""
    logger.info("Initializing application resources...")
    # Additional startup tasks if needed

@app.on_event("shutdown")
async def shutdown_event():
    """Tasks to run on application shutdown"""
    logger.info("Shutting down application...")
    # Release resources, close connections, etc.

def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment Analysis API")
    parser.add_argument("--model-name", type=str, required=True, help="MLflow model name")
    parser.add_argument("--version", type=str, default=None, help="Model version")
    parser.add_argument("--stage", type=str, default="Production", help="Model stage")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set environment variables for the model manager
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["MODEL_VERSION"] = args.version if args.version else ""
    os.environ["MODEL_STAGE"] = args.stage
    
    logger.info(f"Starting Sentiment Analysis API with model {args.model_name}...")
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=True,
        log_level="info"
    )