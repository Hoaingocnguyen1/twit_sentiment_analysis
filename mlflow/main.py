# import os
# import sys
# import argparse
# import logging
# import uvicorn
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# # ensure app/ is on path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

# from app.router import router

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # Create FastAPI app
# app = FastAPI(
#     title="Sentiment Analysis API",
#     description="API for sentiment analysis returning raw label indices",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(router, prefix="/api/v1")

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}

# @app.on_event("startup")
# async def on_startup():
#     logger.info("API startup: initial checks complete")

# @app.on_event("shutdown")
# async def on_shutdown():
#     logger.info("API shutdown: cleaning up resources")


# def parse_args():
#     parser = argparse.ArgumentParser(description="Sentiment Analysis API")
#     parser.add_argument("--model-name", type=str, required=True, help="MLflow model name")
#     parser.add_argument("--version", type=str, default=None, help="Model version")
#     parser.add_argument("--stage", type=str, default="Production", help="Model stage")
#     parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
#     parser.add_argument("--port", type=int, default=8000, help="Port to bind")
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_args()
#     os.environ["MODEL_NAME"] = args.model_name
#     os.environ["MODEL_VERSION"] = args.version or ""
#     os.environ["MODEL_STAGE"] = args.stage

#     logger.info(f"Starting API with model={args.model_name} version={args.version or args.stage}")
#     uvicorn.run(
#         "main:app",
#         host=args.host,
#         port=args.port,
#         reload=True,
#         log_level="info"
#     )

import os
import sys
import argparse
import logging
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis returning raw label indices",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware with specific origins for better security
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Import router after app is created to avoid circular imports
from app.router import router
app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment Analysis API")
    parser.add_argument("--model-name", type=str, required=True, help="MLflow model name")
    parser.add_argument("--version", type=str, default=None, help="Model version")
    parser.add_argument("--stage", type=str, default="Production", help="Model stage")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--log-level", type=str, default="info", 
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set environment variables
    os.environ["MODEL_NAME"] = args.model_name
    if args.version:
        os.environ["MODEL_VERSION"] = args.version
    os.environ["MODEL_STAGE"] = args.stage
    
    logger.info(f"Starting API with model={args.model_name} version={args.version or args.stage}")
    
    # Use Uvicorn to run the app
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=False,  # Disable reload in production for better performance
        log_level=args.log_level
    )