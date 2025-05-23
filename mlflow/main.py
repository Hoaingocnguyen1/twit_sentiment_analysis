import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from app.dependencies import model_store
from app.models import PredictInput, ErrorResponse
from mlflow.pyfunc import PyFuncModel
from typing import List
import time
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry


BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

load_dotenv(f'{BASE_DIR}/.env')

MLFLOW_MODEL_NAME = os.getenv('MODEL_NAME')

REGISTRY = CollectorRegistry()

LABEL_MAP = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

PREDICT_COUNTER = Counter(
    "predict_requests_total",
    "Total prediction requests",
    ['model_version', 'outcome', 'predicted_label'],
    registry=REGISTRY
)
PREDICT_DURATION = Histogram(
    "predict_duration_seconds",
    "Prediction duration",
    ['model_version'],
    registry=REGISTRY
)
MODEL_INFO = Gauge(
    "model_serving_info",
    "Indicates model version/URI being served (1 if active)",
    ['version_uri'],
    registry=REGISTRY
)
RELOAD_COUNTER = Counter(
    "reload_model_requests_total",
    "Total model reload requests",
    ['outcome'],
    registry=REGISTRY
)
RELOAD_DURATION = Histogram(
    "reload_model_duration_seconds",
    "Model reload duration",
    registry=REGISTRY
)
# Gauge cho health status
HEALTH_STATUS = Gauge(
    "service_health_status",
    "Health status of the service (1 for OK, 0 for error)",
    registry=REGISTRY
)

ITEM_COUNT = Counter("list_item_count", 
        "Number of items in the sentiment_list",
        registry=REGISTRY
)

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
@api.get("/health")
async def health_check():
    is_ok = model_store.model is not None
    HEALTH_STATUS.set(1 if is_ok else 0)
    MODEL_INFO.labels(version_uri=str(model_store.version)).set(1)
    return {"status": "ok" if is_ok else "model_error", "health": is_ok}

@api.post("/reload-model")
async def reload_model():
    outcome = "success"
    start_time = time.time()
    try:
        model_store.reload()
        if model_store.model is None:
            return {"message": "Champion model not reloaded."}
        return {"message": "Champion model reloaded."}
    except Exception as e:
        HEALTH_STATUS.set(0)
        outcome = "error"
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        RELOAD_COUNTER.labels(outcome).inc()
        RELOAD_DURATION.observe(time.time() - start_time)

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
    input_data: PredictInput
) -> List[int]:
    """
    Predict sentiment indices for single or multiple texts.
    Returns a list of integers: 0=NEGATIVE, 1=NEUTRAL, 2=POSITIVE.
    If a single text was provided, the list will contain one element.
    """
    if model_store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.time()
    outcome = "success"
    try:
        texts = input_data.texts  # populated via validator
        if not texts:
            raise HTTPException(status_code=400, detail="Empty text list provided")
        
        ITEM_COUNT.inc(len(texts))
            
        logger.debug(f"Received {len(texts)} texts for prediction")

        predictions = model_store.predict(texts)
        return predictions
    
    except Exception as e:
        logger.exception("Error in /predict endpoint")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        dur = time.time() - start
        PREDICT_DURATION.labels(model_version=model_store.version).observe(dur)
        if predictions:
            for pred in predictions:
                PREDICT_COUNTER.labels(
                model_version=model_store.version,
                outcome=outcome,
                predicted_label=LABEL_MAP[pred] if outcome=="success" else "error"
            ).inc()
                

@api.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
