
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
import logging

from app.dependencies import get_champion_model
from app.models import PredictInput, ErrorResponse, ModelInfo
#from scripts.registry import ModelRegistry
from mlflow.pyfunc import PyFuncModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["sentiment"])

@router.post(
    "/predict",
    response_model=List[int],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict_sentiment(
    input_data: PredictInput,
    model: PyFuncModel = Depends(get_champion_model)
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
    
