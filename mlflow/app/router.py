from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
import logging

from app.dependencies import get_model_manager
from app.models import PredictInput, ErrorResponse, ModelInfo
from scripts.registry import ModelRegistry

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
    model_manager: ModelRegistry = Depends(get_model_manager)
) -> List[int]:
    """
    Predict sentiment indices for single or multiple texts.
    Returns a list of integers: 0=NEGATIVE, 1=NEUTRAL, 2=POSITIVE.
    If a single text was provided, the list will contain one element.
    """
    try:
        texts = input_data.texts  # populated via validator
        if not texts:  # Additional check for empty list
            raise HTTPException(status_code=400, detail="Empty text list provided")
            
        logger.debug(f"Received {len(texts)} texts for prediction")
        predictions = model_manager.predict_batch(texts)
        return predictions
    
    except Exception as e:
        logger.exception("Error in /predict endpoint")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get(
    "/model-info",
    response_model=ModelInfo,
    responses={
        200: {"description": "Model information retrieved successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_model_info(
    model_manager: ModelRegistry = Depends(get_model_manager)
) -> ModelInfo:
    """
    Get information about the current model.
    
    Args:
        model_manager: Model manager instance
        
    Returns:
        ModelInfo: Model information object
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        return ModelInfo(
            model_path=model_manager.model_path,
            model_type=type(model_manager.model).__name__,
            device=str(model_manager.device)
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )