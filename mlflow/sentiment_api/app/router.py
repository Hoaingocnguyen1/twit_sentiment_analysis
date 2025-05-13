from fastapi import APIRouter, Depends, HTTPException
from app.dependencies import get_model_manager
from app.models import TextInput, SentimentResponse, ErrorResponse
from scripts.registry import ModelRegistry
import logging

logger = logging.getLogger(__name__)
router = APIRouter(tags=["sentiment"])

@router.post(
    "/predict",
    response_model=SentimentResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict_sentiment(
    input_data: TextInput,
    model_manager: ModelRegistry = Depends(get_model_manager)
) -> SentimentResponse:
    """
    Predict sentiment for the given text using ModernBERT model.
    Args:
        input_data: Text to analyze
        model_manager: Model manager instance
    
    Returns:
        SentimentResponse: Prediction results with positive, neutral, and negative scores
    
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Get prediction
        result = model_manager.predict(input_data.text)
        
        # Format response with specific scores
        return SentimentResponse(
            text=input_data.text,
            positive_score=result.get("positive", 0.0),
            neutral_score=result.get("neutral", 0.0),
            negative_score=result.get("negative", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Error predicting sentiment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error predicting sentiment: {str(e)}"
        )

@router.get(
    "/model-info",
    response_model=dict,
    responses={
        200: {"description": "Model information retrieved successfully"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_model_info(
    model_manager: ModelRegistry = Depends(get_model_manager)
) -> dict:
    """
    Get information about the current model.
    Args:
        model_manager: Model manager instance
    
    Returns:
        dict: Model information
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        return {
            "model_path": model_manager.model_path,
            "model_type": type(model_manager.model).__name__,
            "device": str(model_manager.device)
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )