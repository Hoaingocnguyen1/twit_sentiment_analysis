from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from typing_extensions import Annotated

class TextInput(BaseModel):
    """Input model for sentiment analysis"""
    text: Annotated[str, Field(description="Text to analyze for sentiment")]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "I really enjoyed this movie, it was fantastic!"
            }
        }
    }

class SentimentScore(BaseModel):
    """Model for individual sentiment scores"""
    label: Annotated[str, Field(description="Sentiment label")]
    score: Annotated[float, Field(description="Confidence score for the label")]

class SentimentResponse(BaseModel):
    """Response model for sentiment analysis"""
    text: Annotated[str, Field(description="Input text that was analyzed")]
    positive_score: Annotated[float, Field(description="Score for positive sentiment")]
    neutral_score: Annotated[float, Field(description="Score for neutral sentiment")]
    negative_score: Annotated[float, Field(description="Score for negative sentiment")]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "I really enjoyed this movie, it was fantastic!",
                "positive_score": 0.95,
                "neutral_score": 0.04,
                "negative_score": 0.01
            }
        }
    }

class ErrorResponse(BaseModel):
    """Error response model"""
    error: Annotated[str, Field(description="Error message")]
    detail: Annotated[Optional[str], Field(default=None, description="Detailed error information")]
