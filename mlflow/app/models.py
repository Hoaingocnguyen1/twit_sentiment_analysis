# from pydantic import BaseModel, Field
# from typing import List, Dict, Any, Optional
# from typing_extensions import Annotated

# class TextInput(BaseModel):
#     """Input model for sentiment analysis"""
#     text: Annotated[str, Field(description="Text to analyze for sentiment")]
    
#     model_config = {
#         "json_schema_extra": {
#             "example": {
#                 "text": "I really enjoyed this movie, it was fantastic!"
#             }
#         }
#     }

# class BatchTextInput(BaseModel):
#     """Input model for batch sentiment analysis"""
#     texts: Annotated[List[str], Field(description="List of texts to analyze for sentiment")]
    
#     model_config = {
#         "json_schema_extra": {
#             "example": {
#                 "texts": [
#                     "I really enjoyed this movie, it was fantastic!",
#                     "The service was terrible and the food was cold.",
#                     "The weather is nice today."
#                 ]
#             }
#         }
#     }

# class SentimentScore(BaseModel):
#     """Model for individual sentiment scores"""
#     label: Annotated[str, Field(description="Sentiment label")]
#     score: Annotated[float, Field(description="Confidence score for the label")]

# class SentimentResponse(BaseModel):
#     """Response model for sentiment analysis"""
#     text: Annotated[str, Field(description="Input text that was analyzed")]
#     sentiment: Annotated[str, Field(description="Sentiment classification result (POSITIVE, NEUTRAL, NEGATIVE)")]
#     positive_score: Annotated[float, Field(description="Score for positive sentiment")]
#     neutral_score: Annotated[float, Field(description="Score for neutral sentiment")]
#     negative_score: Annotated[float, Field(description="Score for negative sentiment")]
    
#     model_config = {
#         "json_schema_extra": {
#             "example": {
#                 "text": "I really enjoyed this movie, it was fantastic!",
#                 "sentiment": "POSITIVE",
#                 "positive_score": 1.0,
#                 "neutral_score": 0.0,
#                 "negative_score": 0.0
#             }
#         }
#     }

# class BatchSentimentResponse(BaseModel):
#     """Response model for batch sentiment analysis"""
#     results: Annotated[List[SentimentResponse], Field(description="List of sentiment analysis results")]

# class ErrorResponse(BaseModel):
#     """Error response model"""
#     error: Annotated[str, Field(description="Error message")]
#     detail: Annotated[Optional[str], Field(default=None, description="Detailed error information")]

# from pydantic import BaseModel, Field, root_validator
# from typing import List, Optional, Union
# from typing_extensions import Annotated

# class PredictInput(BaseModel):
#     """
#     Accept either a single text string or a list of texts.
#     """
#     text: Optional[Annotated[str, Field(None, description="Single text to analyze")]]
#     texts: Optional[Annotated[List[str], Field(None, description="List of texts to analyze")]]

#     @root_validator(pre=True)
#     def ensure_texts(cls, values):
#         text = values.get('text')
#         texts = values.get('texts')
#         if text and texts:
#             raise ValueError("Provide only one of 'text' or 'texts'.")
#         if not text and not texts:
#             raise ValueError("One of 'text' or 'texts' must be provided.")
#         if text:
#             values['texts'] = [text]
#         return values

# class ErrorResponse(BaseModel):
#     error: str = Field(..., description="Error message")
#     detail: Optional[str] = Field(None, description="Additional details")

from pydantic import BaseModel, Field, root_validator
from typing import List, Optional, Union
from typing_extensions import Annotated

class PredictInput(BaseModel):
    """
    Accept either a single text string or a list of texts.
    """
    text: Optional[Annotated[str, Field(None, description="Single text to analyze")]] = None
    texts: Optional[Annotated[List[str], Field(None, description="List of texts to analyze")]] = None
    
    @root_validator(pre=True)
    def ensure_texts(cls, values):
        text = values.get('text')
        texts = values.get('texts')
        
        if text and texts:
            raise ValueError("Provide only one of 'text' or 'texts'.")
        if not text and not texts:
            raise ValueError("One of 'text' or 'texts' must be provided.")
        if text:
            values['texts'] = [text]
        return values

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional details")
    
class ModelInfo(BaseModel):
    """Model for standardized model information response"""
    model_path: str
    model_type: str
    device: str
    version: Optional[str] = None