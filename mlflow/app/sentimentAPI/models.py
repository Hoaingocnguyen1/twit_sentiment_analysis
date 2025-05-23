from pydantic import BaseModel, Field, root_validator
from typing import List, Optional, Union
from typing_extensions import Annotated


class PredictInput(BaseModel):
    """
    Accept either a single text string or a list of texts.
    """

    text: Optional[
        Annotated[str, Field(None, description="Single text to analyze")]
    ] = None
    texts: Optional[
        Annotated[List[str], Field(None, description="List of texts to analyze")]
    ] = None

    @root_validator(pre=True)
    def ensure_texts(cls, values):
        text = values.get("text")
        texts = values.get("texts")

        if text and texts:
            raise ValueError("Provide only one of 'text' or 'texts'.")
        if not text and not texts:
            raise ValueError("One of 'text' or 'texts' must be provided.")
        if text:
            values["texts"] = [text]
        return values


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional details")
