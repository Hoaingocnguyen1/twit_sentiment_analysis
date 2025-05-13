from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

class TwitterQuery(BaseModel):
    query: str
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    max_results: Optional[int] = Field(default=20, ge=1, le=20)
    mode: Literal['Latest', 'Top'] = 'Latest'

class SentimentAnswer(BaseModel):
    str: str
    sentiment: Literal['POSITIVE', 'NEGATIVE', 'NEUTRAL']