from openai import OpenAI 
from .schemas import SentimentAnswer
import json
import time
from dotenv import load_dotenv
from typing import List
import pandas as pd
import os

load_dotenv()
LLM_API_KEY = os.getenv('GEMINI_API')

def label_data(cleaned_tweets: List[str] = None):
    if not cleaned_tweets:
        return
    
    results = []
    counter = 0

    client = OpenAI(api_key=LLM_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    for tweet in cleaned_tweets:
        completion = client.beta.chat.completions.parse(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "Extract the sentiment of the tweet"},
                {"role": "user", "content": tweet},
            ],
            response_format=SentimentAnswer,
            )
        sentiment = json.loads(completion.model_dump()['choices'][0]['message']['content'])['sentiment']
        results.append(sentiment)

        counter += 1
        if counter % 100 == 0:
            print(f"Processed {counter}/{len(cleaned_tweets)} tweets") # Change to logging.info
        
        time.sleep(4.05)
        
    assert len(results) == len(cleaned_tweets)

    df = pd.DataFrame({'tweet': cleaned_tweets, 'sentiment': results})
    
    return df
