import logging
import azure.functions as func
from .db_utils import update_sentiment, get_unprocessed_tweets
from .mlflow_sentiment import SentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer()

def main(triggerInput: func.DocumentList) -> None:
    """
    Azure Function triggered by PostgreSQL changes.
    
    Args:
        triggerInput: List of changed documents from PostgreSQL
    """
    try:
        # Process each changed document
        for doc in triggerInput:
            tweet_id = doc['id']
            text = doc['text']
            
            # Skip if no text
            if not text:
                continue
                
            # Analyze sentiment
            sentiment = analyzer.analyze(text)
            
            # Update database
            update_sentiment(tweet_id, sentiment)
            
            logger.info(f"Processed tweet {tweet_id} with sentiment {sentiment}")
            
    except Exception as e:
        logger.error(f"Error processing trigger: {str(e)}")
        raise 