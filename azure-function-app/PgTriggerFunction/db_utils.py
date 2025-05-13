import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a database connection."""
    try:
        conn = psycopg2.connect(
            os.getenv("POSTGRES_CONNECTION_STRING"),
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def update_sentiment(tweet_id: str, sentiment: Dict[str, Any]) -> None:
    """
    Update tweet sentiment in the database.
    
    Args:
        tweet_id: ID of the tweet
        sentiment: Dictionary containing sentiment analysis results
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE tweets 
                SET sentiment_label = %s,
                    sentiment_score = %s,
                    processed_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (
                sentiment['label'],
                sentiment['score'],
                tweet_id
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Error updating sentiment: {str(e)}")
        raise
    finally:
        conn.close()

def get_unprocessed_tweets() -> List[Dict[str, Any]]:
    """Get tweets that haven't been processed for sentiment."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, text 
                FROM tweets 
                WHERE sentiment_label IS NULL 
                AND text IS NOT NULL
                LIMIT 100
            """)
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Error fetching unprocessed tweets: {str(e)}")
        raise
    finally:
        conn.close()
