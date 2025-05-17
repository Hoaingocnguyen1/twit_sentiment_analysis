# sentiment_prediction.py
import os
import sys
from pathlib import Path
import pandas as pd
import logging
import requests

logger = logging.getLogger(__name__)

def predict_tweet_sentiment(
    staging_table="TWEET_STAGING", 
    product_table="TWEET_PRODUCT",
    batch_size=32
):
    """
    Load model, retrieve tweets from staging, predict sentiment, insert into product.
    """
    try:
        # Step 1: Get unprocessed tweets
        db = get_db()
        rows = db.read(
            table_name=staging_table,
            columns=["tweet_id", "topic", "content", "created_at"],
            conditions={"moved_to_product": False}
        )

        if not rows:
            logger.info("No new records to process.")
            return 0

        logger.info(f"Processing {len(rows)} tweets")

        # Step 2: Prepare batch for prediction
        contents = [row[2] for row in rows]
        payload = {"texts": contents}
        response = requests.post("http://localhost:8000/api/v1/predict", json=payload)

        if response.status_code != 200:
            logger.error(f"Prediction API failed: {response.status_code} - {response.text}")
            return 0

        all_predictions = response.json()

        # Step 3: Prepare data to insert
        data_to_insert = [
            {
                "staging_tweet_id": row[0],
                "topic": row[1],
                "predicted_sentiment": pred,
                "created_at": row[3]
            }
            for row, pred in zip(rows, all_predictions)
        ]

        # Step 4: Insert results into product table
        db.batch_insert(
            table_name=product_table,
            rows=data_to_insert,
            column_mapping={
                "staging_tweet_id": "staging_tweet_id",
                "topic": "topic",
                "predicted_sentiment": "predicted_sentiment",
                "created_at": "created_at"
            }
        )

        # Step 5: Mark staging tweets as processed
        tweet_ids = [row[0] for row in rows]
        for tweet_id in tweet_ids:
            db.update(
                table_name=staging_table,
                data={"moved_to_product": True},
                conditions={"tweet_id": tweet_id}
            )

        logger.info(f"Successfully processed {len(rows)} tweets.")
        return len(rows)

    except Exception as e:
        logger.exception(f"Error during sentiment prediction: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    predict_tweet_sentiment()