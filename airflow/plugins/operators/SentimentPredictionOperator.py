import logging
from typing import Optional
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from plugins.scripts.sentiment_prediction import predict_tweet_sentiment

logger = logging.getLogger(__name__)

class SentimentPredictionOperator(BaseOperator):
    """
    Airflow Operator that fetches unpublished tweets from
    staging table, sends them to sentiment API, and inserts results into product table.
    """

    @apply_defaults
    def __init__(
        self,
        staging_table: str = "TWEET_STAGING",
        product_table: str = "TWEET_PRODUCT",
        batch_size: int = 32,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.staging_table = staging_table
        self.product_table = product_table
        self.batch_size = batch_size

    def execute(self, context):
        """Execute the sentiment prediction operator."""
        logger.info("Starting tweet sentiment prediction using API")

        try:
            num_processed = predict_tweet_sentiment(
                staging_table=self.staging_table,
                product_table=self.product_table,
                batch_size=self.batch_size
            )

            return {
                "records_processed": num_processed,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error in sentiment prediction operator: {str(e)}")
            raise