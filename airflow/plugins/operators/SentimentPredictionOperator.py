# sentiment_prediction_operator.py
import os
import logging
from typing import Dict, List, Optional, Any
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from pathlib import Path
from plugins.scripts.sentiment_prediction import predict_tweet_sentiment

logger = logging.getLogger(__name__)

class SentimentPredictionOperator(BaseOperator):
    """
    Airflow Operator that loads ModernBERT model, fetches unpublished tweets from
    staging table, predicts sentiment, and inserts results into product table.
    """
    
    @apply_defaults
    def __init__(
        self,
        model_name: str = 'modernbert',
        project_root_path: Optional[str] = None,
        staging_table: str = "TWEET_STAGING",
        product_table: str = "TWEET_PRODUCT",
        batch_size: int = 32,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.project_root_path = project_root_path
        self.staging_table = staging_table
        self.product_table = product_table
        self.batch_size = batch_size
        
    def execute(self, context):
        """Execute the sentiment prediction operator."""
        import sys

        if self.project_root_path:
            project_root = Path(self.project_root_path)
            if str(project_root) not in sys.path:
                sys.path.append(str(project_root))
        
        # Import the prediction script
        
        logger.info("Starting tweet sentiment prediction")
        
        # Call the prediction function with parameters from the operator
        try:
            num_processed = predict_tweet_sentiment(
                model_name=self.model_name,
                project_root_path=self.project_root_path,
                staging_table=self.staging_table,
                product_table=self.product_table,
                batch_size=self.batch_size
            )
            
            # Return the result for logging/monitoring
            return {
                "records_processed": num_processed,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment prediction operator: {str(e)}")
            raise