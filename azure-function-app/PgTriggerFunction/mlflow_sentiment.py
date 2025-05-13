import os
import mlflow
import logging
from typing import Dict, Any, Optional
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with MLflow model."""
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the MLflow model from Azure Blob Storage."""
        try:
            # Configure MLflow
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            
            # Get model from registry
            model_name = os.getenv("MODEL_NAME", "sentiment-model")
            model_version = os.getenv("MODEL_VERSION", "latest")
            
            if model_version == "latest":
                client = mlflow.tracking.MlflowClient()
                model_version = client.get_latest_versions(model_name)[0].version
            
            # Load model
            model_uri = f"models:/{model_name}/{model_version}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Model loaded successfully: {model_name} version {model_version}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment label and score
        """
        try:
            if not text:
                return {"label": None, "score": None}
                
            # Get prediction
            prediction = self.model.predict([text])[0]
            
            return {
                "label": prediction["label"],
                "score": float(prediction["score"])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            raise
