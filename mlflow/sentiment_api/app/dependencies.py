from mlflow.scripts.registry import ModelRegistry
from typing import Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_VERSION = os.getenv("MODEL_VERSION")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

logger = logging.getLogger(__name__)

# Global model manager instance
model_manager: Optional[ModelRegistry] = None

def get_model_manager() -> ModelRegistry:
    global model_manager
    if model_manager is None:
        try:
            registry = ModelRegistry()
            # Load model based on version or stage
            if MODEL_VERSION:
                manager = registry.load_model(
                    model_name=MODEL_NAME,
                    version=MODEL_VERSION
                )
            else:
                manager = registry.load_model(
                    model_name=MODEL_NAME,
                    stage=MODEL_STAGE
                )
                # Retrieve and log latest version
                latest = registry.get_latest_version(MODEL_NAME)
                logger.info(f"Auto-chosen latest version: {latest}")
            
            model_manager = manager
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise RuntimeError(f"Failed to initialize model: {e}")
    return model_manager