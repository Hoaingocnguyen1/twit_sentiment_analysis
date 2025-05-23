import os
import logging
from typing import Optional
from functools import lru_cache

# from scripts.registry import ModelRegistry

from mlflow.pyfunc import PyFuncModel
import mlflow.pyfunc
from pathlib import Path

MLFLOW_MODEL_NAME = os.getenv("MODEL_NAME")

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class ModelStore:
    def __init__(self):
        try:
            self.model = mlflow.pyfunc.load_model(
                f"models:/{MLFLOW_MODEL_NAME}@champion"
            )
            self._mlflow_client = mlflow.tracking.MlflowClient()
            self.version = self._mlflow_client.get_model_version_by_alias(
                MLFLOW_MODEL_NAME, "champion"
            ).version
        except:
            self.model = None
            self.version = None

    def predict(self, texts: list[str]):
        return self.model.predict(texts)

    def reload(self):
        try:
            self.model = mlflow.pyfunc.load_model(
                f"models:/{MLFLOW_MODEL_NAME}@champion"
            )
            self.version = self._mlflow_client.get_model_version_by_alias(
                MLFLOW_MODEL_NAME, "champion"
            ).version
            try:
                result = self.model.predict(["test"])
                logger.info(f"Reload test prediction: {result}")
            except Exception as e:
                logger.error("Failed reload test prediction", exc_info=e)
        except Exception as e:
            return {"status": "nope."}


model_store = ModelStore()


def get_model_store() -> ModelStore:
    return model_store
