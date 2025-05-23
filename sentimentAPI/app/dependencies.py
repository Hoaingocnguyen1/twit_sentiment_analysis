# import os
# import logging
# from typing import Optional
# from functools import lru_cache

# # from scripts.registry import ModelRegistry

# from mlflow.pyfunc import PyFuncModel
# import mlflow.pyfunc
# from pathlib import Path

# MLFLOW_MODEL_NAME = os.getenv("MODEL_NAME")

# logger = logging.getLogger(__name__)

# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# class ModelStore:
#     def __init__(self):
#         try:
#             self.model = mlflow.pyfunc.load_model(
#                 f"models:/{MLFLOW_MODEL_NAME}@champion"
#             )
#             self._mlflow_client = mlflow.tracking.MlflowClient()
#             self.version = self._mlflow_client.get_model_version_by_alias(
#                 MLFLOW_MODEL_NAME, "champion"
#             ).version
#         except:
#             self.model = None
#             self.version = None

#     def predict(self, texts: list[str]):
#         return self.model.predict(texts)

#     def reload(self):
#         try:
#             self.model = mlflow.pyfunc.load_model(
#                 f"models:/{MLFLOW_MODEL_NAME}@champion"
#             )
#             self.version = self._mlflow_client.get_model_version_by_alias(
#                 MLFLOW_MODEL_NAME, "champion"
#             ).version
#             try:
#                 result = self.model.predict(["test"])
#                 logger.info(f"Reload test prediction: {result}")
#             except Exception as e:
#                 logger.error("Failed reload test prediction", exc_info=e)
#         except Exception as e:
#             return {"status": "nope."}


# model_store = ModelStore()


# def get_model_store() -> ModelStore:
#     return model_store

import os
import logging
import mlflow
from mlflow.transformers import load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MLFLOW_MODEL_NAME = os.getenv("MODEL_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not MLFLOW_MODEL_NAME or not MLFLOW_TRACKING_URI:
    raise EnvironmentError("Both MODEL_NAME and MLFLOW_TRACKING_URI must be set")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class TransformerModelStore:
    def __init__(self):
        self.model = None
        self.version = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading model: models:/{MLFLOW_MODEL_NAME}@champion")
            self.model = load_model(f"models:/{MLFLOW_MODEL_NAME}@champion")
            client = mlflow.tracking.MlflowClient()
            self.version = client.get_model_version_by_alias(
                MLFLOW_MODEL_NAME, "champion"
            ).version
            logger.info(f"Model loaded successfully (version: {self.version})")
        except Exception as e:
            logger.error("Failed to load model", exc_info=e)

    def predict(self, texts: list[str]):
        if not self.model:
            raise RuntimeError("Model not loaded")
        return self.model.predict(texts)

    def reload(self):
        logger.info("Reloading model...")
        self._load_model()


transformer_store = TransformerModelStore()


def get_transformer_store() -> TransformerModelStore:
    return transformer_store