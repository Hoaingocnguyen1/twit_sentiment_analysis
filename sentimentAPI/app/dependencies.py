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

# # import os
# # import logging
# # import mlflow
# # from mlflow.transformers import load_model

# # logger = logging.getLogger(__name__)
# # logging.basicConfig(level=logging.INFO)

# # MLFLOW_MODEL_NAME = os.getenv("MODEL_NAME")
# # MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# # if not MLFLOW_MODEL_NAME or not MLFLOW_TRACKING_URI:
# #     raise EnvironmentError("Both MODEL_NAME and MLFLOW_TRACKING_URI must be set")

# # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# # class TransformerModelStore:
# #     def __init__(self):
# #         self.model = None
# #         self.version = None
# #         self._load_model()

# #     def _load_model(self):
# #         try:
# #             logger.info(f"Loading model: models:/{MLFLOW_MODEL_NAME}@champion")
# #             self.model = load_model(f"models:/{MLFLOW_MODEL_NAME}@champion")
# #             client = mlflow.tracking.MlflowClient()
# #             self.version = client.get_model_version_by_alias(
# #                 MLFLOW_MODEL_NAME, "champion"
# #             ).version
# #             logger.info(f"Model loaded successfully (version: {self.version})")
# #         except Exception as e:
# #             logger.error("Failed to load model", exc_info=e)

# #     def predict(self, texts: list[str]):
# #         if not self.model:
# #             raise RuntimeError("Model not loaded")
# #         return self.model.predict(texts)

# #     def reload(self):
# #         logger.info("Reloading model...")
# #         self._load_model()


# # transformer_store = TransformerModelStore()


# # def get_transformer_store() -> TransformerModelStore:
# #     return transformer_store


# # dependencies.py - Phiên bản cải thiện
# # dependencies.py - Phiên bản cải thiện
# import os
# import logging
# import mlflow
# from mlflow.transformers import load_model
# import time

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# MLFLOW_MODEL_NAME = os.getenv("MODEL_NAME")
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# if not MLFLOW_MODEL_NAME or not MLFLOW_TRACKING_URI:
#     raise EnvironmentError("Both MODEL_NAME and MLFLOW_TRACKING_URI must be set")

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# class TransformerModelStore:
#     def __init__(self, max_retries=1, retry_delay=5):
#         self.model = None
#         self.version = None
#         self.max_retries = max_retries
#         self.retry_delay = retry_delay
#         self._load_model()

#     def _load_model(self, raise_on_error=True):
#         """
#         Load model với retry mechanism và error handling tốt hơn

#         Args:
#             raise_on_error: Có raise exception khi load fail hay không (mặc định True)
#         """
#         for attempt in range(self.max_retries + 1):
#             try:
#                 logger.info(
#                     f"Loading model (attempt {attempt + 1}/{self.max_retries + 1}): models:/{MLFLOW_MODEL_NAME}@champion"
#                 )

#                 # Load model
#                 self.model = load_model(f"models:/{MLFLOW_MODEL_NAME}@champion")

#                 # Get version info
#                 client = mlflow.tracking.MlflowClient()
#                 self.version = client.get_model_version_by_alias(
#                     MLFLOW_MODEL_NAME, "champion"
#                 ).version

#                 # Test prediction để đảm bảo model hoạt động
#                 test_result = self.model.predict(["test sentiment"])
#                 logger.info(
#                     f"Model loaded successfully (version: {self.version}), test prediction: {test_result}"
#                 )
#                 return True

#             except Exception as e:
#                 logger.error(
#                     f"Attempt {attempt + 1} failed to load model: {str(e)}",
#                     exc_info=True,
#                 )

#                 # Reset model state on failure
#                 self.model = None
#                 self.version = None

#                 # Nếu không phải attempt cuối cùng, đợi rồi thử lại
#                 if attempt < self.max_retries:
#                     logger.info(f"Retrying in {self.retry_delay} seconds...")
#                     time.sleep(self.retry_delay)
#                 else:
#                     # Attempt cuối cùng failed
#                     if raise_on_error:
#                         logger.error(
#                             "All attempts to load model failed. Application will not start."
#                         )
#                         raise RuntimeError(
#                             f"Failed to load model after {self.max_retries + 1} attempts. Last error: {str(e)}"
#                         )
#                     else:
#                         logger.warning(
#                             "All attempts to load model failed, but continuing without raising error."
#                         )
#                         return False

#         return False

#     def predict(self, texts: list[str]):
#         if not self.model:
#             raise RuntimeError("Model not loaded. Cannot make predictions.")

#         try:
#             return self.model.predict(texts)
#         except Exception as e:
#             logger.error(f"Prediction failed: {str(e)}", exc_info=True)
#             raise RuntimeError(f"Prediction failed: {str(e)}")

#     def reload(self):
#         """
#         Reload model - không raise error để tránh crash API
#         """
#         logger.info("Reloading model...")
#         old_model = self.model
#         old_version = self.version

#         try:
#             # Thử load model mới với raise_on_error=False
#             success = self._load_model(raise_on_error=False)
#             if success:
#                 logger.info("Model reloaded successfully")
#                 return True
#             else:
#                 # Giữ model cũ nếu có
#                 if old_model is not None:
#                     self.model = old_model
#                     self.version = old_version
#                     logger.warning("Failed to reload model, keeping old model")
#                 else:
#                     logger.error(
#                         "Failed to reload model and no previous model available"
#                     )
#                 return False
#         except Exception as e:
#             # Restore old model if reload fails
#             self.model = old_model
#             self.version = old_version
#             logger.error(f"Error during model reload: {str(e)}", exc_info=True)
#             return False

#     def is_healthy(self):
#         """Check if model is loaded and working"""
#         if not self.model:
#             return False

#         try:
#             # Simple health check
#             self.model.predict(["health check"])
#             return True
#         except Exception as e:
#             logger.error(f"Health check failed: {str(e)}")
#             return False


# # Tạo instance với error handling
# try:
#     transformer_store = TransformerModelStore()
# except Exception as e:
#     logger.error(f"Failed to initialize model store: {str(e)}")
#     # Có thể chọn exit application hoặc tạo dummy store
#     import sys

#     print(f"CRITICAL ERROR: Cannot start application - {str(e)}")
#     sys.exit(1)  # Exit nếu không thể load model ban đầu


# def get_transformer_store() -> TransformerModelStore:
#     return transformer_store
