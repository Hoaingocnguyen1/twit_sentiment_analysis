import os
import logging
from typing import Optional
from functools import lru_cache
from scripts.registry import ModelRegistry

logger = logging.getLogger(__name__)

@lru_cache()
def get_model_manager() -> ModelRegistry:
    """
    Load model manager as a singleton using lru_cache for efficient reuse.
    Returns cached instance after first call.
    """
    model_name = os.getenv("MODEL_NAME")
    model_version = os.getenv("MODEL_VERSION") or None
    model_stage = os.getenv("MODEL_STAGE", "Production")
    
    if not model_name:
        raise RuntimeError("Environment variable MODEL_NAME is required.")
    
    logger.info(f"Initializing model manager (name={model_name}, version={model_version}, stage={model_stage})")
    registry = ModelRegistry()
    
    try:
        if model_version:
            model_manager = registry.load_model(model_name=model_name, version=model_version)
            logger.info(f"Loaded model {model_name} version {model_version}")
        else:
            model_manager = registry.load_model(model_name=model_name, stage=model_stage)
            latest = registry.get_latest_version(model_name)
            logger.info(f"Loaded model {model_name}@{model_stage} (latest version: {latest})")
        return model_manager
    except Exception as e:
        logger.exception("Failed to load model")
        raise

# import os
# from scripts.registry import ModelRegistry
# from typing import Optional
# import logging

# logger = logging.getLogger(__name__)

# # Load once at startup
# _model_manager: Optional[ModelRegistry] = None


# def get_model_manager() -> ModelRegistry:
#     global _model_manager
#     if _model_manager is None:
#         model_name = os.getenv("MODEL_NAME")
#         model_version = os.getenv("MODEL_VERSION") or None
#         model_stage = os.getenv("MODEL_STAGE", "Production")

#         if not model_name:
#             raise RuntimeError("Environment variable MODEL_NAME is required.")

#         logger.info(f"Initializing model manager (name={model_name}, version={model_version}, stage={model_stage})")
#         registry = ModelRegistry()
#         try:
#             if model_version:
#                 _model_manager = registry.load_model(model_name=model_name, version=model_version)
#                 logger.info(f"Loaded model {model_name} version {model_version}")
#             else:
#                 _model_manager = registry.load_model(model_name=model_name, stage=model_stage)
#                 latest = registry.get_latest_version(model_name)
#                 logger.info(f"Loaded model {model_name}@{model_stage} (latest version: {latest})")
#         except Exception as e:
#             logger.exception("Failed to load model")
#             raise
#     return _model_manager