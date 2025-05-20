import os
import logging
from typing import Optional
from functools import lru_cache
#from scripts.registry import ModelRegistry

from mlflow.pyfunc import PyFuncModel
import mlflow.pyfunc
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(f'{BASE_DIR}/.env')

MLFLOW_MODEL_NAME = os.getenv('MODEL_NAME')

logger = logging.getLogger(__name__)

class ModelStore:
    def __init__(self):
        self.model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}@champion")

    def predict(self, texts: list[str]):
        return self.model.predict(texts)

    def reload(self):
        self.model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}@champion")

