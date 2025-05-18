import mlflow
import os
import sys
import time
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import json
from src.models.transformer_manager import ModelManager
# from config.dataClient import get_blob_storage
import torch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Các hằng số cho kết nối MLflow
MAX_RETRIES = 10
RETRY_INTERVAL = 5  # seconds

def check_mlflow_connection():
    """
    Kiểm tra kết nối tới MLflow server và thử lại nếu không thành công
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    for i in range(MAX_RETRIES):
        try:
            # Thử kết nối tới MLflow
            client = mlflow.tracking.MlflowClient()
            client.search_experiments()
            logger.info(f"Successfully connected to MLflow at {tracking_uri}")
            return True
        except Exception as e:
            logger.warning(f"Attempt {i+1}/{MAX_RETRIES}: Failed to connect to MLflow: {e}")
            if i < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_INTERVAL} seconds...")
                time.sleep(RETRY_INTERVAL)
    
    logger.error(f"Could not connect to MLflow after {MAX_RETRIES} attempts")
    return False


class ModelRegistry:
    def __init__(
        self,
        tracking_uri: Optional[str] = None
    ):
        """
        Khởi tạo ModelRegistry với MLflow tracking URI
        
        Args:
            tracking_uri: Optional URI tới MLflow tracking server
        """
        
        # Thiết lập tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        else:
            # Sử dụng service name từ docker-compose
            mlflow.set_tracking_uri("http://mlflow-server:5000")
            
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Kiểm tra kết nối tới MLflow
        if not check_mlflow_connection():
            logger.error("Exiting due to MLflow connection failure")
            sys.exit(1)
            
        try:
            # Lấy và hiển thị artifact URI hiện tại
            current_artifact_uri = mlflow.get_artifact_uri()
            logger.info(f"Current MLflow artifact URI: {current_artifact_uri}")
        except Exception as e:
            logger.error(f"Error getting MLflow artifact URI: {e}")

        # try:
        #     self.blob_storage = get_blob_storage()
        #     logger.info("Successfully got blob storage instance.")
        # except Exception as e:
        #     logger.error(f"Failed to get blob storage: {e}")
        #     raise

        try:
            self.client = mlflow.tracking.MlflowClient()
            logger.info("MLflow client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow client: {e}")
            raise

    def register_model(
        self,
        run_id: str,
        model_name: str,
        description: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        model_manager: Optional[ModelManager] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a model in MLflow Model Registry with support for description, metrics, and tags.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            description: Optional description for the model
            metrics: Optional dictionary of metrics to log
            model_manager: Optional ModelManager instance to save model artifacts
            tags: Optional dictionary of tags to associate with the model
            
        Returns:
            Version number of the registered model
        """
        try:
            uri = f"runs:/{run_id}/model"
            details = mlflow.register_model(model_uri=uri, name=model_name)
            version = details.version

            # Update description
            if description:
                self.client.update_registered_model(name=model_name, description=description)
                logger.info(f"Updated description for model '{model_name}'")

            # Add tags if provided
            if tags:
                # Add tags to the registered model
                for tag_key, tag_value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=version,
                        key=tag_key,
                        value=tag_value
                    )
                logger.info(f"Added {len(tags)} tags to model '{model_name}' version {version}")

            # Log metrics if provided
            if metrics:
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.client.log_metric(run_id, metric_name, value)
                logger.info(f"Logged {len(metrics)} metrics to run {run_id}")

            # Save model artifacts if model_manager is provided
            if model_manager:
                self.save_model_artifacts(model_manager, run_id)

            logger.info(f"Model '{model_name}' registered as version {version}")
            return version

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def add_aliases(self, model_name: str, version: str, aliases: List[str]) -> None:
        """
        Add aliases to a registered model version.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            aliases: List of aliases to add
        """
        try:
            for alias in aliases:
                try:
                    self.client.set_registered_model_alias(model_name, alias, version)
                    logger.info(f"Added alias '{alias}' to model {model_name} v{version}")
                except Exception as alias_error:
                    logger.warning(f"Couldn't add alias '{alias}': {alias_error}")
            
            logger.info(f"Added {len(aliases)} aliases to model {model_name} v{version}")
        except Exception as e:
            logger.error(f"Error adding aliases: {e}")
            raise

    def save_model_artifacts(self, model_manager: ModelManager, run_id: str, task: str = "text-classification") -> None:
        """Save model artifacts to MLflow run"""
        try:
            # Save model and tokenizer
            mlflow.transformers.log_model(
                transformers_model={
                    "model": model_manager.model,
                    "tokenizer": model_manager.tokenizer
                },
                artifact_path="model",
                run_id=run_id,
                task=task
            )

            # Save model config
            if hasattr(model_manager.model, "config"):
                config_path = os.path.join(model_manager.output_dir, "config_model.json")
                model_manager.model.config.to_json_file(config_path)
                mlflow.log_artifact(config_path, run_id=run_id)

            logger.info(f"Model artifacts saved for run {run_id}")

        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            raise

    def load_model(
            self,
            model_name: str,
            model_path: Optional[str] = None,
            version: Optional[str] = None,
            stage: str = "Production"
        ) -> "ModelManager":
            """Load from local path if provided, otherwise from MLflow registry"""
            # 1) If local model path specified, load directly
            if model_path:
                logger.info(f"Loading model from local path: {model_path}")
                # Try loading local directory
                model_obj = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    local_files_only=True,
                    device_map='cuda:0' if torch.cuda.is_available() else 'cpu'
                )
                tokenizer_obj = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    device_map='cuda:0' if torch.cuda.is_available() else 'cpu'
                )
                return ModelManager(
                    model_name=model_name,
                    model_path=model_path,
                    finetune=False,
                    model_obj=model_obj,
                    tokenizer_obj=tokenizer_obj
                )

            # 2) Otherwise load from MLflow model registry
            model_uri = f"models:/{model_name}/{version or stage}"
            logger.info(f"Loading from MLflow URI: {model_uri}")

            try:
                loaded = mlflow.transformers.load_model(model_uri)
            except ValueError as e:
                if "Repo id must be in the form" in str(e):
                    logger.warning("Retrying with repo_type='model'")
                    loaded = mlflow.transformers.load_model(model_uri, repo_type="model")
                else:
                    logger.error(f"Loading failed: {e}")
                    raise

            # Extract model and tokenizer
            if isinstance(loaded, dict):
                model_obj = loaded.get("model")
                tokenizer_obj = loaded.get("tokenizer")
            elif hasattr(loaded, "model") and hasattr(loaded, "tokenizer"):
                model_obj = loaded.model
                tokenizer_obj = loaded.tokenizer
            else:
                raise RuntimeError("Loaded object does not contain model and tokenizer")

            # Wrap in ModelManager and return
            return ModelManager(
                model_path=model_uri,
                finetune=False,
                model_obj=model_obj,
                tokenizer_obj=tokenizer_obj)

    def list_models(self) -> List[Dict]:
        try:
            regs = self.client.search_registered_models()
            return [
                {
                    'name': m.name,
                    'latest_versions': [
                        {
                            'version': v.version,
                            'status': v.status,
                            'run_id': v.run_id,
                            'metrics': self.client.get_run(v.run_id).data.metrics
                        }
                        for v in m.latest_versions
                    ]
                }
                for m in regs
            ]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise

    def get_latest_version(self, model_name: str) -> str:
        try:
            reg = self.client.get_registered_model(model_name)
            return reg.latest_versions[0].version
        except Exception as e:
            logger.error(f"Error fetching latest version: {e}")
            raise

    def transition_model_stage(self, model_name: str, version: str, stage: str) -> None:
        try:
            self.client.transition_model_version_stage(name=model_name, version=version, stage=stage)
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            raise

    def delete_model_version(self, model_name: str, version: str) -> None:
        try:
            self.client.delete_model_version(name=model_name, version=version)
            logger.info(f"Deleted {model_name} v{version}")
        except Exception as e:
            logger.error(f"Error deleting model version: {e}")
            raise