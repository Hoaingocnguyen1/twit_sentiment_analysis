import mlflow
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import json
from src.models.transformer_manager import ModelManager
from config.dataClient import get_blob_storage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(
        self,
        tracking_uri: Optional[str] = None
    ):
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif not mlflow.get_tracking_uri():
            mlflow.set_tracking_uri("http://localhost:5000")

        try:
            self.blob_storage = get_blob_storage()
            logger.info("Successfully got blob storage instance.")
        except Exception as e:
            logger.error(f"Failed to get blob storage: {e}")
            raise

        try:
            artifact_uri = f"wasbs://artifact@testlakehouse.blob.core.windows.net"
            mlflow.get_artifact_uri(artifact_uri)
            logger.info(f"MLflow artifact URI set to: {artifact_uri}")
        except Exception as e:
            logger.error(f"Error setting MLflow artifact URI: {e}")
            raise

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

    def save_model_artifacts(self, model_manager: ModelManager, run_id: str) -> None:
        """Save model artifacts to MLflow run"""
        try:
            # Save model and tokenizer
            mlflow.transformers.log_model(
                transformers_model={
                    "model": model_manager.model,
                    "tokenizer": model_manager.tokenizer
                },
                artifact_path="model",
                run_id=run_id
            )

            # Save model config
            if hasattr(model_manager.model, "config"):
                config_path = os.path.join(model_manager.output_dir, "config.json")
                model_manager.model.config.to_json_file(config_path)
                mlflow.log_artifact(config_path, run_id=run_id)

            logger.info(f"Model artifacts saved for run {run_id}")

        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            raise

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: str = "Production"
    ) -> "ModelManager":
        """Load a model from registry, fallback with repo_type if needed."""
        # Build MLflow model URI
        model_uri = f"models:/{model_name}/{version or stage}"
        logger.info(f"Loading from MLflow URI: {model_uri}")

        # Try loading via MLflow transformers
        try:
            loaded = mlflow.transformers.load_model(model_uri)
        except ValueError as e:
            if "Repo id must be in the form" in str(e):
                logger.warning("Retrying with repo_type='model'")
                loaded = mlflow.transformers.load_model(
                    model_uri,
                    repo_type="model"
                )
            else:
                logger.error(f"Loading failed: {e}")
                raise

                # Determine how to extract model and tokenizer
        if isinstance(loaded, dict):
            model_obj = loaded.get("model")
            tokenizer_obj = loaded.get("tokenizer")
        elif hasattr(loaded, "model") and hasattr(loaded, "tokenizer"):
            model_obj = loaded.model
            tokenizer_obj = loaded.tokenizer
        else:
            raise RuntimeError("Loaded object does not contain model and tokenizer")

        # Instantiate manager with preloaded objects
        manager = ModelManager(
            model_path=model_uri,
            finetune=False,
            model_obj=model_obj,
            tokenizer_obj=tokenizer_obj
        )
        return manager

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

    def save_model(self, model, tokenizer, output_dir: str) -> None:
        """
        Save model and tokenizer then upload the entire directory to Azure Blob (if blob_storage is configured).
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Save locally
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved model locally at {output_dir}")

            if self.blob_storage:
                # Assuming BlobStorage has upload_directory method
                self.blob_storage.upload_directory(
                    container_name="artifact",
                    local_path=output_dir,
                    remote_path=os.path.basename(output_dir)
                )
                logger.info(f"Uploaded artifacts from {output_dir} to Blob container 'artifact'")

        except Exception as e:
            logger.error(f"Error saving/uploading model: {e}")
            raise