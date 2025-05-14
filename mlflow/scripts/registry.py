# import mlflow
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# from typing import Optional, List, Dict, Any
# import logging
# from datetime import datetime
# import json
# from src.models.transformer_manager import ModelManager
# from config.dataClient import get_blob_storage
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ModelRegistry:
#     def __init__(
#         self,
#         tracking_uri: Optional[str] = None
#     ):
#         # Set tracking URI
#         if tracking_uri:
#             mlflow.set_tracking_uri(tracking_uri)
#         elif not mlflow.get_tracking_uri():
#             mlflow.set_tracking_uri("http://localhost:5000")

#         try:
#             self.blob_storage = get_blob_storage()
#             logger.info("Successfully got blob storage instance.")
#         except Exception as e:
#             logger.error(f"Failed to get blob storage: {e}")
#             raise

#         try:
#             artifact_uri = f"wasbs://artifact@testlakehouse.blob.core.windows.net"
#             mlflow.get_artifact_uri(artifact_uri)
#             logger.info(f"MLflow artifact URI set to: {artifact_uri}")
#         except Exception as e:
#             logger.error(f"Error setting MLflow artifact URI: {e}")
#             raise

#         try:
#             self.client = mlflow.tracking.MlflowClient()
#             logger.info("MLflow client initialized successfully.")
#         except Exception as e:
#             logger.error(f"Failed to initialize MLflow client: {e}")
#             raise

#     def register_model(
#         self,
#         run_id: str,
#         model_name: str,
#         description: Optional[str] = None,
#         metrics: Optional[Dict[str, float]] = None,
#         model_manager: Optional[ModelManager] = None,
#         tags: Optional[Dict[str, str]] = None
#     ) -> str:
#         """
#         Register a model in MLflow Model Registry with support for description, metrics, and tags.
        
#         Args:
#             run_id: MLflow run ID
#             model_name: Name for the registered model
#             description: Optional description for the model
#             metrics: Optional dictionary of metrics to log
#             model_manager: Optional ModelManager instance to save model artifacts
#             tags: Optional dictionary of tags to associate with the model
            
#         Returns:
#             Version number of the registered model
#         """
#         try:
#             uri = f"runs:/{run_id}/model"
#             details = mlflow.register_model(model_uri=uri, name=model_name)
#             version = details.version

#             # Update description
#             if description:
#                 self.client.update_registered_model(name=model_name, description=description)
#                 logger.info(f"Updated description for model '{model_name}'")

#             # Add tags if provided
#             if tags:
#                 # Add tags to the registered model
#                 for tag_key, tag_value in tags.items():
#                     self.client.set_model_version_tag(
#                         name=model_name,
#                         version=version,
#                         key=tag_key,
#                         value=tag_value
#                     )
#                 logger.info(f"Added {len(tags)} tags to model '{model_name}' version {version}")

#             # Log metrics if provided
#             if metrics:
#                 for metric_name, value in metrics.items():
#                     if isinstance(value, (int, float)):
#                         self.client.log_metric(run_id, metric_name, value)
#                 logger.info(f"Logged {len(metrics)} metrics to run {run_id}")

#             # Save model artifacts if model_manager is provided
#             if model_manager:
#                 self.save_model_artifacts(model_manager, run_id)

#             logger.info(f"Model '{model_name}' registered as version {version}")
#             return version

#         except Exception as e:
#             logger.error(f"Error registering model: {e}")
#             raise

#     def add_aliases(self, model_name: str, version: str, aliases: List[str]) -> None:
#         """
#         Add aliases to a registered model version.
        
#         Args:
#             model_name: Name of the registered model
#             version: Version of the model
#             aliases: List of aliases to add
#         """
#         try:
#             for alias in aliases:
#                 try:
#                     self.client.set_registered_model_alias(model_name, alias, version)
#                     logger.info(f"Added alias '{alias}' to model {model_name} v{version}")
#                 except Exception as alias_error:
#                     logger.warning(f"Couldn't add alias '{alias}': {alias_error}")
            
#             logger.info(f"Added {len(aliases)} aliases to model {model_name} v{version}")
#         except Exception as e:
#             logger.error(f"Error adding aliases: {e}")
#             raise

#     def save_model_artifacts(self, model_manager: ModelManager, run_id: str) -> None:
#         """Save model artifacts to MLflow run"""
#         try:
#             # Save model and tokenizer
#             mlflow.transformers.log_model(
#                 transformers_model={
#                     "model": model_manager.model,
#                     "tokenizer": model_manager.tokenizer
#                 },
#                 artifact_path="model",
#                 run_id=run_id
#             )

#             # Save model config
#             if hasattr(model_manager.model, "config"):
#                 config_path = os.path.join(model_manager.output_dir, "config.json")
#                 model_manager.model.config.to_json_file(config_path)
#                 mlflow.log_artifact(config_path, run_id=run_id)

#             logger.info(f"Model artifacts saved for run {run_id}")

#         except Exception as e:
#             logger.error(f"Error saving model artifacts: {e}")
#             raise

#     def load_model(
#         self,
#         model_name: str,
#         version: Optional[str] = None,
#         stage: str = "Production"
#     ) -> "ModelManager":
#         """Load a model from registry, fallback with repo_type if needed."""
#         # Build MLflow model URI
#         model_uri = f"models:/{model_name}/{version or stage}"
#         logger.info(f"Loading from MLflow URI: {model_uri}")

#         # Try loading via MLflow transformers
#         try:
#             loaded = mlflow.transformers.load_model(model_uri)
#         except ValueError as e:
#             if "Repo id must be in the form" in str(e):
#                 logger.warning("Retrying with repo_type='model'")
#                 loaded = mlflow.transformers.load_model(
#                     model_uri,
#                     repo_type="model"
#                 )
#             else:
#                 logger.error(f"Loading failed: {e}")
#                 raise

#                 # Determine how to extract model and tokenizer
#         if isinstance(loaded, dict):
#             model_obj = loaded.get("model")
#             tokenizer_obj = loaded.get("tokenizer")
#         elif hasattr(loaded, "model") and hasattr(loaded, "tokenizer"):
#             model_obj = loaded.model
#             tokenizer_obj = loaded.tokenizer
#         else:
#             raise RuntimeError("Loaded object does not contain model and tokenizer")

#         # Instantiate manager with preloaded objects
#         manager = ModelManager(
#             model_path=model_uri,
#             finetune=False,
#             model_obj=model_obj,
#             tokenizer_obj=tokenizer_obj
#         )
#         return manager

#     def list_models(self) -> List[Dict]:
#         try:
#             regs = self.client.search_registered_models()
#             return [
#                 {
#                     'name': m.name,
#                     'latest_versions': [
#                         {
#                             'version': v.version,
#                             'status': v.status,
#                             'run_id': v.run_id,
#                             'metrics': self.client.get_run(v.run_id).data.metrics
#                         }
#                         for v in m.latest_versions
#                     ]
#                 }
#                 for m in regs
#             ]
#         except Exception as e:
#             logger.error(f"Error listing models: {e}")
#             raise

#     def get_latest_version(self, model_name: str) -> str:
#         try:
#             reg = self.client.get_registered_model(model_name)
#             return reg.latest_versions[0].version
#         except Exception as e:
#             logger.error(f"Error fetching latest version: {e}")
#             raise

#     def transition_model_stage(self, model_name: str, version: str, stage: str) -> None:
#         try:
#             self.client.transition_model_version_stage(name=model_name, version=version, stage=stage)
#             logger.info(f"Transitioned {model_name} v{version} to {stage}")
#         except Exception as e:
#             logger.error(f"Error transitioning model stage: {e}")
#             raise

#     def delete_model_version(self, model_name: str, version: str) -> None:
#         try:
#             self.client.delete_model_version(name=model_name, version=version)
#             logger.info(f"Deleted {model_name} v{version}")
#         except Exception as e:
#             logger.error(f"Error deleting model version: {e}")
#             raise

#     def save_model(self, model, tokenizer, output_dir: str) -> None:
#         """
#         Save model and tokenizer then upload the entire directory to Azure Blob (if blob_storage is configured).
#         """
#         try:
#             os.makedirs(output_dir, exist_ok=True)
#             # Save locally
#             model.save_pretrained(output_dir)
#             tokenizer.save_pretrained(output_dir)
#             logger.info(f"Saved model locally at {output_dir}")

#             if self.blob_storage:
#                 # Assuming BlobStorage has upload_directory method
#                 self.blob_storage.upload_directory(
#                     container_name="artifact",
#                     local_path=output_dir,
#                     remote_path=os.path.basename(output_dir)
#                 )
#                 logger.info(f"Uploaded artifacts from {output_dir} to Blob container 'artifact'")

#         except Exception as e:
#             logger.error(f"Error saving/uploading model: {e}")
#             raise

# import mlflow
# import os
# import sys
# import tempfile
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# from typing import Optional, List, Dict, Any
# import logging
# from datetime import datetime
# import json
# from src.models.transformer_manager import ModelManager
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ModelRegistry:
#     def __init__(
#         self,
#         tracking_uri: Optional[str] = None,
#         artifact_uri: Optional[str] = None
#     ):
#         # Set tracking URI
#         if tracking_uri:
#             mlflow.set_tracking_uri(tracking_uri)
#         elif not mlflow.get_tracking_uri():
#             mlflow.set_tracking_uri("http://localhost:5000")

#         # Set artifact URI for MLflow
#         try:
#             self.artifact_uri = artifact_uri or "wasbs://artifact@testlakehouse.blob.core.windows.net"
#             # Configure MLflow to use this artifact location
#             os.environ["MLFLOW_ARTIFACT_URI"] = self.artifact_uri
#             logger.info(f"MLflow artifact URI set to: {self.artifact_uri}")
#         except Exception as e:
#             logger.error(f"Error setting MLflow artifact URI: {e}")
#             raise

#         # Initialize MLflow client
#         try:
#             self.client = mlflow.tracking.MlflowClient()
#             logger.info("MLflow client initialized successfully.")
#         except Exception as e:
#             logger.error(f"Failed to initialize MLflow client: {e}")
#             raise

#     def register_model(
#         self,
#         run_id: str,
#         model_name: str,
#         description: Optional[str] = None,
#         metrics: Optional[Dict[str, float]] = None,
#         model_manager: Optional[ModelManager] = None,
#         tags: Optional[Dict[str, str]] = None
#     ) -> str:
#         """
#         Register a model in MLflow Model Registry with support for description, metrics, and tags.
        
#         Args:
#             run_id: MLflow run ID
#             model_name: Name for the registered model
#             description: Optional description for the model
#             metrics: Optional dictionary of metrics to log
#             model_manager: Optional ModelManager instance to save model artifacts
#             tags: Optional dictionary of tags to associate with the model
            
#         Returns:
#             Version number of the registered model
#         """
#         try:
#             # Save model artifacts if provided
#             if model_manager:
#                 self.save_model_artifacts(model_manager, run_id)
                
#             # Register model using MLflow
#             uri = f"runs:/{run_id}/model"
#             details = mlflow.register_model(model_uri=uri, name=model_name)
#             version = details.version

#             # Update description
#             if description:
#                 self.client.update_registered_model(name=model_name, description=description)
#                 logger.info(f"Updated description for model '{model_name}'")

#             # Add tags if provided
#             if tags:
#                 # Add model timestamp tag
#                 if "registered_at" not in tags:
#                     tags["registered_at"] = datetime.now().isoformat()
                
#                 # Add tags to the registered model version
#                 for tag_key, tag_value in tags.items():
#                     self.client.set_model_version_tag(
#                         name=model_name,
#                         version=version,
#                         key=tag_key,
#                         value=tag_value
#                     )
#                 logger.info(f"Added {len(tags)} tags to model '{model_name}' version {version}")

#             # Log metrics if provided
#             if metrics:
#                 for metric_name, value in metrics.items():
#                     if isinstance(value, (int, float)):
#                         self.client.log_metric(run_id, metric_name, value)
#                 logger.info(f"Logged {len(metrics)} metrics to run {run_id}")

#             logger.info(f"Model '{model_name}' registered as version {version}")
#             return version

#         except Exception as e:
#             logger.error(f"Error registering model: {e}")
#             raise

#     def add_aliases(self, model_name: str, version: str, aliases: List[str]) -> None:
#         """
#         Add aliases to a registered model version.
        
#         Args:
#             model_name: Name of the registered model
#             version: Version of the model
#             aliases: List of aliases to add
#         """
#         try:
#             for alias in aliases:
#                 try:
#                     self.client.set_registered_model_alias(model_name, alias, version)
#                     logger.info(f"Added alias '{alias}' to model {model_name} v{version}")
#                 except Exception as alias_error:
#                     logger.warning(f"Couldn't add alias '{alias}': {alias_error}")
            
#             logger.info(f"Added {len(aliases)} aliases to model {model_name} v{version}")
#         except Exception as e:
#             logger.error(f"Error adding aliases: {e}")
#             raise

#     def save_model_artifacts(self, model_manager: ModelManager, run_id: str) -> None:
#         """
#         Save model artifacts to MLflow run
        
#         This method saves the model and tokenizer to MLflow's artifact store using MLflow's API,
#         ensuring that they can be properly loaded later from the blob storage.
#         """
#         try:
#             # Log the model using MLflow's transformers flavor
#             mlflow.transformers.log_model(
#                 transformers_model={
#                     "model": model_manager.model,
#                     "tokenizer": model_manager.tokenizer
#                 },
#                 artifact_path="model",
#                 run_id=run_id
#             )
#             logger.info(f"Model artifacts saved for run {run_id} via MLflow")

#             # Save any additional artifacts that might be useful
#             with tempfile.TemporaryDirectory() as tmp_dir:
#                 # Save model config if available
#                 if hasattr(model_manager.model, "config"):
#                     config_path = os.path.join(tmp_dir, "config.json")
#                     model_manager.model.config.to_json_file(config_path)
#                     mlflow.log_artifact(config_path, artifact_path="model_config", run_id=run_id)
                
#                 # Save metadata information
#                 metadata = {
#                     "model_name": model_manager.model_name,
#                     "model_path": model_manager.model_path,
#                     "saved_at": datetime.now().isoformat(),
#                     "device": str(model_manager.device)
#                 }
#                 metadata_path = os.path.join(tmp_dir, "metadata.json")
#                 with open(metadata_path, "w") as f:
#                     json.dump(metadata, f, indent=2)
#                 mlflow.log_artifact(metadata_path, artifact_path="metadata", run_id=run_id)
            
#             logger.info(f"Additional artifacts saved for run {run_id}")
#         except Exception as e:
#             logger.error(f"Error saving model artifacts: {e}")
#             raise

#     def load_model(
#         self,
#         model_name: str,
#         version: Optional[str] = None,
#         stage: str = "Production",
#         alias: Optional[str] = None
#     ) -> "ModelManager":
#         """
#         Load a model from MLflow registry
        
#         Args:
#             model_name: Name of the registered model
#             version: Optional specific version to load
#             stage: Stage to load from if version and alias not provided
#             alias: Optional alias to load from
            
#         Returns:
#             ModelManager instance with loaded model and tokenizer
#         """
#         # Determine the URI based on provided parameters
#         if alias:
#             model_uri = f"models:/{model_name}@{alias}"
#         elif version:
#             model_uri = f"models:/{model_name}/{version}"
#         else:
#             model_uri = f"models:/{model_name}/{stage}"
            
#         logger.info(f"Loading model from MLflow URI: {model_uri}")
        
#         # Try loading with multiple fallback approaches
#         try:
#             # First attempt - standard loading
#             loaded = mlflow.transformers.load_model(model_uri)
#         except ValueError as e:
#             if "Repo id must be in the form" in str(e):
#                 logger.warning("Retrying with repo_type='model'")
#                 try:
#                     # Second attempt - specifying repo_type
#                     loaded = mlflow.transformers.load_model(
#                         model_uri,
#                         repo_type="model"
#                     )
#                 except Exception as e2:
#                     logger.warning(f"Retry failed, trying alternate loading: {e2}")
                    
#                     # Third attempt - download artifacts and load locally
#                     try:
#                         with tempfile.TemporaryDirectory() as tmp_dir:
#                             # Get model version details
#                             if version:
#                                 mv = self.client.get_model_version(model_name, version)
#                             elif alias:
#                                 mv = self.client.get_model_version_by_alias(model_name, alias)
#                             else:
#                                 versions = self.client.get_latest_versions(model_name, stages=[stage])
#                                 if not versions:
#                                     raise ValueError(f"No models found for {model_name} in {stage} stage")
#                                 mv = versions[0]
                            
#                             # Download the model artifacts
#                             artifact_uri = mv.source
#                             local_path = os.path.join(tmp_dir, "model")
#                             self.client.download_artifacts(mv.run_id, "model", local_path)
                            
#                             # Create ModelManager
#                             manager = ModelManager(
#                                 model_path=local_path,
#                                 finetune=False
#                             )
#                             return manager
#                     except Exception as e3:
#                         logger.error(f"All loading attempts failed: {e3}")
#                         raise
#             else:
#                 logger.error(f"Loading failed: {e}")
#                 raise

#         # Determine how to extract model and tokenizer from the loaded object
#         if isinstance(loaded, dict):
#             model_obj = loaded.get("model")
#             tokenizer_obj = loaded.get("tokenizer")
#         elif hasattr(loaded, "model") and hasattr(loaded, "tokenizer"):
#             model_obj = loaded.model
#             tokenizer_obj = loaded.tokenizer
#         else:
#             raise RuntimeError("Loaded object does not contain model and tokenizer")

#         # Instantiate manager with preloaded objects
#         manager = ModelManager(
#             model_path=model_uri,
#             finetune=False,
#             model_obj=model_obj,
#             tokenizer_obj=tokenizer_obj
#         )
#         return manager

#     def list_models(self) -> List[Dict]:
#         """
#         List all registered models with their latest versions and metrics
#         """
#         try:
#             regs = self.client.search_registered_models()
#             result = []
            
#             for m in regs:
#                 model_info = {
#                     'name': m.name,
#                     'latest_versions': []
#                 }
                
#                 # Get all versions with their details
#                 for v in m.latest_versions:
#                     # Try to get run metrics
#                     try:
#                         metrics = self.client.get_run(v.run_id).data.metrics
#                     except Exception:
#                         metrics = {}
                    
#                     # Get tags for this version
#                     try:
#                         tags = {t.key: t.value for t in 
#                                self.client.get_model_version_tags(m.name, v.version)}
#                     except Exception:
#                         tags = {}
                    
#                     # Add to the result
#                     model_info['latest_versions'].append({
#                         'version': v.version,
#                         'status': v.status,
#                         'stage': v.current_stage,
#                         'run_id': v.run_id,
#                         'metrics': metrics,
#                         'tags': tags
#                     })
                
#                 result.append(model_info)
            
#             return result
#         except Exception as e:
#             logger.error(f"Error listing models: {e}")
#             raise

#     def get_latest_version(self, model_name: str, stage: Optional[str] = None) -> str:
#         """
#         Get the latest version number for a model, optionally filtered by stage
#         """
#         try:
#             if stage:
#                 versions = self.client.get_latest_versions(model_name, stages=[stage])
#                 if not versions:
#                     raise ValueError(f"No versions found for model {model_name} in stage {stage}")
#                 return versions[0].version
#             else:
#                 reg = self.client.get_registered_model(model_name)
#                 if not reg.latest_versions:
#                     raise ValueError(f"No versions found for model {model_name}")
#                 return reg.latest_versions[0].version
#         except Exception as e:
#             logger.error(f"Error fetching latest version: {e}")
#             raise

#     def get_aliases(self, model_name: str, version: Optional[str] = None) -> Dict[str, str]:
#         """
#         Get all aliases for a model, or for a specific model version
        
#         Returns:
#             Dictionary mapping alias names to version numbers
#         """
#         try:
#             aliases = {}
#             # Get all registered model details
#             model = self.client.get_registered_model(model_name)
            
#             # Iterate through all versions to find aliases
#             for v in model.latest_versions:
#                 if version and v.version != version:
#                     continue
                    
#                 # Get aliases for this version
#                 v_aliases = self.client.get_model_version_aliases(model_name, v.version)
#                 for alias in v_aliases:
#                     aliases[alias] = v.version
            
#             return aliases
#         except Exception as e:
#             logger.error(f"Error getting aliases: {e}")
#             raise

#     def transition_model_stage(self, model_name: str, version: str, stage: str) -> None:
#         """
#         Transition a model version to a new stage
#         """
#         try:
#             self.client.transition_model_version_stage(name=model_name, version=version, stage=stage)
#             logger.info(f"Transitioned {model_name} v{version} to {stage}")
#         except Exception as e:
#             logger.error(f"Error transitioning model stage: {e}")
#             raise

#     def delete_model_version(self, model_name: str, version: str) -> None:
#         """
#         Delete a model version
#         """
#         try:
#             self.client.delete_model_version(name=model_name, version=version)
#             logger.info(f"Deleted {model_name} v{version}")
#         except Exception as e:
#             logger.error(f"Error deleting model version: {e}")
#             raise

import mlflow
import os
import sys
import tempfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import json
from src.models.transformer_manager import ModelManager
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        artifact_uri: Optional[str] = None,
        azure_storage_connection_string='DefaultEndpointsProtocol=https;AccountName=testlakehouse;AccountKey=nkIF81ebY5Ma27qJ2DsXvhCT/hUvztoKi0duG46L6OBZ+4FZ/1/hJL3eV6yaPqu3G489E6zIzDdj+AStdK3Y5w==;EndpointSuffix=core.windows.net',
        azure_storage_access_key='nkIF81ebY5Ma27qJ2DsXvhCT/hUvztoKi0duG46L6OBZ+4FZ/1/hJL3eV6yaPqu3G489E6zIzDdj+AStdK3Y5w==',
        cmd_timeout: int = 600
    ):
        # 1) Set MLflow tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif not mlflow.get_tracking_uri():
            mlflow.set_tracking_uri("http://localhost:5000")
        logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

        # 2) Configure Azure Blob credentials
        try:
            # Priority: connection string, then access key, then DefaultAzureCredential
            if azure_storage_connection_string:
                os.environ["AZURE_STORAGE_CONNECTION_STRING"] = azure_storage_connection_string
                logger.info("AZURE_STORAGE_CONNECTION_STRING set for artifact storage.")
            elif azure_storage_access_key:
                os.environ["AZURE_STORAGE_ACCOUNT_KEY"] = azure_storage_access_key
                logger.info("AZURE_STORAGE_ACCOUNT_KEY set for artifact storage.")
            else:
                logger.info("No Azure Storage creds provided; using DefaultAzureCredential if available.")

            # Set artifact root URI
            self.artifact_uri = artifact_uri or "wasbs://testartifact@testlakehouse.blob.core.windows.net/mlflowartifact"
            logger.info(f"Configured MLflow artifact root: {self.artifact_uri}")

            # Configure MLflow client-side timeout for upload/download
            os.environ["MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"] = str(cmd_timeout)
            logger.info(f"Set MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT={cmd_timeout} seconds.")

            # Select experiment (artifact_location defined server-side)
            mlflow.set_experiment(experiment_name="Default")

            # Verify artifact location if possible
            try:
                mlflow.artifacts.list_artifacts(self.artifact_uri)
            except Exception:
                logger.warning("Could not verify artifact URI; ensure tracking server uses same artifact root.")
        except Exception as e:
            logger.error(f"Error configuring artifact storage: {e}")
            raise

        # 3) Initialize MLflow client
        try:
            self.client = mlflow.tracking.MlflowClient()
            logger.info("MLflow client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow client: {e}")
            raise

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
            # First verify if the run exists
            try:
                run = self.client.get_run(run_id)
                logger.info(f"Found run {run_id} with status: {run.info.status}")
            except Exception as e:
                logger.error(f"Run {run_id} not found or not accessible: {e}")
                raise ValueError(f"Invalid run_id: {run_id}")
                
            # Save model artifacts if provided
            if model_manager:
                self.save_model_artifacts(model_manager, run_id)
                
            # Register model using MLflow
            uri = f"runs:/{run_id}/model"
            logger.info(f"Registering model from URI: {uri}")
            
            # Check if the model artifact exists in the run
            artifacts = self.client.list_artifacts(run_id, path="model")
            if not artifacts:
                logger.warning(f"No model artifacts found in run {run_id}. Registration may fail.")
                
            # Register the model
            details = mlflow.register_model(model_uri=uri, name=model_name)
            version = details.version

            # Update description
            if description:
                self.client.update_registered_model(name=model_name, description=description)
                logger.info(f"Updated description for model '{model_name}'")

            # Add tags if provided
            if tags:
                # Add model timestamp tag
                if "registered_at" not in tags:
                    tags["registered_at"] = datetime.now().isoformat()
                
                # Add tags to the registered model version
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

            logger.info(f"Model '{model_name}' registered as version {version}")
            return version

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def save_model_artifacts(self, model_manager: ModelManager, run_id: str) -> None:
        """
        Save model artifacts to MLflow run
        
        This method saves the model and tokenizer to MLflow's artifact store using MLflow's API,
        ensuring that they can be properly loaded later from the blob storage.
        """
        try:
            # Verify run exists and is active
            try:
                run = self.client.get_run(run_id)
                if run.info.status != "RUNNING":
                    # If run is not active, start a new run
                    logger.warning(f"Run {run_id} is not active. Starting a new run.")
                    with mlflow.start_run(run_id=run_id) as run:
                        return self._log_model_artifacts(model_manager, run.info.run_id)
                else:
                    return self._log_model_artifacts(model_manager, run_id)
            except Exception as e:
                logger.error(f"Error accessing run {run_id}: {e}")
                logger.info("Starting a new run to save artifacts")
                with mlflow.start_run() as run:
                    result = self._log_model_artifacts(model_manager, run.info.run_id)
                    logger.info(f"Created new run {run.info.run_id} for artifacts")
                    return result
                
        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            raise

    def _log_model_artifacts(self, model_manager: ModelManager, run_id: str) -> None:
        """
        Internal method to log model artifacts to a specific run
        """
        # Log the model using MLflow's transformers flavor
        logger.info(f"Logging transformer model artifacts to run {run_id}")
        
        # Make sure the model artifacts are valid
        if not model_manager.model or not model_manager.tokenizer:
            raise ValueError("Model manager must have both model and tokenizer initialized")
            
        mlflow.transformers.log_model(
            transformers_model={
                "model": model_manager.model,
                "tokenizer": model_manager.tokenizer
            },
            artifact_path="model",
            run_id=run_id
        )
        logger.info(f"Model artifacts saved for run {run_id} via MLflow")

        # Save any additional artifacts that might be useful
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model config if available
            if hasattr(model_manager.model, "config"):
                config_path = os.path.join(tmp_dir, "config.json")
                model_manager.model.config.to_json_file(config_path)
                mlflow.log_artifact(config_path, artifact_path="model_config", run_id=run_id)
                logger.info("Saved model configuration")
            
            # Save metadata information
            metadata = {
                "model_name": model_manager.model_name,
                "model_path": model_manager.model_path,
                "saved_at": datetime.now().isoformat(),
                "device": str(model_manager.device)
            }
            metadata_path = os.path.join(tmp_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            mlflow.log_artifact(metadata_path, artifact_path="metadata", run_id=run_id)
            logger.info("Saved model metadata")
        
        # Verify artifacts were saved
        artifacts = self.client.list_artifacts(run_id)
        if not artifacts:
            logger.warning(f"No artifacts found in run {run_id} after saving!")
        else:
            logger.info(f"Successfully saved {len(artifacts)} artifacts to run {run_id}")

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: str = "Production",
        alias: Optional[str] = None
    ) -> "ModelManager":
        """
        Load a model from MLflow registry
        
        Args:
            model_name: Name of the registered model
            version: Optional specific version to load
            stage: Stage to load from if version and alias not provided
            alias: Optional alias to load from
            
        Returns:
            ModelManager instance with loaded model and tokenizer
        """
        # Determine the URI based on provided parameters
        if alias:
            model_uri = f"models:/{model_name}@{alias}"
        elif version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/{stage}"
            
        logger.info(f"Loading model from MLflow URI: {model_uri}")
        
        # Try loading with multiple fallback approaches
        try:
            # First attempt - standard loading
            loaded = mlflow.transformers.load_model(model_uri)
        except ValueError as e:
            if "Repo id must be in the form" in str(e):
                logger.warning("Retrying with repo_type='model'")
                try:
                    # Second attempt - specifying repo_type
                    loaded = mlflow.transformers.load_model(
                        model_uri,
                        repo_type="model"
                    )
                except Exception as e2:
                    logger.warning(f"Retry failed, trying alternate loading: {e2}")
                    
                    # Third attempt - download artifacts and load locally
                    try:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            # Get model version details
                            if version:
                                mv = self.client.get_model_version(model_name, version)
                            elif alias:
                                mv = self.client.get_model_version_by_alias(model_name, alias)
                            else:
                                versions = self.client.get_latest_versions(model_name, stages=[stage])
                                if not versions:
                                    raise ValueError(f"No models found for {model_name} in {stage} stage")
                                mv = versions[0]
                            
                            # Download the model artifacts
                            artifact_uri = mv.source
                            local_path = os.path.join(tmp_dir, "model")
                            self.client.download_artifacts(mv.run_id, "model", local_path)
                            
                            # Create ModelManager
                            manager = ModelManager(
                                model_path=local_path,
                                finetune=False
                            )
                            return manager
                    except Exception as e3:
                        logger.error(f"All loading attempts failed: {e3}")
                        raise
            else:
                logger.error(f"Loading failed: {e}")
                raise

        # Determine how to extract model and tokenizer from the loaded object
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
        """
        List all registered models with their latest versions and metrics
        """
        try:
            regs = self.client.search_registered_models()
            result = []
            
            for m in regs:
                model_info = {
                    'name': m.name,
                    'latest_versions': []
                }
                
                # Get all versions with their details
                for v in m.latest_versions:
                    # Try to get run metrics
                    try:
                        metrics = self.client.get_run(v.run_id).data.metrics
                    except Exception:
                        metrics = {}
                    
                    # Get tags for this version
                    try:
                        tags = {t.key: t.value for t in 
                               self.client.get_model_version_tags(m.name, v.version)}
                    except Exception:
                        tags = {}
                    
                    # Add to the result
                    model_info['latest_versions'].append({
                        'version': v.version,
                        'status': v.status,
                        'stage': v.current_stage,
                        'run_id': v.run_id,
                        'metrics': metrics,
                        'tags': tags
                    })
                
                result.append(model_info)
            
            return result
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise

    def get_latest_version(self, model_name: str, stage: Optional[str] = None) -> str:
        """
        Get the latest version number for a model, optionally filtered by stage
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
                if not versions:
                    raise ValueError(f"No versions found for model {model_name} in stage {stage}")
                return versions[0].version
            else:
                reg = self.client.get_registered_model(model_name)
                if not reg.latest_versions:
                    raise ValueError(f"No versions found for model {model_name}")
                return reg.latest_versions[0].version
        except Exception as e:
            logger.error(f"Error fetching latest version: {e}")
            raise

    def get_aliases(self, model_name: str, version: Optional[str] = None) -> Dict[str, str]:
        """
        Get all aliases for a model, or for a specific model version
        
        Returns:
            Dictionary mapping alias names to version numbers
        """
        try:
            aliases = {}
            # Get all registered model details
            model = self.client.get_registered_model(model_name)
            
            # Iterate through all versions to find aliases
            for v in model.latest_versions:
                if version and v.version != version:
                    continue
                    
                # Get aliases for this version
                v_aliases = self.client.get_model_version_aliases(model_name, v.version)
                for alias in v_aliases:
                    aliases[alias] = v.version
            
            return aliases
        except Exception as e:
            logger.error(f"Error getting aliases: {e}")
            raise

    def transition_model_stage(self, model_name: str, version: str, stage: str) -> None:
        """
        Transition a model version to a new stage
        """
        try:
            self.client.transition_model_version_stage(name=model_name, version=version, stage=stage)
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            raise

    def delete_model_version(self, model_name: str, version: str) -> None:
        """
        Delete a model version
        """
        try:
            self.client.delete_model_version(name=model_name, version=version)
            logger.info(f"Deleted {model_name} v{version}")
        except Exception as e:
            logger.error(f"Error deleting model version: {e}")
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