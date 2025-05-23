import os 
import sys
import uuid
import logging
import argparse
import pandas as pd
import numpy as np
import io
from typing import List, Optional, Tuple, Dict, Any
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from dotenv import load_dotenv
import getpass
# import urllib.parse

# Add path to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(root_dir)

from src.models.preprocess import preprocess_data, COMMON_SENTIMENT_MAPPINGS
from config.dataClient import get_blob_storage
from src.models.transformer_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s ─ %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up Azure Storage connection
lake = os.getenv("LAKE_STORAGE_CONN_STR")
if lake and not os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = lake
container_name = os.getenv("AZURE_BLOB_CONTAINER", "testartifact")

# Set up MLflow tracking server
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://172.23.51.243:5000")
artifact_uri = f'wasbs://{container_name}@twitlakehouse.blob.core.windows.net/mlflow-artifacts'

mlflow.set_tracking_uri(tracking_uri)
logger.info(f"Tracking URI: {tracking_uri}")
logger.info(f"Artifact URI: {artifact_uri}")

# Create MLflow client
client = MlflowClient(tracking_uri=tracking_uri)


# def encode_model_name(name: str) -> str:
#     return urllib.parse.quote(name, safe='')

def get_current_champion() -> Optional[Tuple[str, str]]:
    """
    Get current champion model name and version by searching across all registered models.
    Returns (model_name, version) tuple or None if no champion found.
    """
    try:
        registered_models = client.search_registered_models()
        
        for model in registered_models:
            try:
                model_details = client.get_registered_model(model.name)
                if model_details.aliases:
                    for alias in model_details.aliases:
                        if alias.alias == "champion":
                            return (model.name, alias.version)
            except Exception:
                continue  # Skip this model and continue searching
                
        return None
    except Exception as e:
        logger.error(f"Error searching for champion: {e}")
        return None

# ==== Registry helper functions ====
def get_model_version(model_name: str, version: Optional[str] = None, stage: Optional[str] = None):
    """Get specific model version from registry"""
    try:
        if version:
            return client.get_model_version(model_name, version)
        elif stage:
            latest_versions = client.get_latest_versions(model_name, stages=[stage])
            if latest_versions:
                return latest_versions[0]
        return None
    except Exception as e:
        logger.error(f"Error getting model version: {e}")
        return None

def load_model_from_registry(model_name: str, version: str):
    """Load model and tokenizer from registry using mlflow.transformers.load_model"""
    try:
        # encoded_name = encode_model_name(model_name)
        model_uri = f"models:/{model_name}/{version}"
        logger.info(f"Loading model from URI: {model_uri}")
        
        # Load using mlflow.transformers.load_model
        try:
            loaded = mlflow.transformers.load_model(model_uri)
        except ValueError as e:
            if "Repo id must be in the form" in str(e):
                logger.warning("Retrying with repo_type='model'")
                loaded = mlflow.transformers.load_model(model_uri, repo_type="model")
            else:
                logger.error(f"Failed to load model: {e}")
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

        if model_obj is None or tokenizer_obj is None:
            raise RuntimeError("Failed to extract model or tokenizer from loaded object")

        # Wrap in ModelManager and return
        return ModelManager(
            model_path=model_uri,
            finetune=False,
            model_obj=model_obj,
            tokenizer_obj=tokenizer_obj
        )
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name} v{version}: {e}")
        raise

# ==== Load data function ====
def load_and_preprocess_from_blob(
    tokenizer, max_length: int = 128, label_map: Optional[dict] = None,
    container_name: str = "testartifact", prefix: str = "data/"
) -> Optional[object]:
    """
    Load test data from blob, convert to DataFrame, preprocess into dataset for model.
    """
    try:
        blob = get_blob_storage()
        test_json_str = blob.read_json_from_container(container_name, prefix + "test_dataset.json")
        logger.info("Successfully loaded JSON data from blob storage.")
        
        if isinstance(test_json_str, list):
            test_df = pd.DataFrame(test_json_str)
        else:
            test_df = pd.read_json(io.StringIO(test_json_str), lines=True)

        logger.info(f"Test shape: {test_df.shape}")

        X_test = test_df['text'].tolist()
        y_test = test_df['sentiment'].tolist()
        test_dataset = preprocess_data(tokenizer=tokenizer, texts=X_test, labels=y_test,
                                       max_length=max_length, label_map=label_map)

        logger.info(f"Successfully preprocessed test dataset size: {len(test_dataset)}")
        return test_dataset

    except Exception as e:
        logger.error(f"Error in load_and_preprocess_from_blob: {e}")
        return None


# ==== Model promotion functions ====
def promote_to_challenger(model_name: str, version: str, run_id: Optional[str] = None) -> bool:
    """Promote model to challenger alias"""
    try:
        try:
            existing_challenger = client.get_model_version_by_alias(model_name, "challenger")
            if existing_challenger:
                client.delete_registered_model_alias(model_name, "challenger")
        except mlflow.exceptions.RestException as e:
            if "Registered model alias challenger not found" not in str(e):
                raise  # only ignore specific alias-not-found error
        
        # Set new challenger alias
        client.set_registered_model_alias(model_name, "challenger", version)
        
        # Set promotion tags
        timestamp = datetime.now().isoformat()
        client.set_model_version_tag(model_name, version, "promotion.challenger_date", timestamp)
        if run_id:
            client.set_model_version_tag(model_name, version, "promotion.challenger_run_id", run_id)
        
        logger.info(f"Successfully promoted {model_name} v{version} to challenger")
        return True
        
    except Exception as e:
        logger.error(f"Failed to promote {model_name} v{version} to challenger: {e}")
        return False


def promote_to_champion(model_name: str, challenger_version: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare challenger with current champion and promote if challenger is better.
    Champion and challenger can be completely different models (different base models).
    Returns detailed comparison results.
    """
    # Ensure champion_comparison is always defined
    champion_comparison = {
        "champion_model_name": None,
        "champion_version": None,
        "champion_f1": None,
        "champion_accuracy": None
    }

    try:
        # Get current champion info using helper function
        champion_info = get_current_champion()
        champion_model_version = None
        champion_model_name = None
        champion_version = None
        
        if champion_info:
            champion_model_name, champion_version = champion_info
            try:
                champion_model_version = client.get_model_version(champion_model_name, champion_version)
                logger.info(f"Found existing champion: {champion_model_name} v{champion_version}")
            except Exception as e:
                logger.error(f"Error getting champion model version: {e}")
                champion_model_version = None
        else:
            logger.info("No existing champion found across all models")

        # Load challenger model
        challenger_mgr = load_model_from_registry(model_name, challenger_version)
        logger.info(f"Loaded challenger model {model_name} v{challenger_version}")

        # Load evaluation dataset with challenger's tokenizer
        challenger_eval_ds = load_and_preprocess_from_blob(
            tokenizer=challenger_mgr.tokenizer,
            label_map=COMMON_SENTIMENT_MAPPINGS.get('text')
        )

        if not challenger_eval_ds:
            raise ValueError("Failed to load evaluation dataset for challenger")

        # Evaluate challenger
        challenger_metrics = challenger_mgr.evaluate(challenger_eval_ds, batch_size=32)
        challenger_f1 = challenger_metrics.get("f1", 0.0)
        challenger_accuracy = challenger_metrics.get("accuracy", 0.0)

        logger.info(f"Challenger ({model_name} v{challenger_version}) metrics - Accuracy: {challenger_accuracy:.4f}, F1: {challenger_f1:.4f}")

        # Log challenger metrics if run_id provided
        if run_id:
            for k, v in challenger_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"challenger_{k}", v)

        # Compare with champion if exists
        promote_challenger = True  # Default to True if no champion exists

        if champion_model_version and champion_model_name and champion_version:
            logger.info(f"Comparing challenger ({model_name} v{challenger_version}) with existing champion ({champion_model_name} v{champion_version})")

            # Load champion model - note: could be a completely different model
            champion_mgr = load_model_from_registry(champion_model_name, champion_version)

            # Load evaluation dataset with champion's tokenizer (separate evaluation)
            # This is important because champion might have different tokenizer/preprocessing
            champion_eval_ds = load_and_preprocess_from_blob(
                tokenizer=champion_mgr.tokenizer,
                label_map=COMMON_SENTIMENT_MAPPINGS.get('text')
            )

            if champion_eval_ds:
                # Evaluate champion on the same task but with its own preprocessing
                champion_metrics = champion_mgr.evaluate(champion_eval_ds, batch_size=32)
                champion_f1 = champion_metrics.get("f1", 0.0)
                champion_accuracy = champion_metrics.get("accuracy", 0.0)

                logger.info(f"Champion ({champion_model_name} v{champion_version}) metrics - Accuracy: {champion_accuracy:.4f}, F1: {champion_f1:.4f}")

                # Log champion metrics if run_id provided
                if run_id:
                    for k, v in champion_metrics.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(f"champion_{k}", v)

                champion_comparison.update({
                    "champion_model_name": champion_model_name,
                    "champion_version": champion_version,
                    "champion_f1": champion_f1,
                    "champion_accuracy": champion_accuracy
                })

                # Compare F1 scores to decide promotion
                promote_challenger = challenger_f1 > champion_f1

                logger.info(f"Model Comparison:")
                logger.info(f"  Challenger: {model_name} v{challenger_version} - F1: {challenger_f1:.4f}")
                logger.info(f"  Champion: {champion_model_name} v{champion_version} - F1: {champion_f1:.4f}")
                logger.info(f"  Promote challenger: {promote_challenger}")
            else:
                logger.warning(f"Failed to load evaluation dataset for champion ({champion_model_name} v{champion_version}), promoting challenger by default")
        else:
            logger.info("No existing champion found, promoting challenger")

        # Promote challenger to champion if better
        promoted = False
        if promote_challenger:
            # Remove existing champion alias if it exists
            if champion_model_version:
                try:
                    # Remove champion alias from the current champion model
                    client.delete_registered_model_alias(champion_model_name, "champion")
                    logger.info(f"Removed champion alias from {champion_model_name}")
                except Exception as delete_error:
                    logger.warning(f"Failed to delete existing champion alias: {delete_error}")

            # Set new champion - assign "champion" alias to the challenger
            client.set_registered_model_alias(model_name, "champion", challenger_version)

            # Add promotion metadata
            timestamp = datetime.now().isoformat()
            client.set_model_version_tag(model_name, challenger_version, "promotion.champion_date", timestamp)
            if run_id:
                client.set_model_version_tag(model_name, challenger_version, "promotion.champion_run_id", run_id)
            
            # Log what model was replaced (if any)
            if champion_model_name and champion_version:
                client.set_model_version_tag(model_name, challenger_version, "promotion.replaced_champion", f"{champion_model_name}_v{champion_version}")

            promoted = True
            logger.info(f"Successfully promoted {model_name} v{challenger_version} to champion (replaced {champion_model_name} v{champion_version} if existed)")
        else:
            logger.info(f"Challenger {model_name} v{challenger_version} not promoted - champion {champion_model_name} v{champion_version} performs better")

        return {
            "status": "success",
            "promoted": promoted,
            "challenger_model_name": model_name,
            "challenger_version": challenger_version,
            "challenger_f1": challenger_f1,
            "challenger_accuracy": challenger_accuracy,
            "champion_comparison": champion_comparison
        }

    except Exception as e:
        error_msg = f"Failed to promote {model_name} v{challenger_version} to champion: {e}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "promoted": False,
            "challenger_model_name": model_name,
            "challenger_version": challenger_version,
            "challenger_f1": None,
            "challenger_accuracy": None,
            "champion_comparison": champion_comparison
        }

# ==== Core validation function ====
def validate_model_performance(
    model_name: str,
    version: Optional[str] = None,
    stage: str = "Staging",
    batch_size: int = 32,
    min_accuracy: float = 0.8,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate a model's performance against requirements using mlflow.transformers.load_model.
    Only logs metrics if run_id is provided.
    
    Args:
        model_name: Name of the registered model
        version: Version of the model (optional)
        stage: Stage of the model (used if version is not provided)
        batch_size: Batch size for evaluation
        min_accuracy: Minimum accuracy required to pass validation
        run_id: MLflow run ID for logging (optional)
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Get model version
        model_version_obj = get_model_version(model_name, version, stage)
        if not model_version_obj:
            error_msg = f"No model found for {model_name} with version={version} or stage={stage}"
            logger.error(error_msg)
            return {
                "model_name": model_name,
                "version": version or f"latest from {stage}",
                "status": "failed",
                "error": error_msg
            }
        
        model_version = model_version_obj.version
        logger.info(f"Validating model {model_name} v{model_version}")
        
        # Load model using transformers loader
        try:
            challenger_mgr = load_model_from_registry(model_name, model_version)
            logger.info(f"Loaded model {model_name} v{model_version} for validation")
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            logger.error(error_msg)
            return {
                "model_name": model_name,
                "version": model_version,
                "status": "failed",
                "error": error_msg
            }
        
        # Load evaluation dataset using model's tokenizer
        try:
            eval_ds = load_and_preprocess_from_blob(
                tokenizer=challenger_mgr.tokenizer,
                label_map=COMMON_SENTIMENT_MAPPINGS.get('text')
            )
            
            if not eval_ds:
                raise ValueError("Failed to load evaluation dataset")
                
            logger.info(f"Loaded evaluation dataset with {len(eval_ds)} samples")
        except Exception as e:
            error_msg = f"Failed to load dataset: {e}"
            logger.error(error_msg)
            return {
                "model_name": model_name,
                "version": model_version,
                "status": "failed",
                "error": error_msg
            }
        
        # Evaluate model
        challenger_metrics = challenger_mgr.evaluate(eval_ds, batch_size)
        challenger_accuracy = challenger_metrics.get("accuracy")
        challenger_f1 = challenger_metrics.get("f1")
        logger.info(f"Challenger model metrics - Accuracy: {challenger_accuracy:.4f}, F1: {challenger_f1:.4f}")
        
        # Log metrics if run_id provided
        if run_id:
            for k, v in challenger_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"validation_{k}", v)
        
        # Determine if validation passed
        validation_passed = challenger_accuracy >= min_accuracy
        
        # Set validation tags on model version
        timestamp = datetime.now().isoformat()
        if run_id:
            client.set_model_version_tag(model_name, model_version, "validation.run_id", run_id)
        client.set_model_version_tag(model_name, model_version, "validation.date", timestamp)
        client.set_model_version_tag(model_name, model_version, "validation.accuracy", f"{challenger_accuracy:.4f}")
        client.set_model_version_tag(model_name, model_version, "validation.f1", f"{challenger_f1:.4f}")
        client.set_model_version_tag(model_name, model_version, "validation.status", "approved" if validation_passed else "rejected")

        return {
            "model_name": model_name,
            "version": model_version,
            "status": "success",
            "validation": {
                "passed": validation_passed,
                "accuracy": challenger_accuracy,
                "f1_score": challenger_f1,
                "threshold": min_accuracy
            },
            "metrics": challenger_metrics
        }
        
    except Exception as e:
        error_msg = f"Validation error: {e}"
        logger.error(error_msg)
        return {
            "model_name": model_name,
            "version": version or f"latest from {stage}",
            "status": "failed",
            "error": error_msg
        }


# ==== Complete validation workflow ====
def promote_model(
    model_name: str,
    version: Optional[str] = None,
    stage: str = "Staging",
    batch_size: int = 32,
    min_accuracy: float = 0.8,
    promote_to_champion_if_better: bool = True,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete workflow: validate -> promote to challenger -> promote to champion
    
    Args:
        model_name: Name of the registered model
        version: Version of the model (optional)
        stage: Stage of the model (used if version is not provided)
        batch_size: Batch size for evaluation
        min_accuracy: Minimum accuracy required to pass validation
        promote_to_champion_if_better: Whether to compare with champion and promote if better
        run_id: MLflow run ID for logging (optional)
        
    Returns:
        Dictionary with complete workflow results
    """
    # Step 1: Validate model
    validation_result = validate_model_performance(
        model_name=model_name,
        version=version,
        stage=stage,
        batch_size=batch_size,
        min_accuracy=min_accuracy,
        run_id=run_id
    )
    
    if validation_result["status"] != "success" or not validation_result["validation"]["passed"]:
        return {
            "workflow": "validation_failed",
            "validation": validation_result,
            "challenger_promotion": None,
            "champion_promotion": None
        }
    
    # Step 2: Promote to challenger
    model_version = validation_result["version"]
    challenger_promotion = promote_to_challenger(model_name, model_version, run_id)
    
    if not challenger_promotion:
        return {
            "workflow": "challenger_promotion_failed",
            "validation": validation_result,
            "challenger_promotion": False,
            "champion_promotion": None
        }
    
    # Step 3: Promote to champion if requested
    champion_promotion = None
    if promote_to_champion_if_better:
        champion_promotion = promote_to_champion(model_name, model_version, run_id)
    
    return {
        "workflow": "completed",
        "validation": validation_result,
        "challenger_promotion": challenger_promotion,
        "champion_promotion": champion_promotion
    }


# ==== Entry point ====
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Validate and promote a registered model")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name of the registered model to validate")
    parser.add_argument("--version", type=str, default=None,
                        help="Version of the model to validate (optional)")
    parser.add_argument("--stage", type=str, default="Staging",
                        help="Stage of the model to validate (used if version is not provided)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--experiment-name", type=str, default="model-validation",
                        help="MLflow experiment name")
    parser.add_argument("--min-accuracy", type=float, default=0.5,
                        help="Minimum accuracy required to pass validation (default: 0.8)")
    parser.add_argument("--promote-to-champion", type=bool, default=True,
                        help="Promote to champion if better than current champion")
    
    args = parser.parse_args()
    
    # Set up MLflow experiment
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        print(f"Creating new experiment: {args.experiment_name}")
        experiment_id = mlflow.create_experiment(
            name=args.experiment_name,
            artifact_location=artifact_uri
        )
        print(f"Created experiment with ID: {experiment_id}")
    else:
        print(f"Using existing experiment: {args.experiment_name} (ID: {experiment.experiment_id})")
        experiment_id = experiment.experiment_id
    
    # Print MLflow configuration
    print("\nMLflow Configuration:")
    print(f"- Tracking URI: {tracking_uri}")
    print(f"- Artifact URI: {artifact_uri}")
    print(f"- Azure Storage: {'Configured' if lake else 'Not configured'}")
    print(f"- Container: {container_name}")
    print("\n" + "-"*50 + "\n")
    
    # Start MLflow run
    mlflow.set_experiment(args.experiment_name)
    
    if mlflow.active_run():
        mlflow.end_run()
    
    run_name = f"{args.model_name}-validation-{uuid.uuid4().hex[:8]}"
    print(f"Starting new run: {run_name}")
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        run_id = run.info.run_id
        
        # Log parameters
        mlflow.log_params({
            "model_name": args.model_name,
            "version": args.version,
            "stage": args.stage,
            "batch_size": args.batch_size,
            "min_accuracy": args.min_accuracy,
            "promote_to_champion": args.promote_to_champion
        })
        mlflow.set_tag("created_by", getpass.getuser())
        mlflow.set_tag("created_at", datetime.now().isoformat())
        
        # Run validation and promotion workflow
        print(f"Validating model: {args.model_name}")
        print(f"- Version: {args.version or f'(latest from {args.stage})'}")
        print(f"- Minimum accuracy: {args.min_accuracy}")
        print(f"- Promote to champion: {args.promote_to_champion}")
        print("\n" + "-"*50 + "\n")
        
        result = promote_model(
            model_name=args.model_name,
            version=args.version,
            stage=args.stage,
            batch_size=args.batch_size,
            min_accuracy=args.min_accuracy,
            promote_to_champion_if_better=args.promote_to_champion,
            run_id=run_id
        )
        
        # Log workflow result
        mlflow.log_param("workflow_status", result["workflow"])
        
        print(f"Run completed with ID: {run_id}")
        
        # Print results
        print("\nWorkflow Results:")
        print(f"- Workflow status: {result['workflow']}")
        
        if result["validation"]:
            val_result = result["validation"]
            print(f"- Model: {val_result['model_name']} v{val_result['version']}")
            print(f"- Validation status: {val_result['status']}")
            
            if val_result['status'] == 'success':
                print(f"- Validation passed: {val_result['validation']['passed']}")
                print(f"- Accuracy: {val_result['validation']['accuracy']:.4f} (threshold: {val_result['validation']['threshold']})")
                print(f"- F1 Score: {val_result['validation']['f1_score']:.4f}")
            else:
                print(f"- Error: {val_result.get('error', 'Unknown error')}")
        
        if result["challenger_promotion"]:
            print(f"- Challenger promotion: {'Success' if result['challenger_promotion'] else 'Failed'}")
        
        if result["champion_promotion"]:
            champ_result = result["champion_promotion"]
            print(f"- Champion promotion: {champ_result['status']}")
            if champ_result['status'] == 'success':
                print(f"- Promoted to champion: {champ_result['promoted']}")
                if champ_result['champion_comparison']['champion_version']:
                    print(f"- Previous champion: v{champ_result['champion_comparison']['champion_version']} (F1: {champ_result['champion_comparison']['champion_f1']:.4f})")
                print(f"- New challenger: v{champ_result['challenger_version']} (F1: {champ_result['challenger_f1']:.4f})")
        
        print("\n" + "-"*50 + "\n")
        
        # Print run URL
        tracking_uri_clean = mlflow.get_tracking_uri()
        run_url = f"{tracking_uri_clean.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
        print(f"View this run at: {run_url}")
