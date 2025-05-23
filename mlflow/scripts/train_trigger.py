import sys
import os
import logging
import argparse
import getpass
from datetime import datetime as dt
from pathlib import Path
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any
import io
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
# Internal imports
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(root_dir)
from src.models.preprocess import preprocess_data, COMMON_SENTIMENT_MAPPINGS
from config.dataClient import get_blob_storage
from src.models.transformer_manager import ModelManager


# Configure logging
logging.basicConfig(level=logging.INFO)
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

# ==== Load data function with improved error handling ====
def load_and_preprocess_from_blob(
    tokenizer, max_length: int = 128, label_map: Optional[dict] = None,
    container_name: str = "testartifact", prefix: str = "data/"
) -> Tuple[Optional[object], Optional[object]]:
    """
    Load train/test data from blob, convert to DataFrame, preprocess into dataset for model.
    Fixed to handle dimension issues and data type validation.
    """
    try:
        blob = get_blob_storage()

        train_json_str = blob.read_json_from_container(container_name, prefix + "train_dataset.json")
        test_json_str = blob.read_json_from_container(container_name, prefix + "test_dataset.json")

        logger.info("Successfully loaded JSON data from blob storage.")

        # Parse JSON data with better error handling
        if isinstance(train_json_str, list):
            train_df = pd.DataFrame(train_json_str)
        elif isinstance(train_json_str, str):
            train_df = pd.read_json(io.StringIO(train_json_str), lines=True)
        else:
            raise ValueError(f"Unexpected train data type: {type(train_json_str)}")

        if isinstance(test_json_str, list):
            test_df = pd.DataFrame(test_json_str)
        elif isinstance(test_json_str, str):
            test_df = pd.read_json(io.StringIO(test_json_str), lines=True)
        else:
            raise ValueError(f"Unexpected test data type: {type(test_json_str)}")

        logger.info(f"Parsed dataframes - Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        # Validate required columns
        required_columns = ['text', 'sentiment']
        for col in required_columns:
            if col not in train_df.columns:
                raise ValueError(f"Missing required column '{col}' in train dataset")
            if col not in test_df.columns:
                raise ValueError(f"Missing required column '{col}' in test dataset")

        # Extract and validate data types
        X_train = train_df['text'].astype(str).tolist()  # Ensure strings
        y_train = train_df['sentiment'].tolist()
        X_test = test_df['text'].astype(str).tolist()   # Ensure strings
        y_test = test_df['sentiment'].tolist()

        # Validate data integrity
        if not X_train or not y_train:
            raise ValueError("Empty training data")
        if not X_test or not y_test:
            raise ValueError("Empty test data")
        
        if len(X_train) != len(y_train):
            raise ValueError(f"Mismatch in train data: {len(X_train)} texts vs {len(y_train)} labels")
        if len(X_test) != len(y_test):
            raise ValueError(f"Mismatch in test data: {len(X_test)} texts vs {len(y_test)} labels")

        logger.info(f"Successfully extracted lists - train texts: {len(X_train)}, train labels: {len(y_train)}")
        logger.info(f"Test texts: {len(X_test)}, test labels: {len(y_test)}")
        
        # Log sample data for debugging
        if X_train:
            logger.info(f"Sample train text: '{str(X_train[0])[:50]}...'")
            logger.info(f"Sample train label: {y_train[0]} (type: {type(y_train[0])})")

        # Preprocess data with enhanced error handling
        try:
            train_dataset = preprocess_data(
                tokenizer=tokenizer, 
                texts=X_train, 
                labels=y_train,
                max_length=max_length, 
                label_map=label_map
            )
            logger.info(f"Train dataset preprocessed successfully, size: {len(train_dataset)}")
        except Exception as e:
            logger.error(f"Error preprocessing train data: {e}")
            raise ValueError(f"Train data preprocessing failed: {e}")

        try:
            test_dataset = preprocess_data(
                tokenizer=tokenizer, 
                texts=X_test, 
                labels=y_test,
                max_length=max_length, 
                label_map=label_map
            )
            logger.info(f"Test dataset preprocessed successfully, size: {len(test_dataset)}")
        except Exception as e:
            logger.error(f"Error preprocessing test data: {e}")
            raise ValueError(f"Test data preprocessing failed: {e}")

        # Validate dataset structure
        if hasattr(train_dataset, '__len__'):
            logger.info(f"Final validation - train dataset size: {len(train_dataset)}")
        if hasattr(test_dataset, '__len__'):
            logger.info(f"Final validation - test dataset size: {len(test_dataset)}")

        return train_dataset, test_dataset

    except Exception as e:
        logger.error(f"Error in load_and_preprocess_from_blob: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return None, None 

def train_model(manager, run_id, train_ds, eval_ds, params):
    """Train model with enhanced error handling"""
    try:
        # Validate datasets before training
        if train_ds is None or eval_ds is None:
            raise ValueError("Training or evaluation dataset is None")
        
        # Log parameters
        mlflow.log_params(params)
        
        # Log dataset info
        if hasattr(train_ds, '__len__'):
            mlflow.log_metric('train_dataset_size', len(train_ds))
        if hasattr(eval_ds, '__len__'):
            mlflow.log_metric('eval_dataset_size', len(eval_ds))
        
        logger.info(f"Starting training with params: {params}")
        
        # Train the model
        results = manager.train(
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate']
        )
        
        # Log training results
        if 'final_train_loss' in results:
            mlflow.log_metric('final_train_loss', results['final_train_loss'])
        
        # Log evaluation metrics
        eval_metrics = results.get('eval_metrics', {})
        for k, v in eval_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"eval_{k}", v)
        
        # Log confusion matrix if available
        if 'confusion_matrix' in results:
            mlflow.log_figure(results['confusion_matrix'], artifact_file='confusion_matrix.png')
        
        # Log the trained model
        mlflow.transformers.log_model(
            transformers_model={
                "model": manager.model,
                "tokenizer": manager.tokenizer
            },
            artifact_path="model",
            run_id=run_id,
            task="text-classification"
        )
        
        logger.info("Training completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def register_model(model_name, run_id, training_results):
    """Register model with error handling"""
    try:
        # Get next version number
        try:
            latest_versions = client.get_latest_versions(model_name)
            max_version = max([int(v.version) for v in latest_versions]) if latest_versions else 0
            next_version = str(max_version + 1)
        except Exception:
            next_version = "1"

        # Prepare tags and metrics
        tags = {
            'model_type': 'sentiment_analysis',
            'base_model': model_name,
            'version': next_version,
            'framework': 'transformers',
            'registered_by': getpass.getuser(),
            'registered_at': dt.now().isoformat()
        }
        
        # Extract metrics safely
        metrics = {}
        if 'final_train_loss' in training_results:
            metrics['final_train_loss'] = training_results['final_train_loss']
        
        eval_metrics = training_results.get('eval_metrics', {})
        for k, v in eval_metrics.items():
            if isinstance(v, (int, float)):
                metrics[k] = v

        # Create or get registered model
        uri = f"runs:/{run_id}/model"
        try:
            client.get_registered_model(model_name)
            logger.info(f"Using existing registered model: {model_name}")
        except Exception:
            logger.info(f"Creating new registered model: {model_name}")
            client.create_registered_model(model_name)

        # Create model version
        mv = client.create_model_version(
            name=model_name,
            source=uri,
            run_id=run_id
        )

        # Set tags
        for k, v in tags.items():
            client.set_model_version_tag(model_name, mv.version, k, str(v))

        # Transition to staging
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage='Staging',
            archive_existing_versions=False
        )

        # Set aliases
        aliases = [f"v{next_version}", next_version, dt.now().strftime("%Y%m%d")]
        for alias in aliases:
            try:
                client.set_registered_model_alias(name=model_name, alias=alias, version=mv.version)
            except Exception as e:
                logger.warning(f"Could not set alias {alias}: {e}")

        logger.info(f"Successfully registered {model_name} v{mv.version}")
        return str(mv.version)
        
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise

# ==== Model promotion functions ====
def promote_to_challenger(model_name: str, version: str, run_id: Optional[str] = None) -> bool:
    """Promote model to challenger alias"""
    try:
        # Remove existing challenger alias if it exists
        try:
            existing_challenger = client.get_model_version_by_alias(model_name, "challenger")
            if existing_challenger:
                client.delete_registered_model_alias(model_name, "challenger")
                logger.info(f"Removed existing challenger alias from v{existing_challenger.version}")
        except mlflow.exceptions.RestException as e:
            if "Registered model alias challenger not found" not in str(e):
                logger.warning(f"Error checking existing challenger: {e}")
        
        # Set new challenger alias
        client.set_registered_model_alias(model_name, "challenger", version)
        
        # Set promotion tags
        timestamp = dt.now().isoformat()
        client.set_model_version_tag(model_name, version, "promotion.challenger_date", timestamp)
        if run_id:
            client.set_model_version_tag(model_name, version, "promotion.challenger_run_id", run_id)
        
        logger.info(f"Successfully promoted {model_name} v{version} to challenger")
        return True
        
    except Exception as e:
        logger.error(f"Failed to promote {model_name} v{version} to challenger: {e}")
        return False

def promote_to_champion(model_name: str, challenger_version: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Promote challenger to champion with proper evaluation and comparison"""
    # Initialize champion comparison data
    champion_comparison = {
        "champion_version": None,
        "champion_f1": None,
        "champion_accuracy": None
    }

    try:
        logger.info(f"Starting champion promotion process for {model_name} v{challenger_version}")
        
        # Get current champion if exists
        champion_model_version = None
        try:
            champion_model_version = client.get_model_version_by_alias(model_name, "champion")
            logger.info(f"Found existing champion: v{champion_model_version.version}")
        except mlflow.exceptions.RestException:
            logger.info("No existing champion found")

        # Load challenger model
        try:
            challenger_mgr = load_model_from_registry(model_name, challenger_version)
            logger.info(f"Loaded challenger model {model_name} v{challenger_version}")
        except Exception as e:
            raise ValueError(f"Failed to load challenger model: {e}")

        # Load evaluation dataset with challenger's tokenizer
        try:
            _, eval_ds = load_and_preprocess_from_blob(
                tokenizer=challenger_mgr.tokenizer,
                label_map=COMMON_SENTIMENT_MAPPINGS.get('text'),
                container_name=container_name
            )

            if eval_ds is None:
                raise ValueError("Failed to load evaluation dataset")
            
            logger.info(f"Loaded evaluation dataset with {len(eval_ds)} samples")
        except Exception as e:
            raise ValueError(f"Failed to prepare evaluation data: {e}")

        # Evaluate challenger
        try:
            challenger_metrics = challenger_mgr.evaluate(eval_ds, batch_size=32)
            challenger_f1 = challenger_metrics.get("f1", 0.0)
            challenger_accuracy = challenger_metrics.get("accuracy", 0.0)

            logger.info(f"Challenger metrics - Accuracy: {challenger_accuracy:.4f}, F1: {challenger_f1:.4f}")

            # Log challenger metrics if run_id provided
            if run_id:
                for k, v in challenger_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"challenger_{k}", v)
        except Exception as e:
            raise ValueError(f"Failed to evaluate challenger: {e}")

        # Compare with champion if exists
        promote_challenger = True  # Default to True if no champion exists

        if champion_model_version:
            champion_version = champion_model_version.version
            logger.info(f"Evaluating existing champion v{champion_version}")

            try:
                # Load champion model
                champion_mgr = load_model_from_registry(model_name, champion_version)

                # Load evaluation dataset with champion's tokenizer (separate evaluation)
                _, champion_eval_ds = load_and_preprocess_from_blob(
                    tokenizer=champion_mgr.tokenizer,
                    label_map=COMMON_SENTIMENT_MAPPINGS.get('text'),
                    container_name=container_name
                )

                if champion_eval_ds is not None:
                    # Evaluate champion
                    champion_metrics = champion_mgr.evaluate(champion_eval_ds, batch_size=32)
                    champion_f1 = champion_metrics.get("f1", 0.0)
                    champion_accuracy = champion_metrics.get("accuracy", 0.0)

                    logger.info(f"Champion metrics - Accuracy: {champion_accuracy:.4f}, F1: {champion_f1:.4f}")

                    # Log champion metrics if run_id provided
                    if run_id:
                        for k, v in champion_metrics.items():
                            if isinstance(v, (int, float)):
                                mlflow.log_metric(f"champion_{k}", v)

                    champion_comparison.update({
                        "champion_version": champion_version,
                        "champion_f1": champion_f1,
                        "champion_accuracy": champion_accuracy
                    })

                    # Compare F1 scores to decide promotion
                    promote_challenger = challenger_f1 > champion_f1

                    logger.info(f"Comparison - Challenger F1: {challenger_f1:.4f} vs Champion F1: {champion_f1:.4f}")
                    logger.info(f"Promote challenger: {promote_challenger}")
                else:
                    logger.warning("Failed to load evaluation dataset for champion, promoting challenger by default")
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate champion: {e}. Promoting challenger by default.")
        else:
            logger.info("No existing champion found, promoting challenger")

        # Promote challenger to champion if better
        promoted = False
        if promote_challenger:
            try:
                if champion_model_version:
                    client.delete_registered_model_alias(model_name, "champion")

                client.set_registered_model_alias(model_name, "champion", challenger_version)

                timestamp = dt.now().isoformat()
                client.set_model_version_tag(model_name, challenger_version, "promotion.champion_date", timestamp)
                if run_id:
                    client.set_model_version_tag(model_name, challenger_version, "promotion.champion_run_id", run_id)

                promoted = True
                logger.info(f"Successfully promoted {model_name} v{challenger_version} to champion")
            except Exception as e:
                logger.error(f"Failed to set champion alias: {e}")
                raise
        else:
            logger.info("Challenger not promoted - champion performs better")

        return {
            "status": "success",
            "promoted": promoted,
            "challenger_version": challenger_version,
            "challenger_f1": challenger_f1,
            "challenger_accuracy": challenger_accuracy,
            "champion_comparison": champion_comparison
        }

    except Exception as e:
        error_msg = f"Failed to promote to champion: {e}"
        logger.error(error_msg)
        return {
            "status": "failed",
            "error": error_msg,
            "promoted": False,
            "challenger_version": challenger_version,
            "challenger_f1": None,
            "challenger_accuracy": None,
            "champion_comparison": champion_comparison
        }

# ==== Core validation function with enhanced error handling ====
def validate_model_performance(
    model_name: str,
    version: Optional[str] = None,
    stage: str = "Staging",
    batch_size: int = 32,
    min_accuracy: float = 0.8,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
    """Validate model performance with comprehensive error handling"""
    try:
        logger.info(f"Starting validation for {model_name} (version={version}, stage={stage})")
        
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
        logger.info(f"Found model version: {model_version}")
        
        # Load model using transformers loader
        try:
            model_mgr = load_model_from_registry(model_name, model_version)
            logger.info(f"Successfully loaded model {model_name} v{model_version}")
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
            logger.info("Loading evaluation dataset...")
            _, eval_ds = load_and_preprocess_from_blob(
                tokenizer=model_mgr.tokenizer,
                label_map=COMMON_SENTIMENT_MAPPINGS.get('text'),
                container_name=container_name
            )
            
            if eval_ds is None:
                raise ValueError("Evaluation dataset is None")
                
            logger.info(f"Successfully loaded evaluation dataset with {len(eval_ds)} samples")
        except Exception as e:
            error_msg = f"Failed to load evaluation dataset: {e}"
            logger.error(error_msg)
            return {
                "model_name": model_name,
                "version": model_version,
                "status": "failed",
                "error": error_msg
            }
        
        # Evaluate model
        try:
            logger.info(f"Starting model evaluation with batch_size={batch_size}")
            model_metrics = model_mgr.evaluate(eval_ds, batch_size)
            
            model_accuracy = model_metrics.get("accuracy")
            model_f1 = model_metrics.get("f1")
            
            if model_accuracy is None or model_f1 is None:
                raise ValueError("Evaluation returned None for accuracy or F1 score")
                
            logger.info(f"Model evaluation completed - Accuracy: {model_accuracy:.4f}, F1: {model_f1:.4f}")
            
        except Exception as e:
            error_msg = f"Model evaluation failed: {e}"
            logger.error(error_msg)
            return {
                "model_name": model_name,
                "version": model_version,
                "status": "failed",
                "error": error_msg
            }
        
        # Log metrics if run_id provided
        if run_id:
            try:
                for k, v in model_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"validation_{k}", v)
            except Exception as e:
                logger.warning(f"Failed to log validation metrics: {e}")
        
        # Determine if validation passed
        validation_passed = model_accuracy >= min_accuracy
        
        # Set validation tags on model version
        try:
            timestamp = dt.now().isoformat()
            if run_id:
                client.set_model_version_tag(model_name, model_version, "validation.run_id", run_id)
            client.set_model_version_tag(model_name, model_version, "validation.date", timestamp)
            client.set_model_version_tag(model_name, model_version, "validation.accuracy", f"{model_accuracy:.4f}")
            client.set_model_version_tag(model_name, model_version, "validation.f1", f"{model_f1:.4f}")
            client.set_model_version_tag(model_name, model_version, "validation.status", "approved" if validation_passed else "rejected")
        except Exception as e:
            logger.warning(f"Failed to set validation tags: {e}")

        logger.info(f"Validation completed - Passed: {validation_passed}")
        
        return {
            "model_name": model_name,
            "version": model_version,
            "status": "success",
            "validation": {
                "passed": validation_passed,
                "accuracy": model_accuracy,
                "f1_score": model_f1,
                "threshold": min_accuracy
            },
            "metrics": model_metrics
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
    """Complete model promotion workflow"""
    logger.info(f"Starting model promotion workflow for {model_name}")
    
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
        logger.warning("Model validation failed or did not pass requirements")
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
        logger.error("Failed to promote to challenger")
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
    import uuid
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load latest model, train, validate and promote")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name of the registered model to load and train")
    parser.add_argument("--base-model-path", type=str, default=None,
                        help="Base model path if no existing model in registry")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate for training")
    parser.add_argument("--experiment-name", type=str, default="model-training-pipeline",
                        help="MLflow experiment name")
    parser.add_argument("--min-accuracy", type=float, default=0.8,
                        help="Minimum accuracy required to promote to challenger")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum sequence length for tokenization")
    
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
    
    run_name = f"{args.model_name}-pipeline-{uuid.uuid4().hex[:8]}"
    print(f"Starting new run: {run_name}")
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        run_id = run.info.run_id
        
        # Log initial parameters
        mlflow.log_params({
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "min_accuracy": args.min_accuracy,
            "max_length": args.max_length
        })
        mlflow.set_tag("pipeline_type", "complete_training_workflow")
        mlflow.set_tag("created_by", getpass.getuser())
        mlflow.set_tag("created_at", dt.now().isoformat())
        
        print(f"Pipeline Parameters:")
        print(f"- Model: {args.model_name}")
        print(f"- Epochs: {args.epochs}")
        print(f"- Learning Rate: {args.learning_rate}")
        print(f"- Batch Size: {args.batch_size}")
        print(f"- Min Accuracy: {args.min_accuracy}")
        print("\n" + "-"*50 + "\n")
        
        try:
            # STEP 1: Load latest model from registry or create new one
            print("STEP 1: Loading latest model from registry...")
            manager = None
            
            try:
                # Try to get the latest model version
                latest_versions = client.get_latest_versions(args.model_name)
                if latest_versions:
                    latest_version = max(latest_versions, key=lambda x: int(x.version))
                    print(f"Found latest model version: {latest_version.version}")
                    manager = load_model_from_registry(args.model_name, latest_version.version)
                    mlflow.set_tag("base_model_source", f"registry_v{latest_version.version}")
                else:
                    raise Exception("No versions found in registry")
                    
            except Exception as e:
                print(f"No existing model found in registry: {e}")
                if args.base_model_path:
                    print(f"Creating new model manager with base model: {args.base_model_path}")
                    manager = ModelManager(model_path=args.base_model_path, finetune=True)
                    mlflow.set_tag("base_model_source", f"new_model_{args.base_model_path}")
                else:
                    raise ValueError("No existing model in registry and no base model path provided. Use --base-model-path to specify a base model.")
            
            if not manager:
                raise ValueError("Failed to load or create model manager")
                
            print("✓ Model loaded successfully")
            
            # STEP 2: Load and preprocess training data
            print("\nSTEP 2: Loading and preprocessing training data...")
            train_dataset, test_dataset = load_and_preprocess_from_blob(
                tokenizer=manager.tokenizer,
                max_length=args.max_length,
                label_map=COMMON_SENTIMENT_MAPPINGS.get('text'),
                container_name=container_name
            )
            
            if not train_dataset or not test_dataset:
                raise ValueError("Failed to load training or test dataset")
                
            print(f"✓ Data loaded - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
            
            # STEP 3: Train the model
            print("\nSTEP 3: Training model...")
            training_params = {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate
            }
            
            training_results = train_model(
                manager=manager,
                run_id=run_id,
                train_ds=train_dataset,
                eval_ds=test_dataset,
                params=training_params
            )
            
            print(f"✓ Training completed - Final loss: {training_results['final_train_loss']:.4f}")
            
            # STEP 4: Register the trained model
            print("\nSTEP 4: Registering trained model...")
            new_version = register_model(args.model_name, run_id, training_results)
            print(f"✓ Model registered as version {new_version}")
            
            # STEP 5: Validate the trained model
            print(f"\nSTEP 5: Validating model performance...")
            validation_result = validate_model_performance(
                model_name=args.model_name,
                version=new_version,
                batch_size=args.batch_size,
                min_accuracy=args.min_accuracy,
                run_id=run_id
            )
            
            if validation_result["status"] != "success":
                print(f"✗ Validation failed: {validation_result.get('error', 'Unknown error')}")
                mlflow.set_tag("pipeline_status", "validation_failed")
                exit(1)
            
            validation_passed = validation_result["validation"]["passed"]
            accuracy = validation_result["validation"]["accuracy"]
            f1_score = validation_result["validation"]["f1_score"]
            
            print(f"✓ Validation completed - Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
            print(f"  Validation passed: {validation_passed} (threshold: {args.min_accuracy})")
            
            # STEP 6: Promote to challenger if validation passed
            if validation_passed:
                print(f"\nSTEP 6: Promoting to challenger...")
                challenger_promoted = promote_to_challenger(args.model_name, new_version, run_id)
                
                if challenger_promoted:
                    print(f"✓ Successfully promoted v{new_version} to challenger")
                    
                    # STEP 7: Promote to champion if F1 score is better
                    print(f"\nSTEP 7: Evaluating for champion promotion...")
                    champion_result = promote_to_champion(args.model_name, new_version, run_id)
                    
                    if champion_result["status"] == "success":
                        if champion_result["promoted"]:
                            print(f"✓ Successfully promoted v{new_version} to champion!")
                            print(f"  New F1: {champion_result['challenger_f1']:.4f}")
                            if champion_result['champion_comparison']['champion_f1']:
                                print(f"  Previous champion F1: {champion_result['champion_comparison']['champion_f1']:.4f}")
                            mlflow.set_tag("pipeline_status", "champion_promoted")
                        else:
                            print(f"○ Model not promoted to champion - current champion performs better")
                            print(f"  Challenger F1: {champion_result['challenger_f1']:.4f}")
                            print(f"  Champion F1: {champion_result['champion_comparison']['champion_f1']:.4f}")
                            mlflow.set_tag("pipeline_status", "challenger_only")
                    else:
                        print(f"✗ Champion promotion failed: {champion_result.get('error', 'Unknown error')}")
                        mlflow.set_tag("pipeline_status", "champion_promotion_failed")
                        
                else:
                    print(f"✗ Failed to promote to challenger")
                    mlflow.set_tag("pipeline_status", "challenger_promotion_failed")
            else:
                print(f"\n✗ Model not promoted - validation failed (accuracy {accuracy:.4f} < {args.min_accuracy})")
                mlflow.set_tag("pipeline_status", "validation_failed")
            
            # STEP 8: Summary
            print(f"\n" + "="*60)
            print(f"PIPELINE SUMMARY")
            print(f"="*60)
            print(f"Model: {args.model_name} v{new_version}")
            print(f"Training Loss: {training_results['final_train_loss']:.4f}")
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(f"Validation F1: {f1_score:.4f}")
            print(f"Validation Passed: {validation_passed}")
            print(f"Challenger Promotion: {'✓' if validation_passed and challenger_promoted else '✗'}")
            
            if validation_passed and challenger_promoted and champion_result:
                print(f"Champion Promotion: {'✓' if champion_result.get('promoted', False) else '○'}")
            
            print(f"Run ID: {run_id}")
            
            # Print run URL
            tracking_uri_clean = mlflow.get_tracking_uri()
            run_url = f"{tracking_uri_clean.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
            print(f"MLflow Run: {run_url}")
            print(f"="*60)
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            print(f"\n✗ {error_msg}")
            logger.error(error_msg)
            mlflow.set_tag("pipeline_status", "failed")
            mlflow.set_tag("error_message", str(e))
            raise
