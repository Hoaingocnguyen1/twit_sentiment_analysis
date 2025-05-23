import logging
import os
import sys
import uuid
from datetime import datetime as dt
from mlflow.tracking import MlflowClient
import mlflow
from config.dataClient import get_blob_storage
from src.models.preprocess import preprocess_data, COMMON_SENTIMENT_MAPPINGS
import pandas as pd
import io
from src.models.transformer_manager import ModelManager

# Setup logging
logger = logging.getLogger(__name__)


def setup_environment(**context):
    """Setup environment for ML pipeline"""
    logger.info("Setting up environment for ML pipeline")
    
    # Add project root to Python path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    sys.path.append(root_dir)
    
    # Validate environment variables
    required_env_vars = [
        'LAKE_STORAGE_CONN_STR',
        'AZURE_BLOB_CONTAINER',
        'TRACKING_URI'
    ]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    # Generate unique run identifier
    run_uuid = uuid.uuid4().hex[:8]
    context['task_instance'].xcom_push(key='run_uuid', value=run_uuid)
    
    # Set MLflow tracking URI
    tracking_uri = os.getenv("TRACKING_URI", "http://52.163.185.16:5000/")
    mlflow.set_tracking_uri(tracking_uri)
    
    logger.info(f"Environment setup complete. Run UUID: {run_uuid}")
    return {"status": "success", "run_uuid": run_uuid}


def load_champion_model(model_name, **context):
    """Load champion model from registry"""
    logger.info(f"Loading CHAMPION model: {model_name}")
    
    try:
        client = MlflowClient()
        
        # Try to get champion alias
        try:
            champion_version = client.get_model_version_by_alias(model_name, "champion")
            model_version = champion_version.version
            logger.info(f"Found champion model version: {model_version}")
        except Exception as e:
            logger.warning(f"No champion model found: {e}")
            # Fallback to latest version
            latest_versions = client.get_latest_versions(model_name)
            if not latest_versions:
                raise ValueError("No model versions found in registry")
            model_version = max(latest_versions, key=lambda x: int(x.version)).version
            logger.info(f"Using latest version as champion: {model_version}")
        
        # Load model
        model_uri = f"models:/{model_name}/{model_version}"
        loaded_model = mlflow.transformers.load_model(model_uri)
        
        # Store model info in XCom
        context['task_instance'].xcom_push(key='champion_version', value=model_version)
        context['task_instance'].xcom_push(key='champion_model_uri', value=model_uri)
        context['task_instance'].xcom_push(key='champion_loaded', value=True)
        
        logger.info(f"Champion model loaded successfully: v{model_version}")
        return {"status": "success", "version": model_version, "model_uri": model_uri}
        
    except Exception as e:
        logger.error(f"Failed to load champion model: {e}")
        context['task_instance'].xcom_push(key='champion_loaded', value=False)
        raise


def evaluate_champion_model(model_name, min_accuracy_threshold=0.85, **context):
    """Evaluate champion model performance"""
    logger.info(f"Evaluating champion model: {model_name}")
    
    try:
        # Get champion model info from previous task
        champion_version = context['task_instance'].xcom_pull(key='champion_version')
        champion_loaded = context['task_instance'].xcom_pull(key='champion_loaded')
        
        if not champion_loaded:
            logger.warning("Champion model not loaded, proceeding with training")
            context['task_instance'].xcom_push(key='should_train', value=True)
            context['task_instance'].xcom_push(key='champion_accuracy', value=0.0)
            return {"status": "no_champion", "should_train": True}
        
        # Load model for evaluation
        model_mgr = load_model_from_registry(model_name, champion_version)
        
        # Load evaluation dataset
        container_name = os.getenv("AZURE_BLOB_CONTAINER", "artifact")
        _, eval_ds = load_evaluation_dataset(model_mgr.tokenizer, container_name)
        
        # Evaluate model
        logger.info("Starting champion model evaluation...")
        eval_metrics = model_mgr.evaluate(eval_ds, batch_size=16)
        
        champion_accuracy = eval_metrics.get("accuracy", 0.0)
        champion_f1 = eval_metrics.get("f1", 0.0)
        
        logger.info(f"Champion evaluation - Accuracy: {champion_accuracy:.4f}, F1: {champion_f1:.4f}")
        
        # Decide if training is needed
        should_train = champion_accuracy < min_accuracy_threshold
        
        # Store results in XCom
        context['task_instance'].xcom_push(key='champion_accuracy', value=champion_accuracy)
        context['task_instance'].xcom_push(key='champion_f1', value=champion_f1)
        context['task_instance'].xcom_push(key='should_train', value=should_train)
        context['task_instance'].xcom_push(key='accuracy_threshold', value=min_accuracy_threshold)
        
        if should_train:
            logger.info(f"Champion accuracy {champion_accuracy:.4f} < threshold {min_accuracy_threshold}, training new model")
        else:
            logger.info(f"Champion accuracy {champion_accuracy:.4f} >= threshold {min_accuracy_threshold}, skipping training")
        
        return {
            "status": "success",
            "champion_accuracy": champion_accuracy,
            "champion_f1": champion_f1,
            "should_train": should_train,
            "threshold": min_accuracy_threshold
        }
        
    except Exception as e:
        logger.error(f"Champion evaluation failed: {e}")
        # If evaluation fails, proceed with training
        context['task_instance'].xcom_push(key='should_train', value=True)
        context['task_instance'].xcom_push(key='champion_accuracy', value=0.0)
        return {"status": "evaluation_failed", "should_train": True, "error": str(e)}


def check_training_decision(**context):
    """Check if training should proceed based on champion evaluation"""
    should_train = context['task_instance'].xcom_pull(key='should_train')
    champion_accuracy = context['task_instance'].xcom_pull(key='champion_accuracy')
    
    logger.info(f"Training decision check - Should train: {should_train}, Champion accuracy: {champion_accuracy}")
    
    if not should_train:
        logger.info("Skipping training - champion model meets performance threshold")
        return "skip_training"
    else:
        logger.info("Proceeding with training - champion model below threshold")
        return "proceed_training"


def load_and_preprocess_data(**context):
    """Load and preprocess training data"""
    logger.info("Loading and preprocessing training data")
    
    try:
        blob = get_blob_storage()
        container_name = os.getenv("AZURE_BLOB_CONTAINER", "artifact")
        
        # Load train and test data
        train_json_str = blob.read_json_from_container(container_name, "data/train_dataset.json")
        test_json_str = blob.read_json_from_container(container_name, "data/test_dataset.json")
        
        # Parse JSON data
        train_df = pd.DataFrame(train_json_str) if isinstance(train_json_str, list) else pd.read_json(io.StringIO(train_json_str), lines=True)
        test_df = pd.DataFrame(test_json_str) if isinstance(test_json_str, list) else pd.read_json(io.StringIO(test_json_str), lines=True)
        
        # Validate data
        train_size, test_size = len(train_df), len(test_df)
        if train_size == 0 or test_size == 0:
            raise ValueError("Empty dataset detected")
        
        # Store data info in XCom
        context['task_instance'].xcom_push(key='train_size', value=train_size)
        context['task_instance'].xcom_push(key='test_size', value=test_size)
        context['task_instance'].xcom_push(key='data_loaded', value=True)
        
        logger.info(f"Data loaded successfully - Train: {train_size}, Test: {test_size}")
        return {"status": "success", "train_size": train_size, "test_size": test_size}
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        context['task_instance'].xcom_push(key='data_loaded', value=False)
        raise


def train_challenger_model(model_name, training_params=None, **context):
    """Train new challenger model"""
    logger.info(f"Training challenger model: {model_name}")
    
    # Default training parameters
    default_params = {
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'warmup_steps': 500,
        'weight_decay': 0.01
    }
    params = {**default_params, **(training_params or {})}
    
    try:
        # Start MLflow run
        run_uuid = context['task_instance'].xcom_pull(key='run_uuid')
        with mlflow.start_run(run_name=f"challenger_training_{run_uuid}") as run:
            run_id = run.info.run_id
            context['task_instance'].xcom_push(key='challenger_run_id', value=run_id)
            
            # Load and preprocess data
            container_name = os.getenv("AZURE_BLOB_CONTAINER", "artifact")
            train_ds, eval_ds = load_training_datasets(params, container_name)
            
            # Initialize model manager for training
            model_manager = ModelManager(
                model_path="distilbert-base-uncased",  # Base model for fine-tuning
                finetune=True
            )
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_metric('train_dataset_size', len(train_ds))
            mlflow.log_metric('eval_dataset_size', len(eval_ds))
            
            # Train the model
            logger.info(f"Starting training with params: {params}")
            results = model_manager.train(
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate']
            )
            
            # Log training results
            final_train_loss = results.get('final_train_loss', 0.0)
            mlflow.log_metric('final_train_loss', final_train_loss)
            
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
                    "model": model_manager.model,
                    "tokenizer": model_manager.tokenizer
                },
                artifact_path="model",
                task="text-classification"
            )
            
            # Store training results in XCom
            context['task_instance'].xcom_push(key='final_train_loss', value=final_train_loss)
            context['task_instance'].xcom_push(key='eval_metrics', value=eval_metrics)
            context['task_instance'].xcom_push(key='training_completed', value=True)
            
            logger.info(f"Training completed successfully - Run ID: {run_id}")
            return {
                "status": "success",
                "run_id": run_id,
                "final_train_loss": final_train_loss,
                "eval_metrics": eval_metrics
            }
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        context['task_instance'].xcom_push(key='training_completed', value=False)
        raise


def register_challenger_model(model_name, **context):
    """Register the trained challenger model"""
    logger.info("Registering challenger model")
    
    try:
        run_id = context['task_instance'].xcom_pull(key='challenger_run_id')
        if not run_id:
            raise ValueError("No challenger run ID found")
        
        client = MlflowClient()
        
        # Get next version number
        try:
            latest_versions = client.get_latest_versions(model_name)
            max_version = max([int(v.version) for v in latest_versions]) if latest_versions else 0
            next_version = str(max_version + 1)
        except Exception:
            next_version = "1"
        
        # Create or get registered model
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(model_name)
        
        # Create model version
        uri = f"runs:/{run_id}/model"
        mv = client.create_model_version(
            name=model_name,
            source=uri,
            run_id=run_id
        )
        
        # Set tags
        timestamp = dt.now().isoformat()
        tags = {
            'model_type': 'sentiment_analysis',
            'framework': 'transformers',
            'registered_at': timestamp,
            'role': 'challenger',
            'airflow_dag': 'ml_training_pipeline'
        }
        
        for k, v in tags.items():
            client.set_model_version_tag(model_name, mv.version, k, str(v))
        
        # Transition to staging
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage='Staging',
            archive_existing_versions=False
        )
        
        context['task_instance'].xcom_push(key='challenger_version', value=mv.version)
        
        logger.info(f"Challenger model registered successfully as version {mv.version}")
        return {"status": "success", "version": mv.version}
        
    except Exception as e:
        logger.error(f"Challenger model registration failed: {e}")
        raise


def validate_challenger_model(model_name, **context):
    """Validate challenger model performance"""
    logger.info("Validating challenger model performance")
    
    try:
        challenger_version = context['task_instance'].xcom_pull(key='challenger_version')
        if not challenger_version:
            raise ValueError("No challenger version found")
        
        # Load challenger model
        model_mgr = load_model_from_registry(model_name, challenger_version)
        
        # Load evaluation dataset
        container_name = os.getenv("AZURE_BLOB_CONTAINER", "artifact")
        _, eval_ds = load_evaluation_dataset(model_mgr.tokenizer, container_name)
        
        # Evaluate challenger
        logger.info("Starting challenger model evaluation...")
        eval_metrics = model_mgr.evaluate(eval_ds, batch_size=16)
        
        challenger_accuracy = eval_metrics.get("accuracy", 0.0)
        challenger_f1 = eval_metrics.get("f1", 0.0)
        
        # Compare with champion
        champion_accuracy = context['task_instance'].xcom_pull(key='champion_accuracy') or 0.0
        
        # Determine if challenger should be promoted
        should_promote = challenger_accuracy > champion_accuracy
        improvement = challenger_accuracy - champion_accuracy
        
        logger.info(f"Challenger validation - Accuracy: {challenger_accuracy:.4f}, F1: {challenger_f1:.4f}")
        logger.info(f"Improvement over champion: {improvement:.4f}")
        
        # Store validation results
        context['task_instance'].xcom_push(key='challenger_accuracy', value=challenger_accuracy)
        context['task_instance'].xcom_push(key='challenger_f1', value=challenger_f1)
        context['task_instance'].xcom_push(key='should_promote', value=should_promote)
        context['task_instance'].xcom_push(key='accuracy_improvement', value=improvement)
        
        # Log validation metrics to MLflow if run exists
        run_id = context['task_instance'].xcom_pull(key='challenger_run_id')
        if run_id:
            try:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metric("validation_accuracy", challenger_accuracy)
                    mlflow.log_metric("validation_f1", challenger_f1)
                    mlflow.log_metric("accuracy_improvement", improvement)
            except Exception as e:
                logger.warning(f"Failed to log validation metrics: {e}")
        
        if should_promote:
            logger.info("Challenger outperforms champion - will promote to champion")
        else:
            logger.info("Challenger does not outperform champion - keeping current champion")
        
        return {
            "status": "success",
            "challenger_accuracy": challenger_accuracy,
            "challenger_f1": challenger_f1,
            "should_promote": should_promote,
            "improvement": improvement
        }
        
    except Exception as e:
        logger.error(f"Challenger validation failed: {e}")
        context['task_instance'].xcom_push(key='should_promote', value=False)
        raise


def promote_challenger_to_champion(model_name, **context):
    """Promote challenger to champion if validation passed"""
    logger.info("Checking champion promotion")
    
    try:
        should_promote = context['task_instance'].xcom_pull(key='should_promote')
        challenger_version = context['task_instance'].xcom_pull(key='challenger_version')
        
        if not should_promote:
            logger.info("Skipping promotion - challenger did not outperform champion")
            return {"status": "skipped", "promoted": False}
        
        if not challenger_version:
            raise ValueError("No challenger version found for promotion")
        
        client = MlflowClient()
        run_id = context['task_instance'].xcom_pull(key='challenger_run_id')
        
        # Remove existing champion alias if it exists
        try:
            existing_champion = client.get_model_version_by_alias(model_name, "champion")
            if existing_champion:
                client.delete_registered_model_alias(model_name, "champion")
                logger.info(f"Removed champion alias from version {existing_champion.version}")
        except Exception as e:
            logger.info(f"No existing champion alias to remove: {e}")
        
        # Set new champion alias
        client.set_registered_model_alias(model_name, "champion", challenger_version)
        
        # Transition to production
        client.transition_model_version_stage(
            name=model_name,
            version=challenger_version,
            stage='Production',
            archive_existing_versions=False
        )
        
        # Set promotion tags
        timestamp = dt.now().isoformat()
        promotion_tags = {
            "promotion.champion_date": timestamp,
            "promotion.champion_run_id": run_id or "unknown",
            "role": "champion"
        }
        
        for k, v in promotion_tags.items():
            client.set_model_version_tag(model_name, challenger_version, k, str(v))
        
        logger.info(f"Successfully promoted challenger v{challenger_version} to champion")
        
        return {
            "status": "success",
            "promoted": True,
            "new_champion_version": challenger_version
        }
        
    except Exception as e:
        logger.error(f"Champion promotion failed: {e}")
        raise

# Helper functions

def load_model_from_registry(model_name: str, version: str):
    """Load model and tokenizer from registry using mlflow.transformers.load_model"""
    try:
        model_uri = f"models:/{model_name}/{version}"
        logger.info(f"Loading model from URI: {model_uri}")
        
        # Load using mlflow.transformers.load_model
        loaded = mlflow.transformers.load_model(model_uri)
        
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


def load_evaluation_dataset(tokenizer, container_name):
    """Load and preprocess evaluation dataset"""
    try:
        blob = get_blob_storage()
        test_json_str = blob.read_json_from_container(container_name, "data/test_dataset.json")
        
        # Parse JSON data
        test_df = pd.DataFrame(test_json_str) if isinstance(test_json_str, list) else pd.read_json(io.StringIO(test_json_str), lines=True)
        
        # Preprocess data
        _, eval_ds = preprocess_data(
            train_df=pd.DataFrame(),  # Empty for eval-only
            test_df=test_df,
            tokenizer=tokenizer,
            label_map=COMMON_SENTIMENT_MAPPINGS.get('text'),
            max_length=128
        )
        
        return None, eval_ds
        
    except Exception as e:
        logger.error(f"Failed to load evaluation dataset: {e}")
        raise


def load_training_datasets(params, container_name):
    """Load and preprocess training datasets"""
    try:
        blob = get_blob_storage()
        
        # Load data
        train_json_str = blob.read_json_from_container(container_name, "data/train_dataset.json")
        test_json_str = blob.read_json_from_container(container_name, "data/test_dataset.json")
        
        # Parse JSON data
        train_df = pd.DataFrame(train_json_str) if isinstance(train_json_str, list) else pd.read_json(io.StringIO(train_json_str), lines=True)
        test_df = pd.DataFrame(test_json_str) if isinstance(test_json_str, list) else pd.read_json(io.StringIO(test_json_str), lines=True)
        
        # Initialize tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Preprocess data
        train_ds, eval_ds = preprocess_data(
            train_df=train_df,
            test_df=test_df,
            tokenizer=tokenizer,
            label_map=COMMON_SENTIMENT_MAPPINGS.get('text'),
            max_length=128
        )
        
        return train_ds, eval_ds
        
    except Exception as e:
        logger.error(f"Failed to load training datasets: {e}")
        raise


def cleanup_pipeline(**context):
    """Cleanup pipeline resources and log final results"""
    logger.info("Cleaning up pipeline resources")
    
    try:
        # Get final results
        should_train = context['task_instance'].xcom_pull(key='should_train')
        champion_accuracy = context['task_instance'].xcom_pull(key='champion_accuracy') or 0.0
        
        results = {
            "pipeline_status": "completed",
            "champion_accuracy": champion_accuracy,
            "training_executed": should_train
        }
        
        if should_train:
            challenger_accuracy = context['task_instance'].xcom_pull(key='challenger_accuracy') or 0.0
            should_promote = context['task_instance'].xcom_pull(key='should_promote') or False
            challenger_version = context['task_instance'].xcom_pull(key='challenger_version')
            
            results.update({
                "challenger_accuracy": challenger_accuracy,
                "challenger_version": challenger_version,
                "promoted_to_champion": should_promote,
                "accuracy_improvement": challenger_accuracy - champion_accuracy
            })
        
        logger.info(f"Pipeline completed successfully: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Pipeline cleanup failed: {e}")
        return {"pipeline_status": "completed_with_errors", "error": str(e)}