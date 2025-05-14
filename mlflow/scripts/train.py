# import sys
# import os
# import uuid
# import getpass
# import datetime
# import argparse
# import mlflow
# import logging
# import traceback
# from registry import ModelRegistry
# from pathlib import Path
# from typing import Dict, Any, Optional, Tuple
# import pandas as pd

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# from src.models.preprocess import preprocess_data
# from src.models.transformer_manager import ModelManager
# from config.dataClient import get_blob_storage

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ==== Function to load data ====
# def load_data(texts_file: str, labels_file: str) -> Tuple[list, list]:
#     """Read CSV files and return lists of texts and labels."""
#     try:
#         texts = pd.read_csv(texts_file)['text'].tolist()
#         labels = pd.read_csv(labels_file)['sentiment'].tolist()
#         return texts, labels
#     except Exception as e:
#         logger.error(f"Error loading data: {str(e)}")
#         raise

# # ==== Function to train model ====
# def train_model(
#     model_name: str,
#     version: str, 
#     output_dir: str,
#     train_texts: list,
#     train_labels: list,
#     eval_texts: Optional[list] = None,
#     eval_labels: Optional[list] = None,
#     epochs: int = 3,
#     batch_size: int = 16,
#     learning_rate: float = 5e-5,
#     finetune: bool = True,
#     registry: Optional[ModelRegistry] = None,
#     exp_id: str = None
# ) -> Dict[str, Any]:
#     """Train sentiment analysis model and log to MLflow."""
#     try:
#         # Make sure there are no active runs before starting a new one
#         if mlflow.active_run():
#             logger.info("Ending existing active run before starting a new one")
#             mlflow.end_run()
            
#         # Start MLflow run
#         with mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-training") as run:
#             run_id = run.info.run_id
#             logger.info(f"Started new run with ID: {run_id}")
            
#             # Log parameters
#             mlflow.log_params({
#                 "model_name": model_name,
#                 "epochs": epochs,
#                 "batch_size": batch_size,
#                 "learning_rate": learning_rate,
#                 "finetune": finetune,
#                 "train_samples": len(train_texts),
#                 "eval_samples": len(eval_texts) if eval_texts else 0
#             })

#             # Initialize model manager
#             manager = ModelManager(
#                 model_path=model_name,
#                 output_dir=output_dir,
#                 finetune=finetune
#             )
            
#             # Preprocess training data
#             train_dataset = preprocess_data(manager.tokenizer, train_texts, train_labels)
            
#             # Preprocess evaluation data if available
#             eval_dataset = None
#             if eval_texts and eval_labels:
#                 eval_dataset = preprocess_data(manager.tokenizer, eval_texts, eval_labels)
            
#             # Train model
#             training_results = manager.train(
#                 train_dataset=train_dataset,
#                 eval_dataset=eval_dataset,
#                 epochs=epochs,
#                 batch_size=batch_size,
#                 learning_rate=learning_rate
#             )
            
#             # Log training metrics
#             mlflow.log_metrics({"final_train_loss": training_results["final_train_loss"]})
            
#             # Log evaluation metrics if available
#             if "eval_metrics" in training_results and training_results["eval_metrics"]:
#                 mlflow.log_metrics({
#                     f"eval_{k}": v for k, v in training_results["eval_metrics"].items() 
#                     if k != 'confusion_matrix_fig' and isinstance(v, (int, float))
#                 })
            
#             # Save model
#             try:
#                 blob_storage = get_blob_storage()

#                 model_save_path = manager.save_model(
#                     blob_storage=blob_storage,
#                     version=version,               
#                     metrics={k: v for k, v in training_results.get("eval_metrics", {}).items() 
#                             if k != "confusion_matrix_fig" and isinstance(v, (int, float))},    
#                     cm_fig=training_results.get("confusion_matrix_fig", None)
#                 )
#                 logger.info(f"Model saved at {model_save_path}")
#             except Exception as save_error:
#                 logger.error(f"Error saving model: {save_error}")
#                 logger.debug(traceback.format_exc())

#             # Register model if registry is provided
#             if registry:
#                 try:
#                     # Get current run ID
#                     current_run_id = run.info.run_id
                    
#                     # Create model tags
#                     model_tags = {
#                         "model_type": "sentiment_analysis",
#                         "base_model": model_name,
#                         "version": version,
#                         "framework": "transformers",
#                         "batch_size": str(batch_size),
#                         "registered_by": getpass.getuser(),
#                         "registered_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                     }

#                     # Add evaluation metrics to tags
#                     if "eval_metrics" in training_results:
#                         for k, v in training_results["eval_metrics"].items():
#                             if isinstance(v, (int, float)) and k != "confusion_matrix_fig":
#                                 model_tags[f"metric.{k}"] = str(round(v, 4))

#                     # Register model with tags
#                     model_version = registry.register_model(
#                         run_id=current_run_id,
#                         model_name=model_name,
#                         description=f"Model trained on {len(train_texts)} samples",
#                         metrics={
#                             "final_train_loss": training_results["final_train_loss"],
#                             **{k: v for k, v in training_results.get("eval_metrics", {}).items() 
#                                if k != "confusion_matrix_fig" and isinstance(v, (int, float))}
#                         },
#                         model_manager=manager,
#                         tags=model_tags
#                     )

#                     logger.info(f"Model registered with version: {model_version}")
#                     print(f"Model registered with version: {model_version}")

#                     # Transition to Staging
#                     registry.transition_model_stage(model_name, model_version, "Staging")
#                     logger.info(f"Model {model_name} v{model_version} transitioned to STAGING")
                    
#                     # Define useful aliases
#                     aliases = [
#                         f"v{version}",
#                         version,
#                         datetime.datetime.now().strftime("%Y%m%d")
#                     ]
                    
#                     # Add aliases to the model
#                     registry.add_aliases(model_name, model_version, aliases)
                    
#                 except Exception as reg_error:
#                     logger.error(f"Error during model registration: {reg_error}")
#                     print(f"Model registration failed: {reg_error}")
#                     logger.error(traceback.format_exc())

#             # Return results
#             return {
#                 "run_id": run.info.run_id,
#                 "train_metrics": {"final_train_loss": training_results["final_train_loss"]},
#                 "eval_metrics": {k: v for k, v in training_results.get("eval_metrics", {}).items() 
#                                 if k != "confusion_matrix_fig" and isinstance(v, (int, float))}
#             }
            
#     except Exception as e:
#         logger.error(f"Error training model: {str(e)}")
#         logger.error(traceback.format_exc())
#         # Make sure to end any active run if an exception occurs
#         if mlflow.active_run():
#             mlflow.end_run()
#         raise

# # ==== Function to parse arguments ====
# def parse_args():
#     parser = argparse.ArgumentParser(description="Train a sentiment analysis model")
#     parser.add_argument("--model-name", type=str, required=True, 
#                         help="Name of the model to train")
#     parser.add_argument("--version", type=str, default="v1.0.0",
#                         help="Version tag for this model (used in path and blob storage)")
#     parser.add_argument("--train-texts", type=str, required=True, 
#                         help="Path to training texts file")
#     parser.add_argument("--train-labels", type=str, required=True, 
#                         help="Path to training labels file")
#     parser.add_argument("--eval-texts", type=str, 
#                         help="Path to evaluation texts file")
#     parser.add_argument("--eval-labels", type=str, 
#                         help="Path to evaluation labels file")
#     parser.add_argument("--epochs", type=int, default=3, 
#                         help="Number of training epochs")
#     parser.add_argument("--batch-size", type=int, default=16, 
#                         help="Training batch size")
#     parser.add_argument("--learning-rate", type=float, default=5e-05, 
#                         help="Learning rate")
#     parser.add_argument("--finetune", type=bool, default=True, 
#                         help="Whether to finetune the model")
#     parser.add_argument("--experiment-name", type=str, default="sentiment-analysis-training", 
#                         help="MLflow experiment name")
    
#     return parser.parse_args()

# # ==== Main ====
# if __name__ == "__main__":
#     # First ensure there are no active runs at all
#     try:
#         while mlflow.active_run():
#             logger.info("Ending existing active run")
#             mlflow.end_run()
#     except Exception as e:
#         logger.warning(f"Error when trying to end active runs: {e}")
#         # Force reset MLflow state by clearing all environment variables
#         for env_var in list(os.environ.keys()):
#             if env_var.startswith('MLFLOW_'):
#                 del os.environ[env_var]
    
#     args = parse_args()
#     output_dir = f"artifact/models/{args.model_name}"
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)

#     # Clear MLflow environment variables that could cause conflicts
#     for env_var in ['MLFLOW_RUN_ID', 'MLFLOW_EXPERIMENT_ID', 'MLFLOW_EXPERIMENT_NAME']:
#         if env_var in os.environ:
#             del os.environ[env_var]
    
#     # Set up experiment
#     exp = mlflow.get_experiment_by_name(args.experiment_name)
#     if exp is None:
#         exp_id = mlflow.create_experiment(args.experiment_name)
#     else:
#         exp_id = exp.experiment_id
#     logger.info(f"Using experiment: {args.experiment_name} (ID: {exp_id})")
    
#     # Load data
#     train_texts, train_labels = load_data(args.train_texts, args.train_labels)
#     logger.info(f"Loaded {len(train_texts)} samples for training")
    
#     eval_texts, eval_labels = None, None
#     if args.eval_texts and args.eval_labels:
#         eval_texts, eval_labels = load_data(args.eval_texts, args.eval_labels)
#         logger.info(f"Loaded {len(eval_texts)} samples for evaluation")
    
#     # Initialize registry
#     registry = ModelRegistry()
    
#     try:
#         # Train model
#         results = train_model(
#             model_name=args.model_name,
#             version=args.version,
#             output_dir=output_dir,
#             train_texts=train_texts,
#             train_labels=train_labels,
#             eval_texts=eval_texts,
#             eval_labels=eval_labels,
#             epochs=args.epochs,
#             batch_size=args.batch_size,
#             learning_rate=args.learning_rate,
#             finetune=args.finetune,
#             registry=registry,
#             exp_id=exp_id
#         )
        
#         # Print results to console
#         logger.info(f"Training completed. Run ID: {results['run_id']}")
        
#         # Print training metrics
#         print("=== Training metrics ===")
#         train_metrics = results["train_metrics"]
#         if isinstance(train_metrics, dict):
#             for metric, value in train_metrics.items():
#                 print(f"{metric}: {value:.4f}")
#         else:
#             print(f"loss: {train_metrics:.4f}")
        
#         # Print evaluation metrics if available
#         if results["eval_metrics"]:
#             print("=== Evaluation metrics ===")
#             for metric, value in results["eval_metrics"].items():
#                 if isinstance(value, (int, float)):
#                     print(f"{metric}: {value:.4f}")
    
#     except Exception as e:
#         logger.error(f"Error in main training process: {str(e)}")
#         logger.error(traceback.format_exc())
#         raise
    
#     finally:
#         # End any active run to ensure clean exit
#         if mlflow.active_run():
#             mlflow.end_run()


import sys
import os
import uuid
import getpass
import datetime
import argparse
import mlflow
import logging
import traceback
from registry import ModelRegistry
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.preprocess import preprocess_data
from src.models.transformer_manager import ModelManager
from config.dataClient import get_blob_storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== Function to load data ====
def load_data(texts_file: str, labels_file: str) -> Tuple[list, list]:
    """Read CSV files and return lists of texts and labels."""
    try:
        texts = pd.read_csv(texts_file)['text'].tolist()
        labels = pd.read_csv(labels_file)['sentiment'].tolist()
        return texts, labels
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# ==== Function to train model ====
def train_model(
    model_name: str,
    version: str, 
    output_dir: str,
    train_texts: list,
    train_labels: list,
    eval_texts: Optional[list] = None,
    eval_labels: Optional[list] = None,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    finetune: bool = True,
    registry: Optional[ModelRegistry] = None,
    exp_id: str = None
) -> Dict[str, Any]:
    """Train sentiment analysis model and log to MLflow."""
    try:
        # Make sure there are no active runs before starting a new one
        if mlflow.active_run():
            logger.info("Ending existing active run before starting a new one")
            mlflow.end_run()
            
        # Start MLflow run
        with mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-training") as run:
            run_id = run.info.run_id
            logger.info(f"Started new run with ID: {run_id}")
            
            # Log parameters
            mlflow.log_params({
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "finetune": finetune,
                "train_samples": len(train_texts),
                "eval_samples": len(eval_texts) if eval_texts else 0
            })

            # Initialize model manager
            manager = ModelManager(
                model_path=model_name,
                output_dir=output_dir,
                finetune=finetune
            )
            
            # Preprocess training data
            train_dataset = preprocess_data(manager.tokenizer, train_texts, train_labels)
            
            # Preprocess evaluation data if available
            eval_dataset = None
            if eval_texts and eval_labels:
                eval_dataset = preprocess_data(manager.tokenizer, eval_texts, eval_labels)
            
            # Train model
            training_results = manager.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # Log training metrics
            mlflow.log_metrics({"final_train_loss": training_results["final_train_loss"]})
            
            # Log evaluation metrics if available
            if "eval_metrics" in training_results and training_results["eval_metrics"]:
                mlflow.log_metrics({
                    f"eval_{k}": v for k, v in training_results["eval_metrics"].items() 
                    if k != 'confusion_matrix_fig' and isinstance(v, (int, float))
                })
            
            # Register model if registry is provided
            if registry:
                try:
                    # Get current run ID
                    current_run_id = run.info.run_id
                    
                    # Create model tags
                    model_tags = {
                        "model_type": "sentiment_analysis",
                        "base_model": model_name,
                        "version": version,
                        "framework": "transformers",
                        "batch_size": str(batch_size),
                        "registered_by": getpass.getuser(),
                        "registered_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # Add evaluation metrics to tags
                    if "eval_metrics" in training_results:
                        for k, v in training_results["eval_metrics"].items():
                            if isinstance(v, (int, float)) and k != "confusion_matrix_fig":
                                model_tags[f"metric.{k}"] = str(round(v, 4))

                    # Register model with tags - registry will automatically handle artifact storage
                    model_version = registry.register_model(
                        run_id=current_run_id,
                        model_name=model_name,
                        description=f"Model trained on {len(train_texts)} samples",
                        metrics={
                            "final_train_loss": training_results["final_train_loss"],
                            **{k: v for k, v in training_results.get("eval_metrics", {}).items() 
                               if k != "confusion_matrix_fig" and isinstance(v, (int, float))}
                        },
                        model_manager=manager,  # The registry will automatically save artifacts from this manager
                        tags=model_tags
                    )

                    logger.info(f"Model registered with version: {model_version}")
                    print(f"Model registered with version: {model_version}")

                    # Transition to Staging
                    registry.transition_model_stage(model_name, model_version, "Staging")
                    logger.info(f"Model {model_name} v{model_version} transitioned to STAGING")
                    
                    # Define useful aliases
                    aliases = [
                        f"v{version}",
                        version,
                        datetime.datetime.now().strftime("%Y%m%d")
                    ]
                    
                    # Add aliases to the model
                    registry.add_aliases(model_name, model_version, aliases)
                    
                except Exception as reg_error:
                    logger.error(f"Error during model registration: {reg_error}")
                    print(f"Model registration failed: {reg_error}")
                    logger.error(traceback.format_exc())

            # Return results
            return {
                "run_id": run.info.run_id,
                "train_metrics": {"final_train_loss": training_results["final_train_loss"]},
                "eval_metrics": {k: v for k, v in training_results.get("eval_metrics", {}).items() 
                                if k != "confusion_matrix_fig" and isinstance(v, (int, float))}
            }
            
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.error(traceback.format_exc())
        # Make sure to end any active run if an exception occurs
        if mlflow.active_run():
            mlflow.end_run()
        raise

# ==== Function to parse arguments ====
def parse_args():
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model")
    parser.add_argument("--model-name", type=str, required=True, 
                        help="Name of the model to train")
    parser.add_argument("--version", type=str, default="v1.0.0",
                        help="Version tag for this model (used in path and blob storage)")
    parser.add_argument("--train-texts", type=str, required=True, 
                        help="Path to training texts file")
    parser.add_argument("--train-labels", type=str, required=True, 
                        help="Path to training labels file")
    parser.add_argument("--eval-texts", type=str, 
                        help="Path to evaluation texts file")
    parser.add_argument("--eval-labels", type=str, 
                        help="Path to evaluation labels file")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, 
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-05, 
                        help="Learning rate")
    parser.add_argument("--finetune", type=bool, default=True, 
                        help="Whether to finetune the model")
    parser.add_argument("--experiment-name", type=str, default="sentiment-analysis-training", 
                        help="MLflow experiment name")
    
    return parser.parse_args()

# ==== Main ====
if __name__ == "__main__":
    # First ensure there are no active runs at all
    try:
        while mlflow.active_run():
            logger.info("Ending existing active run")
            mlflow.end_run()
    except Exception as e:
        logger.warning(f"Error when trying to end active runs: {e}")
        # Force reset MLflow state by clearing all environment variables
        for env_var in list(os.environ.keys()):
            if env_var.startswith('MLFLOW_'):
                del os.environ[env_var]
    
    args = parse_args()
    output_dir = f"artifact/models/{args.model_name}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Clear MLflow environment variables that could cause conflicts
    for env_var in ['MLFLOW_RUN_ID', 'MLFLOW_EXPERIMENT_ID', 'MLFLOW_EXPERIMENT_NAME']:
        if env_var in os.environ:
            del os.environ[env_var]
    
    # Set up experiment
    exp = mlflow.get_experiment_by_name(args.experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(args.experiment_name)
    else:
        exp_id = exp.experiment_id
    logger.info(f"Using experiment: {args.experiment_name} (ID: {exp_id})")
    
    # Load data
    train_texts, train_labels = load_data(args.train_texts, args.train_labels)
    logger.info(f"Loaded {len(train_texts)} samples for training")
    
    eval_texts, eval_labels = None, None
    if args.eval_texts and args.eval_labels:
        eval_texts, eval_labels = load_data(args.eval_texts, args.eval_labels)
        logger.info(f"Loaded {len(eval_texts)} samples for evaluation")
    
    # Initialize registry with automatic upload configuration
    registry = ModelRegistry()
    
    try:
        # Train model
        results = train_model(
            model_name=args.model_name,
            version=args.version,
            output_dir=output_dir,
            train_texts=train_texts,
            train_labels=train_labels,
            eval_texts=eval_texts,
            eval_labels=eval_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            finetune=args.finetune,
            registry=registry,
            exp_id=exp_id
        )
        
        # Print results to console
        logger.info(f"Training completed. Run ID: {results['run_id']}")
        
        # Print training metrics
        print("=== Training metrics ===")
        train_metrics = results["train_metrics"]
        if isinstance(train_metrics, dict):
            for metric, value in train_metrics.items():
                print(f"{metric}: {value:.4f}")
        else:
            print(f"loss: {train_metrics:.4f}")
        
        # Print evaluation metrics if available
        if results["eval_metrics"]:
            print("=== Evaluation metrics ===")
            for metric, value in results["eval_metrics"].items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.4f}")
    
    except Exception as e:
        logger.error(f"Error in main training process: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    finally:
        # End any active run to ensure clean exit
        if mlflow.active_run():
            mlflow.end_run()