# import sys
# import os
# import logging
# import argparse
# import pandas as pd
# import mlflow
# import matplotlib.pyplot as plt
# from datetime import datetime
# from registry import ModelRegistry

# # Add path to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# # Internal imports
# from src.models.preprocess import preprocess_data, COMMON_SENTIMENT_MAPPINGS

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ==== Load data function ====
# def load_data(texts_file: str, labels_file: str):
#     """Read CSV file and return lists of texts and labels."""
#     try:
#         texts = pd.read_csv(texts_file)['text'].tolist()
#         labels = pd.read_csv(labels_file)['sentiment'].tolist()
#         return texts, labels
#     except Exception as e:
#         logger.error(f"Error loading data: {e}")
#         raise

# # ==== Parse arguments function ====
# def parse_args():
#     parser = argparse.ArgumentParser(description="Validate a registered model")
#     parser.add_argument("--model-name", type=str, required=True,
#                         help="Name of the registered model to validate")
#     parser.add_argument("--version", type=str, default=None,
#                         help="Version of the model to validate (optional if stage is provided)")
#     parser.add_argument("--stage", type=str, default="Staging",
#                         help="Stage of the model to validate (used if version is not provided)")
#     parser.add_argument("--eval-texts", type=str, required=True,
#                         help="Path to the file containing evaluation texts")
#     parser.add_argument("--eval-labels", type=str, required=True,
#                         help="Path to the file containing evaluation labels")
#     parser.add_argument("--batch-size", type=int, default=32,
#                         help="Batch size for evaluation")
#     parser.add_argument("--experiment-name", type=str, default="model-validation",
#                         help="MLflow experiment name")
#     parser.add_argument("--promote-to-prod", type=bool, default=True,
#                         help="Promote model to Production if validation passes")
#     parser.add_argument("--min-accuracy", type=float, default=0.7,
#                         help="Minimum accuracy required to pass validation")
#     return parser.parse_args()


# def ensure_no_active_run():
#     """Ensure there's no active MLflow run and clean environment variables using multiple methods."""
#     # Try conventional approach
#     try:
#         if mlflow.active_run():
#             active_info = mlflow.active_run().info.run_id
#             logger.info(f"Ending existing active run: {active_info}")
#             mlflow.end_run()
#     except Exception as e:
#         logger.warning(f"Error when trying to end active run: {e}")
    
#     # Try more aggressive approach - Access mlflow internals to reset state
#     # This is a fallback method when the regular approach doesn't work
#     try:
#         # Force reset the active run stack in MLflow's internal state
#         # This is using internal API which might change, but helps in stubborn cases
#         if hasattr(mlflow, '_active_run_stack'):
#             logger.info("Forcibly clearing MLflow's internal run stack")
#             if hasattr(mlflow._active_run_stack, 'clear'):
#                 mlflow._active_run_stack.clear()
#             else:
#                 while len(mlflow._active_run_stack) > 0:
#                     mlflow._active_run_stack.pop()
#     except Exception as e:
#         logger.warning(f"Error when trying to reset MLflow's internal state: {e}")
    
#     # Clean MLflow environment variables that might cause conflicts
#     mlflow_env_vars = ['MLFLOW_RUN_ID', 'MLFLOW_EXPERIMENT_ID', 'MLFLOW_EXPERIMENT_NAME', 
#                       'MLFLOW_TRACKING_URI', 'MLFLOW_FLATTEN_PARAMS']
#     for var in mlflow_env_vars:
#         if var in os.environ:
#             logger.info(f"Removing environment variable: {var}")
#             del os.environ[var]
            
#     # Final verification - this should now be None
#     if mlflow.active_run():
#         logger.warning("ATTENTION: Still have an active run after cleanup attempts!")


# def validate_model(model_name, version, stage, eval_texts, eval_labels, batch_size, 
#                   experiment_name, promote_to_prod=False, min_accuracy=0.7):
#     """
#     Validate a model from the registry and optionally promote it to Production
#     if it meets the validation criteria.
#     """
#     # Ensure no active runs before starting a new one
#     ensure_no_active_run()
    
#     # Force any active run to end one more time before proceeding
#     try:
#         if mlflow.active_run():
#             mlflow.end_run()
#     except:
#         pass
    
#     # Create MLflow experiment if it doesn't exist
#     exp = mlflow.get_experiment_by_name(experiment_name)
#     if exp is None:
#         exp_id = mlflow.create_experiment(experiment_name)
#     else:
#         exp_id = exp.experiment_id
    
#     logger.info(f"Using experiment: {experiment_name} (ID: {exp_id})")

#     # Load evaluation data
#     logger.info(f"Loading evaluation data from {eval_texts} and {eval_labels}")
#     texts, labels = load_data(eval_texts, eval_labels)
#     logger.info(f"Loaded {len(texts)} samples for validation")

#     # Initialize ModelRegistry
#     logger.info("Initializing model registry")
#     registry = ModelRegistry()

#     # Try the validation process without MLflow tracking first if needed
#     # This is a fallback approach in case MLflow is causing issues
#     if os.environ.get('BYPASS_MLFLOW_TRACKING', '').lower() == 'true':
#         logger.info("BYPASS_MLFLOW_TRACKING is enabled, running without MLflow tracking")
#         return _perform_validation_without_mlflow(
#             registry, model_name, version, stage, texts, labels, 
#             batch_size, min_accuracy, promote_to_prod
#         )
        
#     # Normal flow with MLflow tracking
#     try:
#         # Use a completely fresh MLflow instance by resetting internal tracking
#         # mlflow.set_tracking_uri(mlflow.get_tracking_uri())
        
#         # Try with nested=True as a fallback option if regular run creation fails
#         try:
#             logger.info("Attempting to start MLflow run")
#             run = mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-validation")
#         except Exception as e:
#             logger.warning(f"Could not create regular run, trying nested run: {e}")
#             run = mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-validation", nested=True)
            
#         run_id = run.info.run_id
#         logger.info(f"Successfully started validation run with ID: {run_id}")
        
#         try:
#             # Log parameters
#             mlflow.log_params({
#                 "model_name": model_name,
#                 "version": version if version else "N/A",
#                 "stage": stage,
#                 "batch_size": batch_size,
#                 "eval_texts": eval_texts,
#                 "eval_labels": eval_labels,
#                 "min_accuracy": min_accuracy
#             })

#             # Load model from registry
#             logger.info(f"Loading model {model_name} (version={version}, stage={stage})")
#             manager = registry.load_model(
#                 model_name=model_name,
#                 version=version,
#                 stage=stage if not version else None
#             )
            
#             # Get the actual version that was loaded
#             if not version:
#                 try:
#                     version = registry.get_latest_version(model_name)
#                     logger.info(f"Using latest version: {version}")
#                 except Exception as e:
#                     logger.warning(f"Could not determine model version: {e}")
#                     version = "unknown"

#             # Preprocess evaluation data
#             logger.info("Preprocessing evaluation data")
#             dataset = preprocess_data(
#                 manager.tokenizer, 
#                 texts, 
#                 labels, 
#                 label_map=COMMON_SENTIMENT_MAPPINGS["text"]
#             )
            
#             # Evaluate model
#             logger.info("Evaluating model")
#             metrics = manager.evaluate(dataset)
            
#             # Log metrics
#             for k, v in metrics.items():
#                 if isinstance(v, (int, float)) and k != "confusion_matrix_fig":
#                     mlflow.log_metric(k, v)
#                     logger.info(f"Metric {k}: {v:.4f}")

#             # Log confusion matrix if available
#             if "confusion_matrix_fig" in metrics and hasattr(metrics["confusion_matrix_fig"], "savefig"):
#                 cm_path = f"/tmp/confusion_matrix_{model_name}_v{version}.png"
#                 metrics["confusion_matrix_fig"].savefig(cm_path)
#                 plt.close(metrics["confusion_matrix_fig"])
#                 mlflow.log_artifact(cm_path)
#                 logger.info(f"Confusion matrix saved to {cm_path} and logged to MLflow")
            
#             # Determine if validation passes
#             validation_passed = metrics.get("accuracy", 0) >= min_accuracy
#             mlflow.log_metric("validation_passed", 1 if validation_passed else 0)
            
#             # Add validation result as a tag to the model version
#             validation_tag = {
#                 "validation.passed": str(validation_passed),
#                 "validation.accuracy": str(round(metrics.get("accuracy", 0), 4)),
#                 "validation.run_id": run_id,
#                 "validation.date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             }
            
#             # Add validation tags to the model version
#             for tag_key, tag_value in validation_tag.items():
#                 registry.client.set_model_version_tag(
#                     name=model_name,
#                     version=version,
#                     key=tag_key,
#                     value=tag_value
#                 )
            
#             logger.info(f"Validation {'passed' if validation_passed else 'failed'} "
#                        f"(accuracy: {metrics.get('accuracy', 0):.4f}, threshold: {min_accuracy})")
            
#             # Promote to Production if validation passes and promotion is requested
#             if validation_passed and promote_to_prod:
#                 logger.info(f"Promoting model {model_name} v{version} to Production")
#                 registry.transition_model_stage(model_name, version, "Production")
#                 logger.info(f"Model {model_name} v{version} promoted to PRODUCTION")
                
#                 # Add production promotion tag
#                 registry.client.set_model_version_tag(
#                     name=model_name,
#                     version=version,
#                     key="promotion.date",
#                     value=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 )
            
#             # Return validation results
#             return {
#                 "model_name": model_name,
#                 "version": version,
#                 "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix_fig"},
#                 "validation_passed": validation_passed,
#                 "promoted_to_prod": validation_passed and promote_to_prod
#             }
            
#         except Exception as e:
#             logger.error(f"Error during model validation: {e}")
#             import traceback
#             logger.error(traceback.format_exc())
#             try:
#                 mlflow.log_metric("validation_error", 1)
#             except:
#                 pass
#             raise
#         finally:
#             # Always make sure to end the run
#             try:
#                 logger.info(f"Ending MLflow run: {run_id}")
#                 mlflow.end_run()
#             except Exception as e:
#                 logger.warning(f"Error when ending MLflow run: {e}")
#     except Exception as e:
#         logger.error(f"Error starting MLflow run: {e}")
        
#         # Fallback to non-MLflow validation as a last resort
#         logger.info("Attempting validation without MLflow as fallback")
#         return _perform_validation_without_mlflow(
#             registry, model_name, version, stage, texts, labels, 
#             batch_size, min_accuracy, promote_to_prod
#         )


# def _perform_validation_without_mlflow(registry, model_name, version, stage, texts, labels, 
#                                       batch_size, min_accuracy, promote_to_prod):
#     """
#     Perform model validation without MLflow tracking.
#     This is a fallback function when MLflow tracking fails.
#     """
#     logger.info("Running validation without MLflow tracking")
#     try:
#         # Load model from registry
#         manager = registry.load_model(
#             model_name=model_name,
#             version=version,
#             stage=stage if not version else None
#         )
        
#         # Get the actual version that was loaded
#         if not version:
#             try:
#                 version = registry.get_latest_version(model_name)
#                 logger.info(f"Using latest version: {version}")
#             except Exception as e:
#                 logger.warning(f"Could not determine model version: {e}")
#                 version = "unknown"

#         # Preprocess evaluation data
#         dataset = preprocess_data(
#             manager.tokenizer, 
#             texts, 
#             labels, 
#             label_map=COMMON_SENTIMENT_MAPPINGS["text"]
#         )
        
#         # Evaluate model
#         metrics = manager.evaluate(dataset)
        
#         # Log metrics to console instead
#         for k, v in metrics.items():
#             if isinstance(v, (int, float)) and k != "confusion_matrix_fig":
#                 logger.info(f"Metric {k}: {v:.4f}")

#         # Save confusion matrix if available
#         if "confusion_matrix_fig" in metrics and hasattr(metrics["confusion_matrix_fig"], "savefig"):
#             cm_path = f"/tmp/confusion_matrix_{model_name}_v{version}.png"
#             metrics["confusion_matrix_fig"].savefig(cm_path)
#             plt.close(metrics["confusion_matrix_fig"])
#             logger.info(f"Confusion matrix saved to {cm_path}")
        
#         # Determine if validation passes
#         validation_passed = metrics.get("accuracy", 0) >= min_accuracy
        
#         # Add validation result as a tag to the model version
#         validation_tag = {
#             "validation.passed": str(validation_passed),
#             "validation.accuracy": str(round(metrics.get("accuracy", 0), 4)),
#             "validation.date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
        
#         # Add validation tags to the model version
#         for tag_key, tag_value in validation_tag.items():
#             try:
#                 registry.client.set_model_version_tag(
#                     name=model_name,
#                     version=version,
#                     key=tag_key,
#                     value=tag_value
#                 )
#             except Exception as e:
#                 logger.warning(f"Could not set tag {tag_key}: {e}")
        
#         logger.info(f"Validation {'passed' if validation_passed else 'failed'} "
#                    f"(accuracy: {metrics.get('accuracy', 0):.4f}, threshold: {min_accuracy})")
        
#         # Promote to Production if validation passes and promotion is requested
#         if validation_passed and promote_to_prod:
#             logger.info(f"Promoting model {model_name} v{version} to Production")
#             registry.transition_model_stage(model_name, version, "Production")
#             logger.info(f"Model {model_name} v{version} promoted to PRODUCTION")
            
#             # Add production promotion tag
#             try:
#                 registry.client.set_model_version_tag(
#                     name=model_name,
#                     version=version,
#                     key="promotion.date",
#                     value=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 )
#             except Exception as e:
#                 logger.warning(f"Could not set promotion date tag: {e}")
        
#         # Return validation results
#         return {
#             "model_name": model_name,
#             "version": version,
#             "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix_fig"},
#             "validation_passed": validation_passed,
#             "promoted_to_prod": validation_passed and promote_to_prod,
#             "mlflow_tracking": False
#         }
        
#     except Exception as e:
#         logger.error(f"Error during non-MLflow validation: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         raise


# if __name__ == "__main__":
#     args = parse_args()
    
#     # Last resort: kill all MLflow state by setting an emergency environment variable
#     if "--emergency-mode" in sys.argv:
#         logger.warning("EMERGENCY MODE: Running with MLflow tracking bypassed")
#         os.environ["BYPASS_MLFLOW_TRACKING"] = "true"
    
#     try:
#         # Make absolutely sure there's no active run before starting
#         ensure_no_active_run()
        
#         results = validate_model(
#             model_name=args.model_name,
#             version=args.version,
#             stage=args.stage,
#             eval_texts=args.eval_texts,
#             eval_labels=args.eval_labels,
#             batch_size=args.batch_size,
#             experiment_name=args.experiment_name,
#             promote_to_prod=args.promote_to_prod,
#             min_accuracy=args.min_accuracy
#         )
        
#         # Print results to console
#         print("\n=== Validation Results ===")
#         print(f"Model: {results['model_name']} v{results['version']}")
#         print(f"Validation passed: {results['validation_passed']}")
#         if results.get('promoted_to_prod'):
#             print(f"Model promoted to PRODUCTION")
        
#         print("\n=== Metrics ===")
#         for k, v in results['metrics'].items():
#             if isinstance(v, (int, float)):
#                 print(f"{k}: {v:.4f}")
        
#         if results.get('mlflow_tracking') is False:
#             print("\nNote: Validation performed without MLflow tracking")
            
#         # Exit with appropriate status code
#         sys.exit(0 if results['validation_passed'] else 1)
        
#     except Exception as e:
#         print(f"Validation failed with error: {e}")
#         sys.exit(2)

# import sys
# import os
# import logging
# import argparse
# import pandas as pd
# import mlflow
# import matplotlib.pyplot as plt
# from datetime import datetime
# from registry import ModelRegistry

# # Add this method to the beginning of the file, after the imports

# def add_registry_methods():
#     """
#     Add helper methods to ModelRegistry class to improve error handling.
#     """
#     # Check if the ModelRegistry class already has the model_exists method
#     if hasattr(ModelRegistry, 'model_exists') and callable(getattr(ModelRegistry, 'model_exists')):
#         return

#     def model_exists(self, model_name):
#         """Check if a model exists in the registry."""
#         try:
#             self.client.get_registered_model(model_name)
#             return True
#         except Exception:
#             return False
    
#     def list_models(self):
#         """List all registered models."""
#         try:
#             # Get all registered model names
#             registered_models = self.client.list_registered_models()
#             return [model.name for model in registered_models]
#         except Exception as e:
#             logger.error(f"Error listing models: {e}")
#             return []
    
#     # Add the methods to the ModelRegistry class
#     setattr(ModelRegistry, 'model_exists', model_exists)
#     setattr(ModelRegistry, 'list_models', list_models)

# # Call this function to add the methods to the ModelRegistry class
# add_registry_methods()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ==== Load data function ====
# def load_data(texts_file: str, labels_file: str):
#     """Read CSV file and return lists of texts and labels."""
#     try:
#         texts = pd.read_csv(texts_file)['text'].tolist()
#         labels = pd.read_csv(labels_file)['sentiment'].tolist()
#         return texts, labels
#     except Exception as e:
#         logger.error(f"Error loading data: {e}")
#         raise

# # ==== Parse arguments function ====
# def parse_args():
#     parser = argparse.ArgumentParser(description="Validate a registered model")
#     parser.add_argument("--model-name", type=str, required=True,
#                         help="Name of the registered model to validate")
#     parser.add_argument("--version", type=str, default=None,
#                         help="Version of the model to validate (optional if stage is provided)")
#     parser.add_argument("--stage", type=str, default="Staging",
#                         help="Stage of the model to validate (used if version is not provided)")
#     parser.add_argument("--eval-texts", type=str, required=True,
#                         help="Path to the file containing evaluation texts")
#     parser.add_argument("--eval-labels", type=str, required=True,
#                         help="Path to the file containing evaluation labels")
#     parser.add_argument("--batch-size", type=int, default=32,
#                         help="Batch size for evaluation")
#     parser.add_argument("--experiment-name", type=str, default="model-validation",
#                         help="MLflow experiment name")
#     parser.add_argument("--promote-to-prod", type=bool, default=True,
#                         help="Promote model to Production if validation passes")
#     parser.add_argument("--min-accuracy", type=float, default=0.7,
#                         help="Minimum accuracy required to pass validation")
#     return parser.parse_args()


# def ensure_no_active_run():
#     """Ensure there's no active MLflow run and clean environment variables using multiple methods."""
#     # Try conventional approach
#     try:
#         if mlflow.active_run():
#             active_info = mlflow.active_run().info.run_id
#             logger.info(f"Ending existing active run: {active_info}")
#             mlflow.end_run()
#     except Exception as e:
#         logger.warning(f"Error when trying to end active run: {e}")
    
#     # Try more aggressive approach - Access mlflow internals to reset state
#     # This is a fallback method when the regular approach doesn't work
#     try:
#         # Force reset the active run stack in MLflow's internal state
#         # This is using internal API which might change, but helps in stubborn cases
#         if hasattr(mlflow, '_active_run_stack'):
#             logger.info("Forcibly clearing MLflow's internal run stack")
#             if hasattr(mlflow._active_run_stack, 'clear'):
#                 mlflow._active_run_stack.clear()
#             else:
#                 while len(mlflow._active_run_stack) > 0:
#                     mlflow._active_run_stack.pop()
#     except Exception as e:
#         logger.warning(f"Error when trying to reset MLflow's internal state: {e}")
    
#     # Clean MLflow environment variables that might cause conflicts
#     mlflow_env_vars = ['MLFLOW_RUN_ID', 'MLFLOW_EXPERIMENT_ID', 'MLFLOW_EXPERIMENT_NAME', 
#                       'MLFLOW_TRACKING_URI', 'MLFLOW_FLATTEN_PARAMS']
#     for var in mlflow_env_vars:
#         if var in os.environ:
#             logger.info(f"Removing environment variable: {var}")
#             del os.environ[var]
            
#     # Final verification - this should now be None
#     if mlflow.active_run():
#         logger.warning("ATTENTION: Still have an active run after cleanup attempts!")


# def validate_model(model_name, version, stage, eval_texts, eval_labels, batch_size, 
#                   experiment_name, promote_to_prod=False, min_accuracy=0.7):
#     """
#     Validate a model from the registry and optionally promote it to Production
#     if it meets the validation criteria.
#     """
#     # Ensure no active runs before starting a new one
#     ensure_no_active_run()
    
#     # Force any active run to end one more time before proceeding
#     try:
#         if mlflow.active_run():
#             mlflow.end_run()
#     except:
#         pass
    
#     # Create MLflow experiment if it doesn't exist
#     try:
#         exp = mlflow.get_experiment_by_name(experiment_name)
#         if exp is None:
#             exp_id = mlflow.create_experiment(experiment_name)
#         else:
#             exp_id = exp.experiment_id
        
#         logger.info(f"Using experiment: {experiment_name} (ID: {exp_id})")
#     except Exception as e:
#         logger.warning(f"Error setting up MLflow experiment: {e}, will use default experiment")
#         exp_id = "0"  # Use default experiment as fallback

#     # Load evaluation data
#     logger.info(f"Loading evaluation data from {eval_texts} and {eval_labels}")
#     texts, labels = load_data(eval_texts, eval_labels)
#     logger.info(f"Loaded {len(texts)} samples for validation")

#     # Initialize ModelRegistry and check if model exists
#     logger.info("Initializing model registry")
#     registry = ModelRegistry()
    
#     # First check if the model exists in the registry
#     try:
#         # Try to get model info from registry to verify it exists
#         logger.info(f"Checking if model {model_name} exists in registry")
        
#         # Direct check if the model exists before trying to load it
#         model_exists = registry.model_exists(model_name)
        
#         if not model_exists:
#             # If built-in check doesn't exist, try different check
#             try:
#                 # Try to access model directly
#                 registry.client.get_registered_model(model_name)
#                 model_exists = True
#             except Exception as e:
#                 logger.error(f"Model {model_name} not found in registry: {e}")
#                 model_exists = False
            
#         if not model_exists:
#             raise ValueError(f"Model {model_name} not found in registry. Please check the model name or register the model first.")
#     except Exception as e:
#         logger.error(f"Error checking model existence: {e}")
#         if "not found" in str(e).lower():
#             print(f"\n⚠️ MODEL NOT FOUND ERROR ⚠️")
#             print(f"The model '{model_name}' does not exist in the model registry.")
#             print("Available registered models are:")
#             try:
#                 models = registry.list_models()
#                 if models:
#                     for i, model in enumerate(models, 1):
#                         print(f"  {i}. {model}")
#                 else:
#                     print("  No models found in registry")
#             except Exception as list_err:
#                 print(f"  Could not list models: {list_err}")
#             print("\nPlease check the model name or register the model first.")
#             sys.exit(1)
#         raise

#     # Try the validation process without MLflow tracking first if needed
#     # This is a fallback approach in case MLflow is causing issues
#     if os.environ.get('BYPASS_MLFLOW_TRACKING', '').lower() == 'true':
#         logger.info("BYPASS_MLFLOW_TRACKING is enabled, running without MLflow tracking")
#         return _perform_validation_without_mlflow(
#             registry, model_name, version, stage, texts, labels, 
#             batch_size, min_accuracy, promote_to_prod
#         )
        
#     # Normal flow with MLflow tracking
#     try:
#         # Use a completely fresh MLflow instance by resetting internal tracking
#         mlflow.set_tracking_uri(mlflow.get_tracking_uri())
        
#         # Try with nested=True as a fallback option if regular run creation fails
#         try:
#             logger.info("Attempting to start MLflow run")
#             run = mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-validation")
#         except Exception as e:
#             logger.warning(f"Could not create regular run, trying nested run: {e}")
#             run = mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-validation", nested=True)
            
#         run_id = run.info.run_id
#         logger.info(f"Successfully started validation run with ID: {run_id}")
        
#         try:
#             # Log parameters
#             mlflow.log_params({
#                 "model_name": model_name,
#                 "version": version if version else "N/A",
#                 "stage": stage,
#                 "batch_size": batch_size,
#                 "eval_texts": eval_texts,
#                 "eval_labels": eval_labels,
#                 "min_accuracy": min_accuracy
#             })

#             # Load model from registry
#             logger.info(f"Loading model {model_name} (version={version}, stage={stage})")
#             manager = registry.load_model(
#                 model_name=model_name,
#                 version=version,
#                 stage=stage if not version else None
#             )
            
#             # Get the actual version that was loaded
#             if not version:
#                 try:
#                     version = registry.get_latest_version(model_name)
#                     logger.info(f"Using latest version: {version}")
#                 except Exception as e:
#                     logger.warning(f"Could not determine model version: {e}")
#                     version = "unknown"

#             # Preprocess evaluation data
#             logger.info("Preprocessing evaluation data")
#             dataset = preprocess_data(
#                 manager.tokenizer, 
#                 texts, 
#                 labels, 
#                 label_map=COMMON_SENTIMENT_MAPPINGS["text"]
#             )
            
#             # Evaluate model
#             logger.info("Evaluating model")
#             metrics = manager.evaluate(dataset)
            
#             # Log metrics
#             for k, v in metrics.items():
#                 if isinstance(v, (int, float)) and k != "confusion_matrix_fig":
#                     mlflow.log_metric(k, v)
#                     logger.info(f"Metric {k}: {v:.4f}")

#             # Log confusion matrix if available
#             if "confusion_matrix_fig" in metrics and hasattr(metrics["confusion_matrix_fig"], "savefig"):
#                 cm_path = f"/tmp/confusion_matrix_{model_name}_v{version}.png"
#                 metrics["confusion_matrix_fig"].savefig(cm_path)
#                 plt.close(metrics["confusion_matrix_fig"])
#                 mlflow.log_artifact(cm_path)
#                 logger.info(f"Confusion matrix saved to {cm_path} and logged to MLflow")
            
#             # Determine if validation passes
#             validation_passed = metrics.get("accuracy", 0) >= min_accuracy
#             mlflow.log_metric("validation_passed", 1 if validation_passed else 0)
            
#             # Add validation result as a tag to the model version
#             validation_tag = {
#                 "validation.passed": str(validation_passed),
#                 "validation.accuracy": str(round(metrics.get("accuracy", 0), 4)),
#                 "validation.run_id": run_id,
#                 "validation.date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             }
            
#             # Add validation tags to the model version
#             for tag_key, tag_value in validation_tag.items():
#                 registry.client.set_model_version_tag(
#                     name=model_name,
#                     version=version,
#                     key=tag_key,
#                     value=tag_value
#                 )
            
#             logger.info(f"Validation {'passed' if validation_passed else 'failed'} "
#                        f"(accuracy: {metrics.get('accuracy', 0):.4f}, threshold: {min_accuracy})")
            
#             # Promote to Production if validation passes and promotion is requested
#             if validation_passed and promote_to_prod:
#                 logger.info(f"Promoting model {model_name} v{version} to Production")
#                 registry.transition_model_stage(model_name, version, "Production")
#                 logger.info(f"Model {model_name} v{version} promoted to PRODUCTION")
                
#                 # Add production promotion tag
#                 registry.client.set_model_version_tag(
#                     name=model_name,
#                     version=version,
#                     key="promotion.date",
#                     value=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 )
            
#             # Return validation results
#             return {
#                 "model_name": model_name,
#                 "version": version,
#                 "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix_fig"},
#                 "validation_passed": validation_passed,
#                 "promoted_to_prod": validation_passed and promote_to_prod
#             }
            
#         except Exception as e:
#             logger.error(f"Error during model validation: {e}")
#             import traceback
#             logger.error(traceback.format_exc())
#             try:
#                 mlflow.log_metric("validation_error", 1)
#             except:
#                 pass
#             raise
#         finally:
#             # Always make sure to end the run
#             try:
#                 logger.info(f"Ending MLflow run: {run_id}")
#                 mlflow.end_run()
#             except Exception as e:
#                 logger.warning(f"Error when ending MLflow run: {e}")
#     except Exception as e:
#         logger.error(f"Error starting MLflow run: {e}")
        
#         # Fallback to non-MLflow validation as a last resort
#         logger.info("Attempting validation without MLflow as fallback")
#         return _perform_validation_without_mlflow(
#             registry, model_name, version, stage, texts, labels, 
#             batch_size, min_accuracy, promote_to_prod
#         )


# def _perform_validation_without_mlflow(registry, model_name, version, stage, texts, labels, 
#                                       batch_size, min_accuracy, promote_to_prod):
#     """
#     Perform model validation without MLflow tracking.
#     This is a fallback function when MLflow tracking fails.
#     """
#     logger.info("Running validation without MLflow tracking")
#     try:
#         # Load model from registry
#         manager = registry.load_model(
#             model_name=model_name,
#             version=version,
#             stage=stage if not version else None
#         )
        
#         # Get the actual version that was loaded
#         if not version:
#             try:
#                 version = registry.get_latest_version(model_name)
#                 logger.info(f"Using latest version: {version}")
#             except Exception as e:
#                 logger.warning(f"Could not determine model version: {e}")
#                 version = "unknown"

#         # Preprocess evaluation data
#         dataset = preprocess_data(
#             manager.tokenizer, 
#             texts, 
#             labels, 
#             label_map=COMMON_SENTIMENT_MAPPINGS["text"]
#         )
        
#         # Evaluate model
#         metrics = manager.evaluate(dataset)
        
#         # Log metrics to console instead
#         for k, v in metrics.items():
#             if isinstance(v, (int, float)) and k != "confusion_matrix_fig":
#                 logger.info(f"Metric {k}: {v:.4f}")

#         # Save confusion matrix if available
#         if "confusion_matrix_fig" in metrics and hasattr(metrics["confusion_matrix_fig"], "savefig"):
#             cm_path = f"/tmp/confusion_matrix_{model_name}_v{version}.png"
#             metrics["confusion_matrix_fig"].savefig(cm_path)
#             plt.close(metrics["confusion_matrix_fig"])
#             logger.info(f"Confusion matrix saved to {cm_path}")
        
#         # Determine if validation passes
#         validation_passed = metrics.get("accuracy", 0) >= min_accuracy
        
#         # Add validation result as a tag to the model version
#         validation_tag = {
#             "validation.passed": str(validation_passed),
#             "validation.accuracy": str(round(metrics.get("accuracy", 0), 4)),
#             "validation.date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
        
#         # Add validation tags to the model version
#         for tag_key, tag_value in validation_tag.items():
#             try:
#                 registry.client.set_model_version_tag(
#                     name=model_name,
#                     version=version,
#                     key=tag_key,
#                     value=tag_value
#                 )
#             except Exception as e:
#                 logger.warning(f"Could not set tag {tag_key}: {e}")
        
#         logger.info(f"Validation {'passed' if validation_passed else 'failed'} "
#                    f"(accuracy: {metrics.get('accuracy', 0):.4f}, threshold: {min_accuracy})")
        
#         # Promote to Production if validation passes and promotion is requested
#         if validation_passed and promote_to_prod:
#             logger.info(f"Promoting model {model_name} v{version} to Production")
#             registry.transition_model_stage(model_name, version, "Production")
#             logger.info(f"Model {model_name} v{version} promoted to PRODUCTION")
            
#             # Add production promotion tag
#             try:
#                 registry.client.set_model_version_tag(
#                     name=model_name,
#                     version=version,
#                     key="promotion.date",
#                     value=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 )
#             except Exception as e:
#                 logger.warning(f"Could not set promotion date tag: {e}")
        
#         # Return validation results
#         return {
#             "model_name": model_name,
#             "version": version,
#             "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix_fig"},
#             "validation_passed": validation_passed,
#             "promoted_to_prod": validation_passed and promote_to_prod,
#             "mlflow_tracking": False
#         }
        
#     except Exception as e:
#         logger.error(f"Error during non-MLflow validation: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         raise


# if __name__ == "__main__":
#     args = parse_args()
    
#     # Last resort: kill all MLflow state by setting an emergency environment variable
#     if "--emergency-mode" in sys.argv:
#         logger.warning("EMERGENCY MODE: Running with MLflow tracking bypassed")
#         os.environ["BYPASS_MLFLOW_TRACKING"] = "true"
    
#     # Add option to list available models and exit
#     if "--list-models" in sys.argv:
#         try:
#             registry = ModelRegistry()
#             models = registry.list_models()
#             print("\n=== Available Registered Models ===")
#             if models:
#                 for i, model in enumerate(models, 1):
#                     print(f"{i}. {model}")
#             else:
#                 print("No models found in registry")
#             sys.exit(0)
#         except Exception as e:
#             print(f"Error listing models: {e}")
#             sys.exit(1)
    
#     # Add mode to check model exists
#     if "--check-model" in sys.argv:
#         model_name = args.model_name
#         try:
#             registry = ModelRegistry()
#             exists = registry.model_exists(model_name)
#             if exists:
#                 print(f"\nModel '{model_name}' exists in the registry.")
#                 sys.exit(0)
#             else:
#                 print(f"\nModel '{model_name}' does NOT exist in the registry.")
#                 print("\nAvailable models:")
#                 models = registry.list_models()
#                 if models:
#                     for i, model in enumerate(models, 1):
#                         print(f"  {i}. {model}")
#                 else:
#                     print("  No models found in registry")
#                 sys.exit(1)
#         except Exception as e:
#             print(f"Error checking model: {e}")
#             sys.exit(1)
    
#     try:
#         # Make absolutely sure there's no active run before starting
#         ensure_no_active_run()
        
#         results = validate_model(
#             model_name=args.model_name,
#             version=args.version,
#             stage=args.stage,
#             eval_texts=args.eval_texts,
#             eval_labels=args.eval_labels,
#             batch_size=args.batch_size,
#             experiment_name=args.experiment_name,
#             promote_to_prod=args.promote_to_prod,
#             min_accuracy=args.min_accuracy
#         )
        
#         # Print results to console
#         print("\n=== Validation Results ===")
#         print(f"Model: {results['model_name']} v{results['version']}")
#         print(f"Validation passed: {results['validation_passed']}")
#         if results.get('promoted_to_prod'):
#             print(f"Model promoted to PRODUCTION")
        
#         print("\n=== Metrics ===")
#         for k, v in results['metrics'].items():
#             if isinstance(v, (int, float)):
#                 print(f"{k}: {v:.4f}")
        
#         if results.get('mlflow_tracking') is False:
#             print("\nNote: Validation performed without MLflow tracking")
            
#         # Exit with appropriate status code
#         sys.exit(0 if results['validation_passed'] else 1)
        
#     except Exception as e:
#         print(f"Validation failed with error: {e}")
        
#         # Provide helpful suggestions for common errors
#         error_str = str(e).lower()
#         if "registered model" in error_str and "not found" in error_str:
#             print("\nSuggestion: The model may not exist in the registry.")
#             print("Try running with '--list-models' to see available models:")
#             print("    python mlflow/scripts/validate.py --list-models")
#             print("\nOr check if your specific model exists:")
#             print(f"    python validate_model_fixed.py --model-name {args.model_name} --check-model")
        
#         sys.exit(2)

import sys
import os
import logging
import argparse
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from datetime import datetime
from registry import ModelRegistry

# Add path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Internal imports
from src.models.preprocess import preprocess_data, COMMON_SENTIMENT_MAPPINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== Load data function ====
def load_data(texts_file: str, labels_file: str):
    """Read CSV file and return lists of texts and labels."""
    try:
        texts = pd.read_csv(texts_file)['text'].tolist()
        labels = pd.read_csv(labels_file)['sentiment'].tolist()
        return texts, labels
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# ==== Parse arguments function ====
def parse_args():
    parser = argparse.ArgumentParser(description="Validate a registered model")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name of the registered model to validate")
    parser.add_argument("--version", type=str, default=None,
                        help="Version of the model to validate (optional if stage is provided)")
    parser.add_argument("--stage", type=str, default="Staging",
                        help="Stage of the model to validate (used if version is provided)")
    parser.add_argument("--eval-texts", type=str, required=True,
                        help="Path to the file containing evaluation texts")
    parser.add_argument("--eval-labels", type=str, required=True,
                        help="Path to the file containing evaluation labels")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--experiment-name", type=str, default="model-validation",
                        help="MLflow experiment name")
    parser.add_argument("--promote-to-prod", type = bool, default = True,
                        help="Promote model to Production if validation passes")
    parser.add_argument("--min-accuracy", type=float, default=0.7,
                        help="Minimum accuracy required to pass validation")
    return parser.parse_args()

# ==== Robust ensure_no_active_run ====
def ensure_no_active_run():
    """
    Ensure there's no active MLflow run by ending all active (including nested) runs.
    """
    try:
        # Loop until mlflow.active_run() trả về None
        while mlflow.active_run() is not None:
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Ending active MLflow run: {run_id}")
            mlflow.end_run()
    except Exception as e:
        logger.warning(f"Could not cleanly end MLflow runs: {e}")

# ==== Validation function ====
def validate_model(
    model_name, version, stage, eval_texts, eval_labels,
    batch_size, experiment_name, promote_to_prod=False, min_accuracy=0.7
):
    # 1. Đảm bảo không còn run nào đang active
    ensure_no_active_run()

    # 2. Chuẩn bị Experiment
    exp = mlflow.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(experiment_name)
    logger.info(f"Using MLflow experiment '{experiment_name}' (ID: {exp_id})")

    # 3. Load data
    texts, labels = load_data(eval_texts, eval_labels)
    logger.info(f"Loaded {len(texts)} samples for validation")

    # 4. Init registry
    registry = ModelRegistry()

    # 5. Start a new MLflow run (thử nested nếu lỗi)
    try:
        run = mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-validation")
    except Exception:
        run = mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-validation", nested=True)
    run_id = run.info.run_id
    logger.info(f"Started MLflow run: {run_id}")

    try:
        # 6. Log parameters
        mlflow.log_params({
            "model_name": model_name,
            "version": version or "latest",
            "stage": stage,
            "batch_size": batch_size,
            "min_accuracy": min_accuracy
        })

        # 7. Load model từ registry
        manager = registry.load_model(
            model_name=model_name,
            version=version,
            stage=None if version else stage
        )
        if not version:
            version = registry.get_latest_version(model_name)
            logger.info(f"Auto-chosen latest version: {version}")

        # 8. Preprocess & Evaluate
        dataset = preprocess_data(
            manager.tokenizer,
            texts,
            labels,
            label_map=COMMON_SENTIMENT_MAPPINGS['text']
        )
        metrics = manager.evaluate(dataset)

        # 9. Log metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
                logger.info(f"Metric '{k}': {v:.4f}")

        # 11. Quyết định pass/fail
        accuracy = metrics.get('accuracy', 0.0)
        passed = (accuracy >= min_accuracy)
        mlflow.log_metric('validation_passed', int(passed))

        # 12. Gắn tags lên model version
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for tag, val in {
            'validation.passed': str(passed),
            'validation.accuracy': f"{accuracy:.4f}",
            'validation.run_id': run_id,
            'validation.date': timestamp
        }.items():
            registry.client.set_model_version_tag(model_name, version, tag, val)

        # 13. Promote lên Production nếu pass và yêu cầu
        if passed and promote_to_prod:
            registry.transition_model_stage(model_name, version, 'Production')
            registry.client.set_model_version_tag(
                model_name, version, 'promotion.date', timestamp
            )
            logger.info(f"Promoted {model_name} v{version} to Production")

        return {
            'model_name': model_name,
            'version': version,
            'metrics': {k: v for k, v in metrics.items() if k != 'confusion_matrix_fig'},
            'validation_passed': passed,
            'promoted_to_prod': passed and promote_to_prod
        }

    finally:
        # 14. Chắc chắn đã kết thúc run
        mlflow.end_run()

# ==== Entry point ====
if __name__ == '__main__':
    args = parse_args()
    # Đảm bảo clean state trước khi validate
    ensure_no_active_run()
    results = validate_model(
        model_name=args.model_name,
        version=args.version,
        stage=args.stage,
        eval_texts=args.eval_texts,
        eval_labels=args.eval_labels,
        batch_size=args.batch_size,
        experiment_name=args.experiment_name,
        promote_to_prod=args.promote_to_prod,
        min_accuracy=args.min_accuracy
    )
    # In kết quả
    print("\n=== Validation Results ===")
    print(f"Model: {results['model_name']} v{results['version']}")
    print(f"Passed: {results['validation_passed']}")
    if results['promoted_to_prod']:
        print("→ Model đã được promote lên Production")
    print("\nMetrics:")
    for k, v in results['metrics'].items():
        print(f"  {k}: {v:.4f}")

    sys.exit(0 if results['validation_passed'] else 1)
