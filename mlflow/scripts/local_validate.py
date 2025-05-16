import sys
import os
import logging
import argparse
import getpass
from datetime import datetime
from pathlib import Path
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from registry import ModelRegistry
# Internal imports
from src.models.preprocess import preprocess_data, COMMON_SENTIMENT_MAPPINGS

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
def load_data(texts_file: str, labels_file: str):
    """Read CSV files and return lists of texts and labels."""
    try:
        texts = pd.read_csv(texts_file)["text"].tolist()
        labels = pd.read_csv(labels_file)["sentiment"].tolist()
        return texts, labels
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a registered model")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name of the registered model to validate")
    parser.add_argument("--version", type=str, default=None,
                        help="Specific version to validate (optional if stage is provided)")
    parser.add_argument("--stage", type=str, default="Staging",
                        help="Stage to pull model from if version not provided")
    parser.add_argument("--eval-texts", type=str, required=True,
                        help="Path to evaluation texts CSV")
    parser.add_argument("--eval-labels", type=str, required=True,
                        help="Path to evaluation labels CSV")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--experiment-name", type=str, default="model-validation",
                        help="MLflow experiment name to use")
    parser.add_argument("--promote-to-prod", type=bool, default=True,
                        help="Promote to Production if validation passes")
    parser.add_argument("--min-accuracy", type=float, default=0.7,
                        help="Minimum accuracy to pass validation")
    return parser.parse_args()


def ensure_no_active_run():
    """
    End all active MLflow runs (including nested) to avoid conflicts.
    """
    try:
        while mlflow.active_run() is not None:
            current = mlflow.active_run()
            logger.info(f"Ending active run: {current.info.run_id}")
            mlflow.end_run()
    except Exception as e:
        logger.warning(f"Error ending MLflow runs: {e}")


def validate_model(
    model_name,
    version,
    stage,
    eval_texts,
    eval_labels,
    batch_size,
    experiment_name,
    promote_to_prod=False,
    min_accuracy=0.7
):
    # Clean up any existing runs
    ensure_no_active_run()

    # Prepare experiment
    exp = mlflow.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(experiment_name)
    logger.info(f"Using MLflow experiment '{experiment_name}' (ID: {exp_id})")

    # Load data
    texts, labels = load_data(eval_texts, eval_labels)
    logger.info(f"Loaded {len(texts)} samples for validation")

    # Initialize registry
    registry = ModelRegistry()

    # Start MLflow run
    try:
        run = mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-validation")
    except Exception:
        run = mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-validation", nested=True)
    run_id = run.info.run_id
    logger.info(f"Started MLflow run: {run_id}")

    try:
        # Log parameters
        mlflow.log_params({
            "model_name": model_name,
            "version": version or "latest",
            "stage": stage,
            "batch_size": batch_size,
            "min_accuracy": min_accuracy
        })

        model_path=f"{BASE_DIR}/artifact/models/{model_name}"
        model_dir = Path(model_path).resolve()

    # 2) turn it into a file URI (e.g. file:///E:/…)
        # Load model from registry
        manager = registry.load_model(
            model_path=model_dir,
            model_name=model_name,
            version=version,
            stage=None if version else stage
        )

        manager.output_dir = model_dir

        # Preprocess and evaluate
        dataset = preprocess_data(
            manager.tokenizer,
            texts,
            labels,
            label_map=COMMON_SENTIMENT_MAPPINGS['text']
        )
        temp = manager.evaluate(dataset, batch_size=batch_size)

        keys = ['accuracy', 'precision', 'recall', 'f1', 'loss', 'confusion_matrix_fig']

        metric_keys = keys[:-2]
        eval_keys = keys[-2:]

        metrics = {k: temp[k] for k in metric_keys}
        eval_results = {k: temp[k] for k in eval_keys}

        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
                logger.info(f"Metric {key}: {value:.4f}")

        # Register model if needed
        register_tags = {
            "model_type": "sentiment_analysis",
            "base_model": model_name,
            "framework": "transformers",
            "batch_size": str(batch_size),
            "registered_by": getpass.getuser(),
            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        # Add eval metrics as tags
        for k, v in eval_results.items():
            if isinstance(v, (int, float)):
                register_tags[f"eval.{k}"] = str(round(v, 4))

        registered_version = registry.register_model(
            run_id=run_id,
            model_name=model_name,
            description=f"Validation run of {model_name}",
            metrics=metrics,
            model_manager=manager,
            tags=register_tags
        )
        logger.info(f"Model registered under version {registered_version}")

        # Transition to staging
        registry.transition_model_stage(model_name, registered_version, "Staging")
        logger.info(f"Transitioned {model_name} v{registered_version} to Staging")

        # Decide pass/fail
        accuracy = metrics.get('accuracy', 0.0)
        passed = accuracy >= min_accuracy
        mlflow.log_metric('validation_passed', int(passed))

        # Tag model version
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        registry.client.set_model_version_tag(
            model_name, registered_version, 'validation.passed', str(passed)
        )
        registry.client.set_model_version_tag(
            model_name, registered_version, 'validation.accuracy', f"{accuracy:.4f}")
        registry.client.set_model_version_tag(
            model_name, registered_version, 'validation.date', timestamp
        )

        # Promote if passed
        if passed and promote_to_prod:
            registry.transition_model_stage(model_name, registered_version, 'Production')
            registry.client.set_model_version_tag(
                model_name, registered_version, 'promotion.date', timestamp
            )
            logger.info(f"Promoted {model_name} v{registered_version} to Production")

        return {
            'model_name': model_name,
            'version': registered_version,
            'metrics': metrics,
            'validation_passed': passed,
            'promoted_to_prod': passed and promote_to_prod
        }

    finally:
        mlflow.end_run()


if __name__ == '__main__':
    args = parse_args()
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
    # Print results
    print("=== Validation Results ===")
    print(f"Model: {results['model_name']} {results['version']}")
    print(f"Passed: {results['validation_passed']}")
    if results['promoted_to_prod']:
        print("→ Promoted to Production")
    print("Metrics:")
    for k, v in results['metrics'].items():
        print(f"  {k}: {v:.4f}")
    sys.exit(0 if results['validation_passed'] else 1)