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
