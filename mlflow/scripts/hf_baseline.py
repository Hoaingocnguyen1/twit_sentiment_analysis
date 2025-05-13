import sys
import os
import uuid
import getpass
import datetime
import logging
import argparse
import pandas as pd
import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from registry import ModelRegistry

# Add path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Internal imports
from src.models.preprocess import preprocess_data, COMMON_SENTIMENT_MAPPINGS
from src.models.transformer_manager import ModelManager
from config.dataClient import get_blob_storage

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
    parser = argparse.ArgumentParser(description="Run baseline sentiment analysis model")
    parser.add_argument("--model-name", type=str, default="roberta-baseline",
                        help="Alias for this model run")
    parser.add_argument("--hf-model-name", type=str,
                        default="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                        help="Name of the Hugging Face model to use")
    parser.add_argument("--version", type=str, default="v1.0.0",
                        help="Version tag for this model (used in path and blob storage)")
    parser.add_argument("--texts-file", type=str, default="artifact/data/X_test.csv",
                        help="Path to the file containing texts")
    parser.add_argument("--labels-file", type=str, default="artifact/data/y_test.csv",
                        help="Path to the file containing labels")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--experiment-name", type=str, default="sentiment-analysis-baseline",
                        help="MLflow experiment name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_name = args.model_name
    hf_name = args.hf_model_name
    version = args.version
    texts_file = args.texts_file
    labels_file = args.labels_file
    batch_size = args.batch_size
    experiment_name = args.experiment_name
    output_dir = f"artifact/models/{model_name}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

     # Đảm bảo không có active run nào đang tồn tại
    if mlflow.active_run():
        mlflow.end_run()
        logger.info("Ended existing active run")

        # Xóa bỏ các biến môi trường MLflow có thể gây xung đột
    if 'MLFLOW_RUN_ID' in os.environ:
        del os.environ['MLFLOW_RUN_ID']
    if 'MLFLOW_EXPERIMENT_ID' in os.environ:
        del os.environ['MLFLOW_EXPERIMENT_ID']
    if 'MLFLOW_EXPERIMENT_NAME' in os.environ:
        del os.environ['MLFLOW_EXPERIMENT_NAME']

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id
    # Thiết lập experiment một cách chính xác
    logger.info(f"Using experiment: {experiment_name} (ID: {exp_id})")

    # Mở Run duy nhất
    with mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-evaluation") as run:
        run_id = run.info.run_id
        logger.info(f"Started new run with ID: {run_id}")
        
        # Log parameters
        mlflow.log_params({
            "model_name": model_name,
            "base_model": hf_name,
            "batch_size": batch_size,
            "texts_file": texts_file,
            "labels_file": labels_file
        })

        # Load data
        texts, labels = load_data(texts_file, labels_file)
        logger.info(f"Loaded {len(texts)} samples for evaluation")

        # Initialize ModelManager without fine-tuning
        manager = ModelManager(model_path=hf_name, finetune=False, output_dir=output_dir)
        manager.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        manager.model = AutoModelForSequenceClassification.from_pretrained(hf_name)
        manager.model.to(manager.device)

        # Preprocess and evaluate
        dataset = preprocess_data(manager.tokenizer, texts, labels, label_map=COMMON_SENTIMENT_MAPPINGS["text"])
        metrics = manager.evaluate(dataset)

        # Log metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and k != "confusion_matrix_fig":
                mlflow.log_metric(k, v)

        # Xử lý logging ma trận nhầm lẫn
        if "confusion_matrix_fig" in metrics:
            try:
                # Tạo thư mục nếu chưa tồn tại
                cm_dir = os.path.join(output_dir, model_name, version)
                os.makedirs(cm_dir, exist_ok=True)
                
                # Lưu ma trận nhầm lẫn
                cm_path = os.path.join(cm_dir, "confusion_matrix.png")
                
                # Đảm bảo figure được lưu đúng cách
                if hasattr(metrics["confusion_matrix_fig"], "savefig"):
                    metrics["confusion_matrix_fig"].savefig(cm_path)
                    plt.close(metrics["confusion_matrix_fig"])  # Giải phóng bộ nhớ
                    
                    # Log artifact
                    if os.path.exists(cm_path):
                        mlflow.log_artifact(cm_path)
                        logger.info(f"Ma trận nhầm lẫn đã lưu tại {cm_path} và được log vào MLflow")
                    else:
                        logger.warning(f"Không thể tìm thấy tệp ma trận nhầm lẫn: {cm_path}")
            except Exception as e:
                logger.warning(f"Lỗi khi lưu ma trận nhầm lẫn: {e}")

        # Print results to console
        print("=== Baseline metrics ===")
        for k, v in metrics.items():
            if k != "confusion_matrix_fig" and isinstance(v, (int, float)):
                print(f"{k}: {v:.4f}")

        try:
            # Lấy client blob storage
            blob_storage = get_blob_storage()
            
            # Lưu mô hình với blob storage
            try:
                model_save_path = manager.save_model(
                    blob_storage=blob_storage,  # Optional
                    version=version,            # Optional
                    metrics={k: v for k, v in metrics.items() if k != "confusion_matrix_fig"},  # Optional
                    cm_fig=metrics.get("confusion_matrix_fig")  # Optional
                )
                logger.info(f"Mô hình đã lưu tại {model_save_path}")
            except Exception as blob_error:
                # Ghi log lỗi chi tiết
                logger.warning(f"Lỗi khi lưu vào blob storage: {blob_error}")
                import traceback
                logger.debug(f"Chi tiết lỗi blob storage: {traceback.format_exc()}")
                
                # Lưu cục bộ nếu lưu vào blob storage thất bại
                model_save_path = manager.save_model(
                    version=version,
                    metrics={k: v for k, v in metrics.items() if k != "confusion_matrix_fig"},
                    cm_fig=metrics.get("confusion_matrix_fig")
                )
                logger.info(f"Mô hình đã lưu cục bộ tại {model_save_path}")
        except Exception as e:
            logger.error(f"Lỗi trong quá trình lưu mô hình: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # ==== Register model in Staging ====
        try:
            # Đảm bảo run_id vẫn còn hợp lệ
            if not run_id:
                logger.warning("Run ID không hợp lệ hoặc không tìm thấy, sử dụng run hiện tại")
                current_run = mlflow.active_run()
                if current_run:
                    run_id = current_run.info.run_id
                else:
                    with mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-registration") as reg_run:
                        run_id = reg_run.info.run_id
                logger.info(f"Sử dụng run ID: {run_id} cho đăng ký mô hình")
            
        #     registry = ModelRegistry()
        #     model_version = registry.register_model(
        #         run_id=run_id,
        #         model_name=model_name,
        #         description=f"Baseline sentiment analysis model based on {hf_name}",
        #         metrics={k: v for k, v in metrics.items()
        #                 if isinstance(v, (int, float)) and k != "confusion_matrix_fig"},
        #         model_manager=manager
        #     )
        #     logger.info(f"Model registered with version: {model_version}")
        #     print(f"Model registered with version: {model_version}")

        #     # Transition to Staging (awaiting review before Production)
        #     registry.transition_model_stage(model_name, model_version, "Staging")
        #     logger.info(f"Model {model_name} v{model_version} transitioned to STAGING")
            
        # except Exception as e:
        #     logger.error(f"Error during model registration: {e}")
        #     print(f"Model registration failed: {e}")
        #     # Ghi lại stack trace đầy đủ để debug
        #     import traceback
        #     logger.error(traceback.format_exc())

 # Create model tags
            model_tags = {
                "model_type": "sentiment_analysis",
                "base_model": hf_name,
                "version": version,
                "framework": "transformers",
                "data_source": f"{texts_file}|{labels_file}",
                "batch_size": str(batch_size),
                "registered_by": getpass.getuser(),  # User who registered the model
                "registered_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add metrics to tags for easy searching
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k != "confusion_matrix_fig":
                    model_tags[f"metric.{k}"] = str(round(v, 4))
            
            # Initialize registry
            registry = ModelRegistry()
            
            # Register model with tags
            model_version = registry.register_model(
                run_id=run_id,
                model_name=model_name,
                description=f"Baseline sentiment analysis model based on {hf_name}",
                metrics={k: v for k, v in metrics.items()
                        if isinstance(v, (int, float)) and k != "confusion_matrix_fig"},
                model_manager=manager,
                tags=model_tags  # Pass tags to register_model
            )
            logger.info(f"Model registered with version: {model_version}")
            print(f"Model registered with version: {model_version}")

            # Transition to Staging (awaiting review before Production)
            registry.transition_model_stage(model_name, model_version, "Staging")
            logger.info(f"Model {model_name} v{model_version} transitioned to STAGING")
            
            # Define useful aliases
            aliases = [
                f"v{version}",                         # Human-readable version
                f"{version}-{hf_name.split('/')[-1]}", # Combination of version and base model
                datetime.datetime.now().strftime("%Y%m%d")  # Registration date
            ]
            
            # Add aliases to the model
            registry.add_aliases(model_name, model_version, aliases)
            
        except Exception as e:
            logger.error(f"Error during model registration: {e}")
            print(f"Model registration failed: {e}")
            # Log full stack trace for debugging
            import traceback
            logger.error(traceback.format_exc())