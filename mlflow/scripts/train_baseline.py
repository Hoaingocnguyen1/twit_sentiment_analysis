import sys
import os
import uuid
import getpass
import datetime
import argparse
import mlflow
from mlflow.tracking import MlflowClient
import logging
import io
import pandas as pd
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import urllib.parse
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(root_dir)
from src.models.preprocess import preprocess_data
from config.dataClient import get_blob_storage
from src.models.transformer_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s ─ %(message)s"
)
logger = logging.getLogger(__name__)
# Train tay -> registry

load_dotenv()

lake = os.getenv("LAKE_STORAGE_CONN_STR")
if lake and not os.getenv("AZURE_STORAGE_CONNECTION_STRING"):
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = lake
container_name = os.getenv("AZURE_BLOB_CONTAINER", "testartifact")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://172.23.51.243:5000")
artifact_uri = f'wasbs://{container_name}@twitlakehouse.blob.core.windows.net/mlflow-artifacts'

mlflow.set_tracking_uri(tracking_uri)
logger.info(f"Tracking URI: {tracking_uri}")
logger.info(f"Artifact URI: {artifact_uri}")

client = MlflowClient(tracking_uri=tracking_uri)

def encode_model_name(name: str) -> str:
    return urllib.parse.quote(name, safe='')

def load_and_preprocess_from_blob(
    tokenizer, max_length: int = 128, label_map: Optional[dict] = None,
    container_name: str = "testartifact", prefix: str = "data/"
) -> Tuple[Optional[object], Optional[object]]:
    """
    Tải dữ liệu train/test từ blob, chuyển về DataFrame, tiền xử lý thành dataset cho model.
    """
    try:
        blob = get_blob_storage()

        train_json_str = blob.read_json_from_container(container_name, prefix + "train_dataset.json")
        test_json_str = blob.read_json_from_container(container_name, prefix + "test_dataset.json")

        logger.info("Tải dữ liệu JSON từ blob storage thành công.")

        # Kiểm tra dữ liệu trả về có phải list không, nếu không thì đọc JSON từ string
        if isinstance(train_json_str, list):
            train_df = pd.DataFrame(train_json_str)
        else:
            train_df = pd.read_json(io.StringIO(train_json_str), lines=True)

        if isinstance(test_json_str, list):
            test_df = pd.DataFrame(test_json_str)
        else:
            test_df = pd.read_json(io.StringIO(test_json_str), lines=True)

        logger.info(f"Đã parse dataframe - Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        X_train = train_df['text'].tolist()
        y_train = train_df['sentiment'].tolist()
        X_test = test_df['text'].tolist()
        y_test = test_df['sentiment'].tolist()

        logger.info(f"Trích xuất thành công lists - train texts: {len(X_train)}, train labels: {len(y_train)}")
        if X_train:
            logger.info(f"Sample text train: '{X_train[0][:50]}...'")
            logger.info(f"Sample label train: {y_train[0]}")

        # Tiền xử lý dữ liệu
        train_dataset = preprocess_data(tokenizer=tokenizer, texts=X_train, labels=y_train,
                                        max_length=max_length, label_map=label_map)
        test_dataset = preprocess_data(tokenizer=tokenizer, texts=X_test, labels=y_test,
                                       max_length=max_length, label_map=label_map)

        logger.info(f"Tiền xử lý thành công, train dataset size: {len(train_dataset)}, test dataset size: {len(test_dataset)}")
        return train_dataset, test_dataset

    except Exception as e:
        logger.error(f"Lỗi trong load_and_preprocess_from_blob: {e}")
        return None, None

def train_model(manager, run_id, train_ds, eval_ds, params):
    mlflow.log_params(params)
    results = manager.train(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        learning_rate=params['learning_rate']
    )
    mlflow.log_metric('final_train_loss', results['final_train_loss'])
    for k,v in results.get('eval_metrics', {}).items():
        if isinstance(v, (int,float)):
            mlflow.log_metric(f"eval_{k}", v)
    if 'confusion_matrix' in results:
        mlflow.log_figure(results['confusion_matrix'], artifact_file='confusion_matrix.png')
    mlflow.transformers.log_model(
        transformers_model={
                    "model": manager.model,
                    "tokenizer": manager.tokenizer
                },
                artifact_path="model",
                run_id=run_id,
                task="text-classification")
    return results

def register_model(model_name, run_id, training_results):
    try:
        encoded_name = encode_model_name(model_name)
        latest_versions = client.get_latest_versions(encoded_name)
        max_version = max([int(v.version) for v in latest_versions]) if latest_versions else 0
        next_version = str(max_version + 1)
    except Exception:
        next_version = "1"

    tags = {
        'model_type': 'sentiment_analysis',
        'base_model': encoded_name,
        'version': next_version,
        'framework': 'transformers',
        'registered_by': getpass.getuser(),
        'registered_at': datetime.datetime.now().isoformat()
    }
    metrics = {'final_train_loss': training_results['final_train_loss']}
    metrics.update({k: v for k, v in training_results.get('eval_metrics', {}).items() if isinstance(v, (int, float))})

    uri = f"runs:/{run_id}/model"
    try:
        client.get_registered_model(encoded_name)
    except Exception:
        client.create_registered_model(encoded_name)

    mv = client.create_model_version(
        name=encoded_name,
        source=uri,
        run_id=run_id
    )

    for k, v in tags.items():
        client.set_model_version_tag(encoded_name, mv.version, k, v)

    client.transition_model_version_stage(
        name=encoded_name,
        version=mv.version,
        stage='Staging',
        archive_existing_versions=False
    )

    aliases = [f"v{next_version}", next_version, datetime.datetime.now().strftime("%Y%m%d")]
    for alias in aliases:
        try:
            client.set_registered_model_alias(name=encoded_name, alias=alias, version=mv.version)
        except Exception as e:
            logger.warning(f"Could not set alias {alias}: {e}")

    logger.info(f"Registered {encoded_name} v{mv.version}")
    return str(mv.version)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-name', required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--finetune', type=bool, default=True)
    p.add_argument('--experiment-name', type=str, default='default_exp')
    args = p.parse_args()

    print("\nMLflow Configuration:")
    print(f"- Tracking URI: {tracking_uri}")
    print(f"- Artifact URI: {artifact_uri}")
    print(f"- Azure Storage: {'Configured' if lake else 'Not configured'}")
    print(f"- Container: {container_name}")
    print("\n" + "-"*50 + "\n")

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

    mlflow.set_experiment(args.experiment_name)

    mgr = ModelManager(model_path=args.model_name, finetune=args.finetune)

    train_ds, eval_ds = load_and_preprocess_from_blob(tokenizer=mgr.tokenizer)

    if train_ds is None or eval_ds is None:
        logger.error("Failed to load and preprocess datasets. Exiting.")
        exit(1)

    # Giả sử train_ds và eval_ds có thể lấy length trực tiếp
    train_samples = len(train_ds)
    eval_samples = len(eval_ds) if eval_ds else 0

    if mlflow.active_run():
        mlflow.end_run()

    run_name = f"{args.model_name}-train-{uuid.uuid4().hex[:8]}"
    print(f"Starting new run: {run_name}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        }
        mlflow.log_params({
            'model_name': args.model_name,
            'train_samples': train_samples,
            'eval_samples': eval_samples
        })
        mlflow.set_tag("created_by", getpass.getuser())
        mlflow.set_tag("created_at", datetime.datetime.now().isoformat())
        
        run_id = run.info.run_id
        res = train_model(mgr, run_id, train_ds, eval_ds, params)
        print(f"Run completed with ID: {run_id}")

        tracking_uri = mlflow.get_tracking_uri()
        run_url = f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
        print(f"View this run at: {run_url}")

    version = register_model(args.model_name, run_id, res)
    print(f"Registered model: {args.model_name}, version: {version}")
