"""
Script kiểm tra kết nối MLflow từ Windows đến WSL
"""
import os
import mlflow
from dotenv import load_dotenv
import uuid
import datetime

# Load biến môi trường từ file .env
load_dotenv()

# Kiểm tra MLflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    # Lấy thông tin kết nối
    tracking_uri = os.getenv("TRACKING_URI", "http://172.23.51.243:5000")
    mlflow.set_tracking_uri(tracking_uri)
    print("Tracking URI:", mlflow.get_tracking_uri())
    artifact_uri ='wasbs://testartifact@twitlakehouse.blob.core.windows.net/mlflow-artifacts'
    lake = os.environ.get("LAKE_STORAGE_CONN_STR")
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = lake
    azure_storage = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container = os.getenv("AZURE_BLOB_CONTAINER", "testartifact")

    # Hiển thị thông tin cấu hình
    print(f"Cấu hình MLflow:")
    print(f"- Tracking URI: {tracking_uri}")
    print(f"- Artifact URI: {artifact_uri}")
    print(f"- Azure Storage: {'Đã cấu hình' if azure_storage else 'Chưa cấu hình'}")
    print(f"- Container: {container}")
    print("\n" + "-"*50 + "\n")

    # Khởi tạo MLflow client

    client = MlflowClient(tracking_uri=tracking_uri)
    
    # Tìm hoặc tạo experiment
    experiment_name = "windows_test_" + datetime.datetime.now().strftime("%Y%m%d")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Creating new experiment: {experiment_name}")
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_uri
        )
        print(f"Created experiment with ID: {experiment_id}")
    else:
        print(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
        experiment_id = experiment.experiment_id
    
    # Bắt đầu một run mới
    run_name = f"test_run_{uuid.uuid4().hex[:8]}"
    print(f"Starting new run: {run_name}")
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("test_param_1", "value1")
        mlflow.log_param("test_param_2", 42)
        
        # Log metrics
        mlflow.log_metric("accuracy", 0.87)
        mlflow.log_metric("loss", 0.23)
        
        # Log tags
        mlflow.set_tag("test_tag", "Windows test")
        mlflow.set_tag("created_by", os.getenv("USERNAME", "unknown"))
        
        # Log một artifact nhỏ
        with open("test_artifact.txt", "w") as f:
            f.write("This is a test artifact created on Windows.\n")
            f.write(f"Created at: {datetime.datetime.now().isoformat()}\n")
        
        mlflow.log_artifact("test_artifact.txt")
        
        # Lấy thông tin về run
        run_id = run.info.run_id
        print(f"Run completed with ID: {run_id}")
        print(f"View this run at: {tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}")

except ImportError as e:
    print(f"Error: {str(e)}")
    print("Make sure MLflow is installed with: pip install mlflow")
except Exception as e:
    print(f"Error: {str(e)}")
    print("Make sure MLflow server is running and accessible.")