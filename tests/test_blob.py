import os
import sys
import time
import socket
import logging
import subprocess
import requests
import shutil
import psutil  # You may need to pip install psutil

import mlflow
from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# === 1) Load .env và mirror connection string ===
load_dotenv()
lake = os.environ.get("LAKE_STORAGE_CONN_STR")
if lake and not os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = lake

# Log environment status
if not lake:
    logging.warning("Missing LAKE_STORAGE_CONN_STR in .env file")

# === 2) Logging setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s ─ %(message)s"
)
logger = logging.getLogger(__name__)

# === 3) Constants ===
MLFLOW_RETRY = 3
RETRY_INTERVAL = 2  # giây
MLFLOW_PORT = 5002  # Changed from 5000 to 5001

# === 4) Hàm khởi động MLflow server ===
def start_mlflow_server():
    backend = os.environ.get("MLFLOW_BACKEND_STORE_URI", "sqlite:///mlflow.db")
    container = os.environ.get("AZURE_BLOB_CONTAINER", "artifact")
    account   = os.environ.get("AZURE_ACCOUNT_NAME", "twitlakehouse")
    artifact_root = f"wasbs://{container}@{account}.blob.core.windows.net/mlflow-artifacts"

    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", backend,
        "--default-artifact-root", artifact_root,
        "--host", "0.0.0.0",  # listen mọi interface
        "--port", str(MLFLOW_PORT)  # Use the new port
    ]
    logger.info(f"Starting MLflow server:\n  backend-store-uri={backend}\n  artifact-root={artifact_root}\n  port={MLFLOW_PORT}")
    
    # Create a temporary log file for MLflow's output
    log_file = open("mlflow_server.log", "w")
    
    # Capture stdout and stderr to the log file
    return subprocess.Popen(cmd, stdout=log_file, stderr=log_file)

# === 5) Kiểm tra server có lắng nghe HTTP hay không ===
def wait_for_server(uri=None, timeout=30):  # Increased timeout to 30 seconds
    if uri is None:
        uri = f"http://127.0.0.1:{MLFLOW_PORT}"
        
    parsed = urlparse(uri)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or MLFLOW_PORT

    logger.info(f"Waiting for MLflow server at {host}:{port}...")
    start = time.time()
    
    # Wait for a bit to allow the server to start
    time.sleep(3)
    
    while time.time() - start < timeout:
        # Try direct HTTP connection first
        try:
            logger.info(f"Trying direct HTTP connection to {uri}...")
            r = requests.get(uri, timeout=5)
            logger.info(f"Got response status code: {r.status_code}")
            if r.status_code == 200:
                logger.info("MLflow server UI is responding")
                
                # Now check the API
                try:
                    api_uri = f"{uri}/api/2.0/mlflow/experiments/list"
                    logger.info(f"Checking MLflow API at {api_uri}...")
                    r = requests.get(api_uri, timeout=5)
                    logger.info(f"API response code: {r.status_code}")
                    if r.status_code == 200:
                        logger.info("MLflow server API is up and responsive!")
                        return True
                except Exception as e:
                    logger.warning(f"API check failed, but UI is responsive: {e}")
                    # If the UI works but API fails, still consider it working
                    logger.info("Continuing anyway since UI is working")
                    return True
            else:
                logger.warning(f"Got HTTP response {r.status_code} from server")
        except Exception as e:
            logger.warning(f"HTTP connection failed: {e}")
            
            # Try socket connection as fallback
            sock = socket.socket()
            sock.settimeout(2)
            try:
                logger.info(f"Trying socket connection to {host}:{port}...")
                result = sock.connect_ex((host, port))
                if result == 0:
                    logger.info(f"Socket connection succeeded to {host}:{port}")
                    # If socket works but HTTP failed, wait a bit more and retry
                    time.sleep(2)
                    continue
            except Exception as socket_e:
                logger.warning(f"Socket connection failed: {socket_e}")
            finally:
                sock.close()
        
        # Wait before retry
        time.sleep(2)
        
    logger.error(f"MLflow server không phản hồi HTTP sau {timeout} giây")
    return False

# === 6) Test Azure Blob Storage ===
def test_blob():
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        logger.error("Missing AZURE_STORAGE_CONNECTION_STRING")
        return False
    
    try:
        blob_cli = BlobServiceClient.from_connection_string(conn)
        container_name = os.environ.get("AZURE_BLOB_CONTAINER", "artifact")
        cont_cli = blob_cli.get_container_client(container_name)

        # list & round-trip
        blobs = [b for _,b in zip(range(5), cont_cli.list_blobs())]
        logger.info("Found %d blobs", len(blobs))
        bc = cont_cli.get_blob_client("mlflow_test.txt")
        bc.upload_blob("hello", overwrite=True)
        data = bc.download_blob().readall().decode()
        bc.delete_blob()
        logger.info("Blob round-trip got: %r", data)
        return data == "hello"
    except Exception as e:
        logger.error(f"Blob test failed: {str(e)}")
        return False

# === 7) Test MLflow run với Blob artifact ===
def test_mlflow_run():
    mlflow.set_tracking_uri(f"http://127.0.0.1:{MLFLOW_PORT}")  # Use the new port
    client = mlflow.tracking.MlflowClient()

    try:
        exp_name = "local_sqlite_blob_test"
        exp = client.get_experiment_by_name(exp_name)
        exp_id = exp.experiment_id if exp else client.create_experiment(exp_name)

        with mlflow.start_run(experiment_id=exp_id, run_name="py_test") as run:
            mlflow.log_param("p", 1)
            mlflow.log_metric("m", 2)
            fn = "mf.txt"
            with open(fn,"w") as f: f.write("artifact")
            mlflow.log_artifact(fn)
            os.remove(fn)
            logger.info("Logged MLflow run successfully")
        return True
    except Exception as e:
        logger.error(f"MLflow run test failed: {str(e)}")
        return False

# === 8) Main ===
if __name__ == "__main__":
    # Check for missing dependencies
    try:
        import mlflow
        logger.info(f"MLflow version: {mlflow.__version__}")
        
        try:
            import azure.identity
            logger.info("Azure Identity package is installed")
        except ImportError:
            logger.warning("Azure Identity package is not installed. This may be needed for Azure authentication.")
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        logger.error("Please install required packages with: pip install mlflow[extras] azure-storage-blob azure-identity")
        sys.exit(1)
    
    # Check if MLflow executable is available
    mlflow_path = shutil.which("mlflow")
    if not mlflow_path:
        logger.error("MLflow executable not found in PATH. Please make sure MLflow is installed correctly.")
        sys.exit(1)
    logger.info(f"MLflow executable found at: {mlflow_path}")
    
    # 1) Start server
    logger.info("Starting tests with MLflow port: %s", MLFLOW_PORT)
    log_file_path = os.path.abspath("mlflow_server.log")
    logger.info(f"MLflow server logs will be saved to: {log_file_path}")
    proc = start_mlflow_server()
    
    # Wait a bit to check if process is still running
    time.sleep(2)
    if proc.poll() is not None:
        logger.error(f"MLflow server process terminated immediately with code {proc.returncode}")
        logger.error("Check mlflow_server.log for details")
        with open("mlflow_server.log", "r") as f:
            logger.error("MLflow server log contents:\n" + f.read())
        sys.exit(1)
    
    try:
        # 2) Wait for HTTP
        if not wait_for_server(f"http://127.0.0.1:{MLFLOW_PORT}", timeout=20):
            logger.error("Server did not start properly. Checking logs...")
            # Print the MLflow server logs
            try:
                with open("mlflow_server.log", "r") as f:
                    log_content = f.read()
                    logger.error("MLflow server log contents:\n" + log_content)
            except:
                logger.error("Could not read MLflow server logs")
            proc.terminate()
            sys.exit(1)

        # 3) Blob test
        logger.info("Running Azure Blob Storage test...")
        if not test_blob():
            logger.error("Blob test failed. Exiting.")
            proc.terminate()
            sys.exit(1)

        # 4) MLflow run test
        logger.info("Running MLflow run test...")
        if not test_mlflow_run():
            logger.error("MLflow run test failed. Exiting.")
            proc.terminate()
            sys.exit(1)

        logger.info("✅ Tất cả tests thành công!")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
    finally:
        # Always make sure we terminate the process
        try:
            if proc.poll() is None:  # If process is still running
                logger.info("Terminating MLflow server...")
                proc.terminate()
                proc.wait(timeout=5)
        except:
            pass