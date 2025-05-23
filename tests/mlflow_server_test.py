#!/usr/bin/env python3
import os
import sys
import logging
import requests
import time
import json
import mlflow
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# === 1) Setup logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s ─ %(message)s"
)
logger = logging.getLogger(__name__)

# === 2) Load environment variables ===
load_dotenv()
lake = os.environ.get("LAKE_STORAGE_CONN_STR")
if lake and not os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = lake

# === 3) Configuration ===
MLFLOW_PORT = int(os.environ.get("MLFLOW_PORT", 5000))  # Default port from your bash script
MLFLOW_TRACKING_URI = "http://172.23.51.243:5000"

# === 4) Test server connection ===
def test_server_connection(verbose=True):
    """Test if the MLflow server is running and accessible"""
    logger.info(f"Testing connection to MLflow server at {MLFLOW_TRACKING_URI}...")

    try:
        # Test UI endpoint
        logger.info("Testing MLflow UI access...")
        resp = requests.get(MLFLOW_TRACKING_URI, timeout=5)
        if resp.status_code == 200:
            logger.info("MLflow UI is accessible")
        else:
            logger.error(f"MLflow UI returned status code {resp.status_code}")
            logger.error(f"Response content: {resp.text[:200]}...")
            return False

        # Test API endpoint with more detailed debugging
        logger.info("Testing MLflow API access...")
        api_url = f"{TRACKING_URI}/api/2.0/mlflow/experiments/list"
        headers = {"Content-Type": "application/json"}
        payload = {}

        logger.info(f"Making API request to: {api_url}")
        logger.info(f"Headers: {headers}")
        logger.info(f"Payload: {payload}")

        try:
            api_resp = requests.post(api_url, json=payload, headers=headers, timeout=5)

            # Log detailed API response information
            logger.info(f"API response status code: {api_resp.status_code}")
            logger.info(f"API response headers: {dict(api_resp.headers)}")

            if verbose:
                try:
                    # Try to parse and pretty-print the JSON response
                    resp_json = api_resp.json()
                    logger.info(f"API response content: {json.dumps(resp_json, indent=2)}")
                except Exception as e:
                    # If not JSON, log the raw text (truncated)
                    logger.info(f"API response content (not JSON): {api_resp.text[:500]}")

            if api_resp.status_code == 200:
                logger.info("MLflow API is accessible")
                return True
            else:
                logger.error(f"MLflow API returned status code {api_resp.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed with exception: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"Failed to connect to MLflow server: {str(e)}")
        return False

# === 5) Test Blob Storage ===
def test_blob_storage():
    """Test Azure Blob Storage connection and operations"""
    logger.info("Testing Azure Blob Storage connection...")
    
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        logger.error("Missing AZURE_STORAGE_CONNECTION_STRING")
        return False
    
    container_name = os.environ.get("AZURE_BLOB_CONTAINER", "artifact")
    
    try:
        # Create a test blob and read it back
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        container_client = blob_service_client.get_container_client(container_name)
        
        # Check if container exists
        if not container_client.exists():
            logger.error(f"Container '{container_name}' does not exist")
            return False
        
        logger.info(f"Successfully connected to container '{container_name}'")
        
        # Create test blob
        test_blob_name = "mlflow_connection_test.txt"
        test_content = f"MLflow connection test at {time.time()}"
        
        blob_client = container_client.get_blob_client(test_blob_name)
        logger.info(f"Uploading test blob '{test_blob_name}'...")
        blob_client.upload_blob(test_content, overwrite=True)
        
        # Read back the content
        downloaded = blob_client.download_blob().readall().decode('utf-8')
        logger.info(f"Downloaded content: '{downloaded}'")
        
        # Verify content matches
        if downloaded == test_content:
            logger.info("Blob content verification successful")
        else:
            logger.error("Blob content verification failed")
            return False
        
        # Clean up
        logger.info(f"Deleting test blob '{test_blob_name}'...")
        blob_client.delete_blob()
        
        return True
    except Exception as e:
        logger.error(f"Azure Blob Storage test failed: {str(e)}")
        return False

# === 6) Test MLflow Run with Artifact ===
def test_mlflow_with_artifact():
    """Test MLflow run with artifact storage"""
    logger.info(
        f"Testing MLflow run with artifacts using tracking URI: {MLFLOW_TRACKING_URI}"
    )

    # Set the tracking URI to connect to your running server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        # Create or get experiment
        experiment_name = "connection_test_experiment"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

        # Start a run and log artifacts
        with mlflow.start_run(experiment_id=experiment_id, run_name="connection_test") as run:
            run_id = run.info.run_id
            logger.info(f"Started run with ID: {run_id}")

            # Log parameters and metrics
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 42)

            # Create and log a test artifact
            artifact_path = "test_artifact.txt"
            with open(artifact_path, "w") as f:
                f.write("This is a test artifact for MLflow artifact store verification")

            mlflow.log_artifact(artifact_path)
            logger.info(f"Logged artifact: {artifact_path}")

            # Clean up local file
            os.remove(artifact_path)

        logger.info(f"Successfully completed MLflow run with artifacts")
        logger.info(
            f"Run URL: {MLFLOW_TRACKING_URI}/#/experiments/{experiment_id}/runs/{run_id}"
        )
        return True
    except Exception as e:
        logger.error(f"MLflow run test failed: {str(e)}")
        return False

# === 7) Test specific API endpoints ===
def test_api_endpoints():
    """Test specific MLflow API endpoints to narrow down issues"""
    logger.info("Testing specific MLflow API endpoints...")

    endpoints = [
        # Core API endpoints
        {"name": "List Experiments", "method": "POST", "path": "/api/2.0/mlflow/experiments/list", "payload": {}},
        {"name": "Get Experiment", "method": "GET", "path": "/api/2.0/mlflow/experiments/get-by-name", "params": {"experiment_name": "Default"}},
        {"name": "MLflow Ping", "method": "GET", "path": "/api/2.0/mlflow/experiments/list-artifacts", "params": {"run_id": "test", "path": ""}}
    ]

    success_count = 0

    for endpoint in endpoints:
        name = endpoint["name"]
        method = endpoint["method"]
        path = endpoint["path"]
        url = f"{MLFLOW_TRACKING_URI}{path}"

        logger.info(f"Testing endpoint: {name} ({method} {path})")

        try:
            if method == "GET":
                params = endpoint.get("params", {})
                resp = requests.get(url, params=params, timeout=5)
            elif method == "POST":
                payload = endpoint.get("payload", {})
                resp = requests.post(url, json=payload, timeout=5)

            if resp.status_code == 200:
                logger.info(f"Endpoint {name} is accessible (Status: {resp.status_code})")
                success_count += 1
            else:
                logger.warning(f"❌ Endpoint {name} returned status code {resp.status_code}")
                # Don't print entire response which might be error HTML
                logger.warning(f"Response preview: {resp.text[:100]}...")
        except Exception as e:
            logger.error(f"Error testing endpoint {name}: {str(e)}")

    logger.info(f"API endpoint test summary: {success_count}/{len(endpoints)} endpoints accessible")
    return success_count > 0

# === 8) Main function ===
def main():
    logger.info("=== MLflow Server and Artifact Store Test ===")

    # 1. Test basic server connection
    if not test_server_connection():
        logger.warning("Basic MLflow server connection test failed. Testing specific API endpoints...")
        # If the main test fails, try testing specific endpoints
        if not test_api_endpoints():
            logger.error("All API endpoint tests failed. Check if the MLflow server is running correctly.")
            return False

    # 2. Test Azure Blob Storage
    if not test_blob_storage():
        logger.error("Failed to test Azure Blob Storage. Exiting.")
        return False

    # 3. Test MLflow with artifacts
    if not test_mlflow_with_artifact():
        logger.error("Failed to test MLflow with artifacts. Exiting.")
        return False

    logger.info("=== All tests passed successfully! ===")
    logger.info(
        f"Your MLflow server at {MLFLOW_TRACKING_URI} is properly configured with Azure Blob artifact store."
    )
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
