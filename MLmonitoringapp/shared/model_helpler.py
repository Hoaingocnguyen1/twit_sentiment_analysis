import os
import logging
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from azureml.core import Workspace, Model, Run
from azureml.core.authentication import AzureCliAuthentication
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from shared.config import (
    EVALUATION_THRESHOLD, 
    AZURE_SUBSCRIPTION_ID, 
    AZURE_RESOURCE_GROUP, 
    AZURE_ML_WORKSPACE,
    MODEL_REGISTRY_NAME
)

def get_ml_workspace():
    """Lấy Azure ML Workspace"""
    try:
        cli_auth = AzureCliAuthentication()
        workspace = Workspace.get(
            subscription_id=AZURE_SUBSCRIPTION_ID,
            resource_group=AZURE_RESOURCE_GROUP,
            name=AZURE_ML_WORKSPACE,
            auth=cli_auth
        )
        return workspace
    except Exception as e:
        logging.error(f"Error getting ML workspace: {str(e)}")
        return None

def setup_mlflow():
    """Thiết lập MLflow tracking với Azure ML workspace"""
    workspace = get_ml_workspace()
    if workspace:
        mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
        return True
    return False

def get_latest_model():
    """Lấy model mới nhất từ registry"""
    try:
        workspace = get_ml_workspace()
        if not workspace:
            return None
        
        models = Model.list(workspace, name=MODEL_REGISTRY_NAME)
        if not models:
            logging.warning(f"No models found with name {MODEL_REGISTRY_NAME}")
            return None
        
        # Sắp xếp theo version giảm dần và lấy model mới nhất
        latest_model = sorted(models, key=lambda x: x.version, reverse=True)[0]
        logging.info(f"Latest model: {latest_model.name}, version: {latest_model.version}")
        
        return latest_model
    except Exception as e:
        logging.error(f"Error getting latest model: {str(e)}")
        return None

def load_data_from_blob_content(blob_content):
    """
    Tải dữ liệu từ nội dung blob
    
    Args:
        blob_content: Nội dung blob dạng bytes
        
    Returns:
        DataFrame chứa dữ liệu hoặc None nếu lỗi
    """
    try:
        # Giả sử dữ liệu là CSV
        return pd.read_csv(pd.io.common.BytesIO(blob_content))
    except Exception as e:
        logging.error(f"Error loading data from blob content: {str(e)}")
        return None

def evaluate_model(model, test_data):
    """
    Đánh giá mô hình với dữ liệu test
    
    Args:
        model: Model cần đánh giá
        test_data: DataFrame chứa dữ liệu test
        
    Returns:
        dict chứa các metric hoặc None nếu lỗi
    """
    try:
        # Giả sử X_test, y_test đã được tách từ test_data
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        # Dự đoán và tính toán các metric
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        logging.info(f"Model evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        return None

def is_model_performance_acceptable(metrics):
    """
    Kiểm tra xem hiệu suất mô hình có chấp nhận được không
    
    Args:
        metrics: Dict chứa các metric
        
    Returns:
        True nếu hiệu suất chấp nhận được, False nếu không
    """
    if not metrics:
        return False
    
    # Sử dụng f1 score để đánh giá
    return metrics.get('f1', 0) >= EVALUATION_THRESHOLD

def register_model_to_registry(model, metrics):
    """
    Đăng ký model mới vào registry
    
    Args:
        model: Model cần đăng ký
        metrics: Dict chứa các metric
        
    Returns:
        Model đã đăng ký hoặc None nếu lỗi
    """
    try:
        workspace = get_ml_workspace()
        if not workspace:
            return None
        
        # Thiết lập MLflow tracking
        setup_mlflow()
        
        # Bắt đầu run mới
        with mlflow.start_run() as run:
            # Log các metric
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            model_uri = f"runs:/{run.info.run_id}/model"
            
            # Đăng ký model với Azure ML
            registered_model = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
            
            logging.info(f"Model registered: {registered_model.name} version {registered_model.version}")
            return registered_model
    except Exception as e:
        logging.error(f"Error registering model: {str(e)}")
        return None

def train_model(train_data):
    """
    Huấn luyện mô hình mới
    
    Args:
        train_data: DataFrame chứa dữ liệu huấn luyện
        
    Returns:
        Model đã huấn luyện hoặc None nếu lỗi
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Giả sử X_train, y_train đã được tách từ train_data
        X_train = train_data.drop('target', axis=1)
        y_train = train_data['target']
        
        # Tạo và huấn luyện model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        logging.info("Model training completed")
        return model
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        return None