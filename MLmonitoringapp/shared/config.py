import os
from datetime import datetime, timedelta

# Cấu hình Storage
STORAGE_CONNECTION_STRING = os.environ.get("STORAGE_CONNECTION_STRING")
DATA_CONTAINER_NAME = os.environ.get("DATA_CONTAINER_NAME", "ml-data")
DATA_PROCESSING_QUEUE = os.environ.get("DATA_PROCESSING_QUEUE", "data-processing-queue")

# Cấu hình ML
EVALUATION_THRESHOLD = float(os.environ.get("EVALUATION_THRESHOLD", "0.75"))
DAYS_TO_SCAN = int(os.environ.get("DAYS_TO_SCAN", "5"))
MODEL_REGISTRY_NAME = os.environ.get("MODEL_REGISTRY_NAME", "ml-model-registry")

# Cấu hình Azure ML
AZURE_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP")
AZURE_ML_WORKSPACE = os.environ.get("AZURE_ML_WORKSPACE")

# Định dạng file data
DATA_FILE_PREFIX = "clean_"
DATA_FILE_PATTERN = DATA_FILE_PREFIX + r"\d{8}_\d{3}"  # Mẫu định dạng: clean_20250521_001

# Hàm tiện ích
def get_date_range(days=DAYS_TO_SCAN):
    """Lấy khoảng thời gian từ now - days đến hiện tại"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date