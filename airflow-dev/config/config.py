"""
Configuration settings for ML training pipeline
"""
import os
from airflow.models import Variable


class MLConfig:
    """Configuration class for ML pipeline settings"""
    
    # Model Configuration
    MODEL_NAME = Variable.get("model_name", default_var="sentiment-classifier")
    BASE_MODEL_PATH = Variable.get("base_model_path", default_var="bert-base-uncased")
    
    # Training Parameters
    BATCH_SIZE = int(Variable.get("batch_size", default_var="32"))
    EPOCHS = int(Variable.get("epochs", default_var="3"))
    LEARNING_RATE = float(Variable.get("learning_rate", default_var="2e-5"))
    MAX_LENGTH = int(Variable.get("max_length", default_var="128"))
    
    # Validation Parameters
    MIN_ACCURACY = float(Variable.get("min_accuracy", default_var="0.8"))
    MIN_F1_SCORE = float(Variable.get("min_f1_score", default_var="0.75"))
    
    # MLflow Configuration
    EXPERIMENT_NAME = Variable.get("experiment_name", default_var="ml-training-pipeline")
    TRACKING_URI = os.getenv("TRACKING_URI", "http://172.23.51.243:5000")
    
    # Storage Configuration
    AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "testartifact")
    LAKE_STORAGE_CONN_STR = os.getenv("LAKE_STORAGE_CONN_STR")
    
    # Data Paths
    TRAIN_DATA_PATH = "data/train_dataset.json"
    TEST_DATA_PATH = "data/test_dataset.json"
    
    # Required Environment Variables
    REQUIRED_ENV_VARS = [
        'LAKE_STORAGE_CONN_STR',
        'AZURE_BLOB_CONTAINER',
        'TRACKING_URI'
    ]
    
    # Notification Configuration
    NOTIFICATION_CHANNELS = {
        'email': Variable.get("notification_email", default_var=None),
        'slack_webhook': Variable.get("slack_webhook", default_var=None)
    }
    
    @classmethod
    def get_training_config(cls):
        """Get training configuration as dictionary"""
        return {
            'model_name': cls.MODEL_NAME,
            'base_model_path': cls.BASE_MODEL_PATH,
            'batch_size': cls.BATCH_SIZE,
            'epochs': cls.EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'max_length': cls.MAX_LENGTH,
            'experiment_name': cls.EXPERIMENT_NAME
        }
    
    @classmethod
    def get_validation_config(cls):
        """Get validation configuration as dictionary"""
        return {
            'min_accuracy': cls.MIN_ACCURACY,
            'min_f1_score': cls.MIN_F1_SCORE
        }
    
    @classmethod
    def validate_environment(cls):
        """Validate required environment variables"""
        missing_vars = []
        for var in cls.REQUIRED_ENV_VARS:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        return True