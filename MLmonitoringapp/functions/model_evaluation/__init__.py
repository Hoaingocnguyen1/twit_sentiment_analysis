
import logging
import json
import azure.functions as func
from shared.blob_helper import download_blob_content
from shared.model_helper import (
    get_latest_model, 
    load_data_from_blob_content, 
    evaluate_model, 
    is_model_performance_acceptable,
    setup_mlflow
)
from shared.queue_helper import decode_queue_message, send_message_to_queue

def main(msg: func.QueueMessage) -> None:
    """
    Azure Function được kích hoạt khi có message mới trong queue đánh giá
    Đánh giá hiệu suất mô hình hiện tại với dữ liệu mới
    Quyết định xem có cần huấn luyện lại mô hình hay không
    """
    try:
        # Thiết lập MLflow
        setup_mlflow()
        
        # Giải mã message
        message_data = decode_queue_message(msg)
        if not message_data or not isinstance(message_data, dict):
            logging.error("Invalid message format")
            return
        
        blob_name = message_data.get('blob_name')
        container_name = message_data.get('container_name')
        has_target = message_data.get('has_target', False)
        
        if not blob_name or not container_name or not has_target:
            logging.error("Missing required fields in message or target column not found")
            return
        
        logging.info(f"Evaluating model with data from blob: {blob_name}")
        
        # Lấy model mới nhất từ registry
        latest_model = get_latest_model()
        if not latest_model:
            logging.error("No model found in registry")
            return
        
        # Tải nội dung blob
        blob_content = download_blob_content(blob_name, container_name)
        if not blob_content:
            logging.error(f"Failed to download blob content for {blob_name}")
            return
        
        # Tải dữ liệu từ nội dung blob
        test_data = load_data_from_blob_content(blob_content)
        if test_data is None:
            logging.error(f"Failed to load data from blob {blob_name}")
            return
        
        # Đánh giá model với dữ liệu test
        metrics = evaluate_model(latest_model, test_data)
        if not metrics:
            logging.error("Failed to evaluate model")
            return
        
        logging.info(f"Model evaluation metrics: {metrics}")
        
        # Kiểm tra xem hiệu suất model có đạt threshold không
        if not is_model_performance_acceptable(metrics):
            logging.info(f"Model performance below threshold. Scheduling retraining.")
            
            # Gửi message để trigger huấn luyện lại model
            training_message = {
                'blob_name': blob_name,
                'container_name': container_name,
                'metrics': metrics,
                'current_model_version': latest_model.version
            }
            
            success = send_message_to_queue(training_message, queue_name="model-training-queue")
            if success:
                logging.info("Model retraining scheduled")
            else:
                logging.error("Failed to schedule model retraining")
        else:
            logging.info("Model performance is acceptable. No retraining needed.")
            
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")