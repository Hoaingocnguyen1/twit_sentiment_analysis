import logging
import json
import azure.functions as func
from shared.blob_helper import download_blob_content
from shared.model_helper import (
    load_data_from_blob_content,
    train_model,
    evaluate_model,
    register_model_to_registry,
    setup_mlflow
)
from shared.queue_helper import decode_queue_message

def main(msg: func.QueueMessage) -> None:
    """
    Azure Function được kích hoạt khi có message mới trong queue huấn luyện
    Huấn luyện lại mô hình với dữ liệu mới
    Đánh giá và đăng ký mô hình vào registry nếu hiệu suất được cải thiện
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
        current_metrics = message_data.get('metrics', {})
        current_model_version = message_data.get('current_model_version')
        
        if not blob_name or not container_name:
            logging.error("Missing blob_name or container_name in message")
            return
        
        logging.info(f"Training new model with data from blob: {blob_name}")
        
        # Tải nội dung blob
        blob_content = download_blob_content(blob_name, container_name)
        if not blob_content:
            logging.error(f"Failed to download blob content for {blob_name}")
            return
        
        # Tải dữ liệu từ nội dung blob
        train_data = load_data_from_blob_content(blob_content)
        if train_data is None:
            logging.error(f"Failed to load data from blob {blob_name}")
            return
        
        # Huấn luyện model mới
        new_model = train_model(train_data)
        if not new_model:
            logging.error("Failed to train new model")
            return
        
        logging.info("Model training completed successfully")
        
        # Đánh giá model mới với chính dữ liệu huấn luyện
        # (trong thực tế, nên có riêng tập validation)
        new_metrics = evaluate_model(new_model, train_data)
        if not new_metrics:
            logging.error("Failed to evaluate new model")
            return
        
        logging.info(f"New model metrics: {new_metrics}")
        
        # So sánh với hiệu suất của model cũ
        improved = False
        if current_metrics:
            # So sánh f1 score
            if new_metrics.get('f1', 0) > current_metrics.get('f1', 0):
                improved = True
                logging.info(f"New model has better performance (F1: {new_metrics.get('f1')} vs {current_metrics.get('f1')})")
            else:
                logging.info(f"New model does not improve performance (F1: {new_metrics.get('f1')} vs {current_metrics.get('f1')})")
        else:
            # Không có metrics cũ để so sánh, coi như cải thiện
            improved = True
        
        # Nếu model mới cải thiện hiệu suất, đăng ký vào registry
        if improved:
            registered_model = register_model_to_registry(new_model, new_metrics)
            if registered_model:
                logging.info(f"New model registered successfully. New version: {registered_model.version}")
            else:
                logging.error("Failed to register new model")
        else:
            logging.info("New model not registered as it does not improve performance")
            
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")