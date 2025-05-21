import logging
import json
import azure.functions as func
from shared.blob_helper import download_blob_content
from shared.model_helper import load_data_from_blob_content
from shared.queue_helper import decode_queue_message, send_message_to_queue

def main(msg: func.QueueMessage) -> None:
    """
    Azure Function được kích hoạt khi có message mới trong queue
    Xử lý nội dung file từ blob và chuẩn bị dữ liệu cho bước đánh giá
    """
    try:
        # Giải mã message
        message_data = decode_queue_message(msg)
        if not message_data or not isinstance(message_data, dict):
            logging.error("Invalid message format")
            return
        
        blob_name = message_data.get('blob_name')
        container_name = message_data.get('container_name')
        
        if not blob_name or not container_name:
            logging.error("Missing blob_name or container_name in message")
            return
        
        logging.info(f"Processing blob: {blob_name} from container: {container_name}")
        
        # Tải nội dung blob
        blob_content = download_blob_content(blob_name, container_name)
        if not blob_content:
            logging.error(f"Failed to download blob content for {blob_name}")
            return
        
        # Tải dữ liệu từ nội dung blob
        data = load_data_from_blob_content(blob_content)
        if data is None:
            logging.error(f"Failed to load data from blob {blob_name}")
            return
        
        logging.info(f"Successfully loaded data from blob {blob_name} with {len(data)} rows")
        
        # Gửi thông tin file đã xử lý vào queue cho bước tiếp theo
        evaluation_message = {
            'blob_name': blob_name,
            'container_name': container_name,
            'rows_count': len(data),
            'columns': list(data.columns),
            'has_target': 'target' in data.columns
        }
        
        success = send_message_to_queue(evaluation_message, queue_name="model-evaluation-queue")
        if success:
            logging.info(f"Added file {blob_name} to evaluation queue")
        else:
            logging.error(f"Failed to add file {blob_name} to evaluation queue")
            
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")