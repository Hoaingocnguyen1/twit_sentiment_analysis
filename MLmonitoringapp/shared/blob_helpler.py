import re
import logging
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from shared.config import STORAGE_CONNECTION_STRING, DATA_CONTAINER_NAME, DATA_FILE_PATTERN

def get_blob_service_client():
    """Khởi tạo BlobServiceClient từ connection string"""
    return BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

def list_blobs_in_container(container_name=DATA_CONTAINER_NAME):
    """Liệt kê tất cả các blob trong container"""
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)
    return list(container_client.list_blobs())

def filter_blobs_by_pattern_and_date_range(blobs, start_date, end_date):
    """
    Lọc các blob theo pattern và thời gian tạo
    
    Args:
        blobs: Danh sách các blob
        start_date: Thời gian bắt đầu
        end_date: Thời gian kết thúc
    
    Returns:
        Danh sách các blob phù hợp với điều kiện
    """
    pattern = re.compile(DATA_FILE_PATTERN)
    filtered_blobs = []
    
    for blob in blobs:
        if pattern.match(blob.name):
            # Lấy ngày tạo từ tên file: clean_YYYYMMDD_iii
            try:
                date_part = blob.name.split('_')[1]
                blob_date = datetime.strptime(date_part, '%Y%m%d')
                
                if start_date <= blob_date <= end_date:
                    filtered_blobs.append(blob)
            except (IndexError, ValueError) as e:
                logging.warning(f"Could not parse date from blob {blob.name}: {str(e)}")
                continue
    
    return filtered_blobs

def download_blob_content(blob_name, container_name=DATA_CONTAINER_NAME):
    """Tải nội dung của một blob"""
    blob_service_client = get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    download_stream = blob_client.download_blob()
    return download_stream.readall()

def parse_date_from_blob_name(blob_name):
    """Trích xuất ngày từ tên blob (clean_YYYYMMDD_iii)"""
    try:
        parts = blob_name.split('_')
        if len(parts) >= 2:
            date_part = parts[1]
            return datetime.strptime(date_part, '%Y%m%d')
    except (IndexError, ValueError) as e:
        logging.error(f"Error parsing date from blob name {blob_name}: {str(e)}")
    
    return None