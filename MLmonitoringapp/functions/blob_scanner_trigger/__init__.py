import logging
import datetime
import azure.functions as func
from shared.config import get_date_range, DATA_CONTAINER_NAME
from shared.blob_helper import list_blobs_in_container, filter_blobs_by_pattern_and_date_range
from shared.queue_helper import send_message_to_queue

def main(timer: func.TimerRequest) -> None:
    """
    Azure Function được kích hoạt theo timer, chạy mỗi 3 ngày
    Quét Blob Storage để tìm các file dữ liệu mới phù hợp với pattern
    và gửi chúng vào queue để xử lý
    """
    if timer.past_due:
        logging.info('The timer is past due!')

    logging.info('Blob Scanner Timer trigger function executed at %s', datetime.datetime.utcnow())
    
    try:
        # Lấy khoảng thời gian để quét: 5 ngày gần nhất
        start_date, end_date = get_date_range()
        logging.info(f"Scanning blobs from {start_date} to {end_date}")
        
        # Liệt kê tất cả các blob trong container
        all_blobs = list_blobs_in_container(DATA_CONTAINER_NAME)
        logging.info(f"Found {len(all_blobs)} total blobs in container")
        
        # Lọc các blob theo pattern và thời gian
        filtered_blobs = filter_blobs_by_pattern_and_date_range(all_blobs, start_date, end_date)
        logging.info(f"Found {len(filtered_blobs)} blobs matching pattern and date range")
        
        # Gửi các blob vào queue để xử lý
        for blob in filtered_blobs:
            message = {
                'blob_name': blob.name,
                'container_name': DATA_CONTAINER_NAME,
                'scanned_at': datetime.datetime.utcnow().isoformat()
            }
            
            success = send_message_to_queue(message)
            if success:
                logging.info(f"Added blob {blob.name} to processing queue")
            else:
                logging.error(f"Failed to add blob {blob.name} to queue")
        
        logging.info("Blob scanning completed successfully")
    except Exception as e:
        logging.error(f"Error in blob scanner function: {str(e)}")