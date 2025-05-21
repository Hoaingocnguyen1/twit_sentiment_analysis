import json
import logging
import base64
from azure.storage.queue import QueueServiceClient, QueueClient, BinaryBase64EncodePolicy
from shared.config import STORAGE_CONNECTION_STRING, DATA_PROCESSING_QUEUE

def get_queue_service_client():
    """Khởi tạo QueueServiceClient từ connection string"""
    return QueueServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

def get_queue_client(queue_name=DATA_PROCESSING_QUEUE):
    """Lấy queue client cho queue cụ thể"""
    queue_service_client = get_queue_service_client()
    return queue_service_client.get_queue_client(queue_name)

def ensure_queue_exists(queue_name=DATA_PROCESSING_QUEUE):
    """Đảm bảo queue đã tồn tại, nếu chưa thì tạo mới"""
    queue_service_client = get_queue_service_client()
    queues = [q.name for q in queue_service_client.list_queues()]
    
    if queue_name not in queues:
        logging.info(f"Creating queue: {queue_name}")
        queue_service_client.create_queue(queue_name)
    
    return get_queue_client(queue_name)

def send_message_to_queue(message_content, queue_name=DATA_PROCESSING_QUEUE):
    """
    Gửi message tới queue
    
    Args:
        message_content: Nội dung message (dict hoặc str)
        queue_name: Tên của queue
    
    Returns:
        True nếu gửi thành công, False nếu thất bại
    """
    try:
        queue_client = ensure_queue_exists(queue_name)
        
        # Nếu message là dict thì chuyển thành JSON string
        if isinstance(message_content, dict):
            message_content = json.dumps(message_content)
            
        # Mã hóa Base64 để tránh các vấn đề với ký tự đặc biệt
        encoded_content = base64.b64encode(message_content.encode('utf-8')).decode('utf-8')
        
        queue_client.send_message(encoded_content)
        return True
    except Exception as e:
        logging.error(f"Error sending message to queue {queue_name}: {str(e)}")
        return False

def decode_queue_message(message):
    """Giải mã message từ queue"""
    try:
        decoded_content = base64.b64decode(message.content).decode('utf-8')
        try:
            # Thử parse JSON
            return json.loads(decoded_content)
        except json.JSONDecodeError:
            # Nếu không phải JSON thì trả về string
            return decoded_content
    except Exception as e:
        logging.error(f"Error decoding queue message: {str(e)}")
        return None