import os
import logging
import sys
sys.path.append('/opt/airflow')
from config.dataClient import get_blob_storage

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def push_datalake(container_name, blob_name, file_path,**context):
    """
    Upload a local file to Azure Blob Storage.

    :param context: Airflow context dictionary containing parameters like container_name, blob_name, base_dir, and file.
    """
    try:
        # Kiểm tra file cục bộ
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local file '{file_path}' does not exist.")

        # Kiểm tra nếu file rỗng
        if os.stat(file_path).st_size == 0:
            raise ValueError(f"File '{file_path}' is empty and will not be uploaded.")

        blob_storage = get_blob_storage()

        # Tải file lên Blob Storage
        blob_url = blob_storage.upload_file(
            container_name=container_name,
            blob_name=blob_name,
            file_path=file_path,
            overwrite=True
        )
        logging.info(f"File uploaded successfully to Blob Storage")
        return blob_url

    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {fnf_error}")
        raise
    except ValueError as val_error:
        logging.error(f"Value error: {val_error}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise
