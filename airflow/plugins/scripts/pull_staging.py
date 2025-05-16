from config.dataClient import get_blob_storage, get_db
import os
import logging
import sys
sys.path.append('/opt/airflow')
from src.data import clean_data_v2
import json


def pull_staging(container_name, file_path, column_mapping_path, **context):
    """
    Pull data from Azure Blob Storage, clean it, and insert it into the staging database.
    :param context: Airflow context dictionary containing parameters like conn_str, container, file_path, and db_config.
    """
    try:
        if not os.path.exists(column_mapping_path):
            raise FileNotFoundError(f"Column mapping file '{column_mapping_path}' does not exist.")
        with open(column_mapping_path, 'r', encoding='utf-8') as f:
            column_mapping = json.load(f)

        blob_storage = get_blob_storage()
        # Đọc dữ liệu từ Blob Storage
        logging.info(f"Reading data from container '{container_name}', file '{file_path}'")
        data = blob_storage.read_json_from_container(
            container=container_name, file_path=file_path
        )

        # Làm sạch dữ liệu
        texts = [item['text'] for item in data]
        cleaned_texts = clean_data_v2(texts)

        # Cập nhật dữ liệu đã làm sạch
        for i, item in enumerate(data):
            item['text'] = cleaned_texts[i]

        db=get_db()

        # Chèn dữ liệu vào bảng staging
        db.batch_insert(table_name="TWEET_STAGING", tweets_data=data, column_mapping=column_mapping)
        logging.info("Data successfully inserted into TWEET_STAGING.")

    except Exception as e:
        logging.error(f"Error in pull_staging: {e}")
        raise
