import sys
from datetime import datetime, timedelta

from airflow import DAG

print(f"Python path: {sys.path}")
import os

print(f"Working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")
print(f"Files in plugins: {os.listdir('/opt/airflow/plugins')}")

from plugins.operators.TweetOperator import (
    ExtractDataOperator,
    LabelTweetOperator,
    SaveDataToBlobOperator,
    SplitTrainTestOperator,
)

# ==== 2. Default Args ====
default_args = {
    "owner": "admin",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "start_date": datetime(2025, 5, 20),
    "retry_delay": timedelta(minutes=1),
}

# Khởi tạo DAG
dag = DAG(
    "tweet_sentiment_data_processing_artifact",
    default_args=default_args,
    description="Process tweet data for sentiment analysis",
    schedule_interval=timedelta(days=5),
    catchup=False,
    max_active_runs=1,
    tags=["sentiment", "ml", "data-processing"],
)

# Định nghĩa các task
extract_task = ExtractDataOperator(task_id="extract_data_from_db", dag=dag)

label_task = LabelTweetOperator(task_id="label_tweet_data", dag=dag)

split_task = SplitTrainTestOperator(task_id="split_train_test", dag=dag)

save_task = SaveDataToBlobOperator(task_id="save_data_to_blob", dag=dag)

# Thiết lập dependencies
extract_task >> label_task >> split_task >> save_task
