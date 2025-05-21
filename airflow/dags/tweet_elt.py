from datetime import datetime as dt
from datetime import timedelta
from airflow import DAG
import sys

print(f"Python path: {sys.path}")
import os

print(f"Working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")
print(f"Files in plugins: {os.listdir('/opt/airflow/plugins')}")


from plugins.operators.CrawlTweetsOperator import CrawlTweetsOperator
from plugins.operators.PushDatalakeOperator import PushDatalakeOperator
from plugins.operators.pullStagingOperator import PullStagingOperator
from plugins.sensors.lakeSensor import AzureBlobNewFileSensor
from plugins.operators.SentimentPredictionOperator import SentimentPredictionOperator
from config.constants import BASE_DIR, TWITTER_HASHTAGS
from dotenv import load_dotenv

# ==== 1. Base Directory Setup ====
load_dotenv()
BASE_DIR = os.getenv("BASE_DIR", "/opt/airflow/data_cache")
RAW_DIR = os.path.join(BASE_DIR, "raw")
# TWITTER_HASHTAGS = os.getenv('TWITTER_HASHTAGS')
CONFIG_DIR = os.path.join('/opt/airflow', "config")

# ==== 2. Default Args ====
default_args = {
    "owner": "admin",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ==== 3. DAG Definition ====
with DAG(
    "tweet_ETL",
    default_args=default_args,
    description="""ELT pipeline: crawl tweets, upload to blob, wait, then load to staging DB.""",
    schedule_interval="0 * * * *",
    start_date=dt(2025, 5, 20, 0, 0)
    catchup=False,
    tags=["twitter", "etl"],
) as dag:

    # ==== 4. File Naming Template ====

    file_name_template = "raw_{{ execution_date.in_timezone('Asia/Bangkok').strftime('%Y%m%d_%H') }}.json"
    raw_file_path_template = ("/opt/airflow/data_cache/raw/raw_{{ execution_date.in_timezone('Asia/Bangkok').strftime('%Y%m%d_%H') }}.json")
    blob_path_template = "raw/raw_{{ execution_date.in_timezone('Asia/Bangkok').strftime('%Y%m%d_%H') }}.json"

    # ==== 5. Crawl Tweets Task ====
    crawl_tweets_task = CrawlTweetsOperator(
        task_id="crawl_tweets",
        base_dir=RAW_DIR,
        output_file=file_name_template,
        hashtags_dict=TWITTER_HASHTAGS,
    )

    # ==== 6. Push to Azure Blob ====
    push_datalake_task = PushDatalakeOperator(
        task_id="push_to_blob",
        container_name="data",
        blob_name=blob_path_template,
        file_path=raw_file_path_template,
    )

    # ==== 7. Check for File in Blob ====
    check_new_files = AzureBlobNewFileSensor(
        task_id="check_new_files",
        container_name="data",
        folder_path="raw",
        last_processed_file=None,
        poke_interval=30,
        timeout=600,
    )

    # ==== 8. Load to Staging ====
    pull_staging_task = PullStagingOperator(
        task_id="pull_staging",
        container_name="data",
        file_path=blob_path_template,
        column_mapping_path=os.path.join(CONFIG_DIR, "columns_map_pull.json"),
    )

    # ==== 9. DAG Flow ====
    crawl_tweets_task >> push_datalake_task >> check_new_files >> pull_staging_task