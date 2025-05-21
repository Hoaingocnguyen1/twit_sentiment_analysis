from datetime import datetime as dt
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from config.dataClient import get_db
import logging
import json
import requests

default_args = {
    "owner": "admin",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
logger = logging.getLogger(__name__)

def get_data_from_db(**kwargs):
    db = get_db()

    rows = db.read(
        table_name='TWEET_STAGING',
        columns=["tweet_id", "topic", "content", "created_at"]
    )

    kwargs['ti'].xcom_push(key='tweets', value=rows)


def get_sentiment_from_api(**kwargs):

    rows = kwargs['ti'].xcom_pull(key='tweets', task_ids='get_data')

    if not rows:
        logger.info("No new records to process.")
        return 0

    logger.info(f"Processing {len(rows)} tweets")

    contents = [row[2] for row in rows]
    payload = {"texts": contents}
    response = requests.post("http://localhost:8000/api/v1/predict", json=payload)

    if response.status_code != 200:
        logger.error(f"Prediction API failed: {response.status_code} - {response.text}")
        return 0
    
    all_predictions = response.json()

    logger.info(f"Finished inference for {len(rows)} tweets")

    kwargs['ti'].xcom_push(key='predictions', value=all_predictions)

def insert_into_product_db(**kwargs):

    rows = kwargs['ti'].xcom_pull(key='tweets', task_ids='get_data')
    all_predictions = kwargs['ti'].xcom_pull(key='predictions', task_ids='get_sentiment')

    data_to_insert = [
            {
                "staging_tweet_id": row[0],
                "topic": row[1],
                "predicted_sentiment": pred,
                "created_at": row[3]
            }
            for row, pred in zip(rows, all_predictions)
        ]
    
    db = get_db()
    try:
        db.batch_insert(
            table_name="TWEET_PRODUCT",
            rows=data_to_insert,
            column_mapping={
                "staging_tweet_id": "staging_tweet_id",
                "topic": "topic",
                "predicted_sentiment": "predicted_sentiment",
                "created_at": "created_at"
            }
        )
        logger.info("Finished pushing tweets into product table.")
    except Exception as e:
        logger.error(f"Failed to push tweets into product table. str{e}")

def update_staging_db(**kwargs):
    rows = kwargs['ti'].xcom_pull(key='tweets', task_ids='get_data')

    db = get_db()
    try:
        tweet_ids = [row[0] for row in rows]
        for tweet_id in tweet_ids:
            db.update(
                table_name='TWEET_STAGING',
                data={"moved_to_product": True},
                conditions={"tweet_id": tweet_id}
            )
        logger.info("Finished updating staging table.")
    except Exception as e:
        logger.error(f"Failed to update staging table. str{e}")

with DAG(
    "dashboard_ingestion",
    default_args=default_args,
    description="""Pipeline that ingests data from database, send a request to the API for inference, and push it into the production table for dashboard.""",
    schedule="@daily",
    start_date=dt(2025, 5, 20, 0, 0),
    catchup=False,
    tags=["twitter", "api", "ingestion", "dashboard"],
) as dag:
    
    get_data_from_db_task = PythonOperator(
        task_id="get_data",
        python_callable=get_data_from_db,
    )

    get_sentiment_from_api_task = PythonOperator(
        task_id="get_sentiment",
        python_callable=get_sentiment_from_api,
    )

    insert_into_product_db_task = PythonOperator(
        task_id="insert_into_product_db",
        python_callable=insert_into_product_db,
    )

    update_staging_db_task = PythonOperator(
        task_id="update_staging_db",
        python_callable=update_staging_db,
    )

    get_data_from_db_task >> get_sentiment_from_api_task >> insert_into_product_db_task >> update_staging_db_task
