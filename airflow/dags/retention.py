from airflow import DAG
from plugins.operators.RetentionOperator import DeleteOldDataOperator
from datetime import datetime as dt, timedelta


default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'delete_old_data',
    default_args=default_args,
    description='A DAG to delete old data from the database at the end of the day',
    schedule_interval='59 23 * * *',  # Chạy vào 23:59 mỗi ngày
    start_date = dt.now() - timedelta(minutes=1),
    catchup=False,
) as dag:

    delete_old_data_task = DeleteOldDataOperator(
        task_id='delete_old_data',
        table_names=['TWEET_STAGING', 'TWEET_PRODUCT'],  # Các bảng cần xóa
        date_column='date_created',  # Cột ngày để lọc dữ liệu
        days_to_keep=7  # Số ngày cần giữ lại dữ liệu
    )

delete_old_data_task