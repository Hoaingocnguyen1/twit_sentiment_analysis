from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from config.dataClient import get_db
from datetime import datetime, timedelta


class DeleteOldDataOperator(BaseOperator):
    """
    Custom Operator to delete old data from specified tables in the database.
    """

    @apply_defaults
    def __init__(self, table_names: list, date_column: str, days_to_keep: int = 7, *args, **kwargs):
        """
        :param table_names: List of table names to delete data from.
        :param date_column: Name of the date column to filter old data.
        :param days_to_keep: Number of days to keep data (default is 7 days).
        """
        super().__init__(*args, **kwargs)
        self.table_names = table_names
        self.date_column = date_column
        self.days_to_keep = days_to_keep

    def execute(self, context):
        """
        Execute the deletion of old data.
        """
        # Tính toán ngày cần xóa
        last_week = (datetime.now() - timedelta(days=self.days_to_keep)).strftime('%Y-%m-%d')
        db = get_db()
        # Log thông tin
        self.log.info(f"Deleting data older than {last_week} from tables: {self.table_names}")

        # Thực hiện xóa dữ liệu cho từng bảng
        try:
            for table_name in self.table_names:
                db.delete_data(
                    table_name=table_name,
                    date_column=self.date_column,
                    date_value=last_week
                )
                self.log.info(f"Deleted data older than {last_week} from table {table_name}.")
        except Exception as e:
            self.log.error(f"Error while deleting data: {e}")
            raise
        finally:
            db.close()