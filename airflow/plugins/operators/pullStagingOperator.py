from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
# from plugins.scripts.pull_staging import pull_staging
from plugins.scripts.pull_staging import pull_staging

class PullStagingOperator(BaseOperator):
    """
    Custom Operator to pull data from Azure Blob Storage, clean it, and insert it into the staging database.
    """

    template_fields = ['file_path']  # Hỗ trợ template variables

    @apply_defaults
    def __init__(self, container_name: str, file_path: str, column_mapping_path: str, *args, **kwargs):
        """
        :param container: Name of the container in Azure Blob Storage.
        :param file_path: Path to the file in Blob Storage.
        :param column_mapping_path: Path to the column mapping JSON file.
        """
        super().__init__(*args, **kwargs)
        self.container_name = container_name
        self.file_path = file_path
        self.column_mapping_path = column_mapping_path

    def execute(self, context):
        """
        Execute the pull_staging script.
        """
        # Xử lý file_path với các biến động từ context
        evaluated_file_path = self.file_path
        evaluated_column_mapping_path = self.column_mapping_path

        # Log thông tin
        self.log.info(f"Pulling data from container '{self.container_name}'")
        self.log.info(f"File path: {evaluated_file_path}")
        self.log.info(f"Column mapping path: {evaluated_column_mapping_path}")

        # Gọi hàm pull_staging
        try:
            pull_staging(
                container_name=self.container_name,
                file_path=evaluated_file_path,
                column_mapping_path=evaluated_column_mapping_path,
                **context,
            )
            self.log.info("Staging load completed successfully.")
        except Exception as e:
            self.log.error(f"Error during staging load: {e}")
            raise
