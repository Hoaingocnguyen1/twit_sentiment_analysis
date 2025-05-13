from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from plugins.scripts.push_datalake import push_datalake


class PushDatalakeOperator(BaseOperator):
    """
    Custom Operator to upload a local file to Azure Blob Storage.
    """
    template_fields = ['blob_name', 'file_path']

    @apply_defaults
    def __init__(self, container_name: str, blob_name: str, file_path: str, *args, **kwargs):
        """
        :param container_name: Name of the container in Azure Blob Storage.
        :param blob_name: Name of the blob to upload.
        :param file_path: Path to the local file to upload.
        """
        super().__init__(*args, **kwargs)
        self.container_name = container_name
        self.blob_name = blob_name
        self.file_path = file_path

    def execute(self, context):
        """
        Execute the push to Azure Blob Storage.
        """
        # Xử lý các biến động từ context
        evaluated_file_path = self.file_path
        evaluated_blob_name = self.blob_name

        # Log thông tin
        self.log.info(f"Starting upload to Azure Blob Storage.")
        self.log.info(f"Local file: {evaluated_file_path}")
        self.log.info(f"Blob name: {evaluated_blob_name}")
        self.log.info(f"Container name: {self.container_name}")

        # Gọi hàm push_datalake
        try:
            push_datalake(
                container_name=self.container_name,
                blob_name=evaluated_blob_name,
                file_path=evaluated_file_path,
                **context,
            )
            self.log.info("Push to Azure Blob Storage completed successfully.")
        except Exception as e:
            self.log.error(f"Error during push to Azure Blob Storage: {e}")
            raise
