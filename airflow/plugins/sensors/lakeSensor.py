from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
# from ..config.dataClient import blob_storage
import sys
sys.path.append('/opt/airflow') 
from config.dataClient import get_blob_storage
import logging

class AzureBlobNewFileSensor(BaseSensorOperator):
    """
    Custom Sensor to check for new files in a specific folder in Azure Blob Storage.
    """
    @apply_defaults
    def __init__(self, container_name: str, folder_path: str, last_processed_file: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.container_name = container_name
        self.folder_path = folder_path.rstrip('/')  # Ensure no trailing slash
        self.last_processed_file = last_processed_file

    def poke(self, context):
        """
        Check if there are new files in the specified folder.
        """
        logging.info(f"Checking for new files in folder '{self.folder_path}' in container '{self.container_name}'")
        # List all blobs in the folder
        blob_storage = get_blob_storage()
        blobs = blob_storage.list_blobs_by_path(container=self.container_name, path=self.folder_path)
        new_files = []

        for blob in blobs:
            if self.last_processed_file is None or blob > self.last_processed_file:
                new_files.append(blob)

        if new_files:
            logging.info(f"New files found: {new_files}")
            # Update the context with the latest file
            context['ti'].xcom_push(key='new_files', value=new_files)
            return True
        else:
            logging.info("No new files found.")
            return False