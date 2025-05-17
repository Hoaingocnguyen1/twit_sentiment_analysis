from baseDatabase import BlobStorage
from dotenv import load_dotenv
import os
import json
import tempfile
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO
)

# Define containers and required blobs
setup = {
'data': ['raw', 'clean'],
'artifact': [],  # add blobs as needed
'logs': []
}


if __name__ == '__main__':
    load_dotenv()  # Load environment variables from .env file
    logging.info("Loading environment variables...")
    # Load environment variables
    conn_str = os.getenv('LAKE_STORAGE_CONN_STR')
    if not conn_str:
        raise EnvironmentError("LAKE_STORAGE_CONN_STR not found in environment variables.")
    
    blob_storage = BlobStorage(conn_str)
    logging.info("Datalake connection is completed.")

    # Create containers and empty blobs if not exist
    for container, blobs in setup.items():
        blob_storage.create_container(container)
        logging.info(f"Container {container} is completed.")
        for blob in blobs:
            blob_storage.create_folder_structure(container, blob)
            logging.info(f"Created blob {blob} in container {container}.")

    print("Blob storage setup complete.")