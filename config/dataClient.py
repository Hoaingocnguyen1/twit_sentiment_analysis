from src.database import BlobStorage, Database
import os
from dotenv import load_dotenv

load_dotenv()
# ==== 1. Base Directory Setup ====
import os
from src.database.baseDatabase import BlobStorage, Database

def get_blob_storage():
    conn_str = os.getenv('LAKE_STORAGE_CONN_STR')
    if not conn_str:
        raise ValueError("Missing LAKE_STORAGE_CONN_STR environment variable")
    return BlobStorage(conn_str)

def get_db():
    return Database(
        host=os.getenv('PG_HOST'),
        port=os.getenv('PG_PORT'),
        dbname=os.getenv('PG_DB1'),
        user=os.getenv('PG_USER'),
        password=os.getenv('PG_PASSWORD'),
        sslmode=os.getenv('PG_SSLMODE')
    )