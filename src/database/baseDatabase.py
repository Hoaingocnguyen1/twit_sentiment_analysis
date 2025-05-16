import psycopg2
from psycopg2 import sql, extras
from azure.storage.blob import BlobServiceClient
import pandas as pd
from typing import List, Dict, Optional, Union, Any, Tuple
import json
import os
import logging

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, **kwargs):
        """
        Initialize the database connection.
        """
        try:
            self.connection = psycopg2.connect(
                host=kwargs.get("host"),
                port=kwargs.get("port"),
                dbname=kwargs.get("dbname"),
                user=kwargs.get("user"),
                password=kwargs.get("password"),
                sslmode=kwargs.get("sslmode", "prefer")
            )
            self.cursor = self.connection.cursor()
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            raise

    def create_table(self, table_name: str, columns: Dict[str, str]):
        """
        Create a table with the specified columns.

        :param table_name: Name of the table.
        :param columns: Dictionary of column names and their data types.
        """
        column_defs = [
            f"{col} {dtype}" if "FOREIGN KEY" not in dtype else f"{col} {dtype.split(', FOREIGN KEY ')[0]} {dtype.split(', FOREIGN KEY ')[1]}"
            for col, dtype in columns.items()
        ]
        query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(map(sql.SQL, column_defs))
        )
        self._execute_query(query, f"Error creating table {table_name}")

    def insert(self, table_name: str, data: Dict[str, any]):
        """
        Insert a record into the specified table.

        :param table_name: Name of the table.
        :param data: Dictionary of column names and their values.
        """
        columns = data.keys()
        values = tuple(data.values())
        query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.Placeholder() * len(values))
        )
        self._execute_query(query, f"Error inserting data into {table_name}", values)

    def get_columns(self, table_name: str) -> List[str]:
        """
        Get the list of columns in the specified table.

        :param table_name: Name of the table.
        :return: List of column names.
        """
        try:
            # Chuyển tên bảng sang chữ thường

            query = sql.SQL(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = {}"
            ).format(sql.Literal(table_name))
            self.cursor.execute(query)  # Sử dụng sql.Literal để tránh lỗi định dạng
            return self.cursor.fetchall()
        except Exception as e:
            self.connection.rollback()  # Rollback giao dịch nếu có lỗi
            print(f"Error fetching columns for table '{table_name}': {e}")
            return []


    def read(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Union[Any, Tuple[str, Any]]]] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
        descending: bool = True
    ):
        """
        Read data from the specified table.

        :param table_name: Name of the table.
        :param columns: List of columns to fetch (default: all columns).
        :param conditions: Dict of conditions (value or (op, value)).
        :param limit: Maximum number of rows to fetch.
        :param order_by: Column name to sort by.
        :param descending: If True, use DESC; else ASC.
        :return: List of rows.
        """
        # SELECT … 
        columns_sql = (
            sql.SQL(", ").join(map(sql.Identifier, columns))
            if columns else sql.SQL("*")
        )
        query = sql.SQL("SELECT {} FROM {}").format(
            columns_sql, sql.Identifier(table_name)
        )

        # WHERE …
        params: Dict[str, Any] = {}
        if conditions:
            clauses = []
            for col, cond in conditions.items():
                if isinstance(cond, tuple) and len(cond) == 2:
                    op, value = cond
                else:
                    op, value = "=", cond
                clauses.append(
                    sql.Composed([
                        sql.Identifier(col),
                        sql.SQL(f" {op} "),
                        sql.Placeholder(col)
                    ])
                )
                params[col] = value
            query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses)

        # ORDER BY …
        if order_by:
            direction = sql.SQL("DESC") if descending else sql.SQL("ASC")
            query += sql.SQL(" ORDER BY {} ").format(sql.Identifier(order_by)) + direction

        # LIMIT …
        if limit:
            query += sql.SQL(" LIMIT {}").format(sql.Literal(limit))

        return self._fetch_query(
            query,
            f"Error reading data from {table_name}",
            params
        )

    def update(self, table_name: str, updates: Dict[str, any], conditions: Dict[str, any]):
        """
        Update records in the specified table.

        :param table_name: Name of the table.
        :param updates: Dictionary of columns to update and their new values.
        :param conditions: Dictionary of conditions for the WHERE clause.
        """
        set_clause = sql.SQL(", ").join(
            sql.Composed([sql.Identifier(k), sql.SQL(" = "), sql.Placeholder(f"set_{k}")]) for k in updates.keys()
        )
        where_clause = sql.SQL(" AND ").join(
            sql.Composed([sql.Identifier(k), sql.SQL(" = "), sql.Placeholder(f"where_{k}")]) for k in conditions.keys()
        )
        query = sql.SQL("UPDATE {} SET {} WHERE {}").format(
            sql.Identifier(table_name), set_clause, where_clause
        )
        params = {f"set_{k}": v for k, v in updates.items()}
        params.update({f"where_{k}": v for k, v in conditions.items()})
        self._execute_query(query, f"Error updating data in {table_name}", params)

    def delete(self, table_name: str, conditions: Dict[str, any]):
        """
        Delete records from the specified table.

        :param table_name: Name of the table.
        :param conditions: Dictionary of conditions for the WHERE clause.
        """
        where_clause = sql.SQL(" AND ").join(
            sql.Composed([sql.Identifier(k), sql.SQL(" = "), sql.Placeholder(k)]) for k in conditions.keys()
        )
        query = sql.SQL("DELETE FROM {} WHERE {}").format(sql.Identifier(table_name), where_clause)
        self._execute_query(query, f"Error deleting data from {table_name}", conditions)

    def _execute_query(self, query, error_message, params=None):
        """Execute query with logging."""
        try:
            logger.debug(f"Executing query: {query}")
            self.cursor.execute(query, params)
            self.connection.commit()
            logger.debug("Query executed successfully")
        except Exception as e:
            logger.error(f"{error_message}: {e}")
            self.connection.rollback()
            raise

    def _fetch_query(self, query, error_message, params=None):
        """
        Execute a query and fetch results.

        :param query: SQL query to execute.
        :param error_message: Error message to display if the query fails.
        :param params: Parameters for the query.
        :return: Query results.
        """
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except Exception as e:
            self.connection.rollback()
            print(f"{error_message}: {e}")
            return []

    def close(self):
        """
        Close the database connection.
        """
        self.cursor.close()
        self.connection.close()

    def batch_insert(self, table_name: str, tweets_data: list, column_mapping: dict):
        """
        Insert multiple records into a table based on column mapping.

        :param table_name: Name of the table.
        :param tweets_data: List of dictionaries containing tweet data.
        :param column_mapping: Mapping of input keys to DB columns.
        """
        if not tweets_data:
            logging.info("No data provided for batch insert.")
            return

        try:
            # Extract column names from mapping
            columns = list(column_mapping.values())

            # Prepare values
            values = [
                tuple(tweet[field] for field in column_mapping.keys())
                for tweet in tweets_data
            ]

            insert_query = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
                sql.Identifier(table_name),
                sql.SQL(", ").join(map(sql.Identifier, columns))
            )

            extras.execute_values(self.cursor, insert_query, values)
            self.connection.commit()
            logging.info(f"Successfully inserted {len(tweets_data)} records into {table_name}.")
        except Exception as e:
            self.connection.rollback()
            logging.error(f"Error inserting batch data into {table_name}: {e}")
            raise



class BlobStorage:
    """
    Simple wrapper for Azure Blob Storage operations.
    Usage:
        blob = BlobStorage()
        blob.create_container('raw')
        blob.upload_file('raw', 'tweets.json', '/path/to/tweets.json')
        data = blob.download_blob('raw', 'tweets.json')
    """
    def __init__(self, conn_str: str):
        self.service_client = BlobServiceClient.from_connection_string(conn_str=conn_str)

    def create_container(self, container_name: str):
        """
        Create a container if it does not exist.
        """
        container = self.service_client.get_container_client(container_name)
        try:
            container.create_container()
        except Exception:
            # Container may already exist
            pass
        return container
    
    def upload_file(self, container_name: str, blob_name: str, file_path: str, overwrite: bool = True):
        """
        Upload a local file to a blob with input validation and enhanced error handling.
        """
        try:
            # Kiểm tra nếu file cục bộ tồn tại
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Local file '{file_path}' does not exist.")

            # Lấy container client
            container = self.service_client.get_container_client(container_name)

            # Kiểm tra nếu container không tồn tại, tạo mới
            if not container.exists():
                container.create_container()
                print(f"Container '{container_name}' created.")

            # Lấy blob client
            blob = container.get_blob_client(blob_name)

            # Mở file và tải lên blob
            with open(file_path, 'rb') as data:
                blob.upload_blob(data, overwrite=overwrite)
            print(f"File '{file_path}' uploaded to blob '{blob_name}' in container '{container_name}'.")

            # Trả về URL của blob
            return blob.url

        except FileNotFoundError as fnf_error:
            print(fnf_error)
            raise
        except Exception as e:
            print(f"Error uploading file '{file_path}' to blob '{blob_name}': {e}")
            raise
    
    def upload_blob(self,container_name, blob_name, data, overwrite=False):
        """
        Upload data to Azure Blob Storage. It ensures the blob name is valid.

        Parameters:
        - blob_name: The name of the blob (including folder structure).
        - data: The data to be uploaded (bytes).
        - overwrite: Boolean flag to overwrite if the blob already exists.
        """
        # Remove any trailing slashes from blob name to avoid invalid resource name
        blob_name = blob_name.rstrip('/')
        
        # Check if the blob name is valid
        if any(c in blob_name for c in ['\\', ':', '*', '?', '"', '<', '>', '|']):
            raise ValueError(f"Blob name contains invalid characters: {blob_name}")
        
        container = self.service_client.get_container_client(container_name)
        blob_client = container.get_blob_client(blob_name)

        try:
            # Upload the data
            blob_client.upload_blob(data, overwrite=overwrite)
            print(f"Blob '{blob_name}' uploaded successfully.")
        except Exception as e:
            print(f"Error uploading blob '{blob_name}': {e}")

    def create_folder_structure(self, container_name, folder_prefix):
        """
        Ensure folder structure is created in the container by uploading dummy files.

        Parameters:
        - folder_prefix: The folder structure to create.
        """
        # Ensure the folder structure exists by uploading a dummy file
        dummy_data = b""  # Empty data for a dummy file
        
        # Upload an init file to represent the folder structure
        self.upload_blob(container_name, f"{folder_prefix}/.init", dummy_data, overwrite=True)

    def download_blob(self, container_name: str, blob_name: str) -> bytes:
        """
        Download blob content and return as bytes.
        """
        blob = self.service_client.get_blob_client(container=container_name, blob=blob_name)
        downloader = blob.download_blob()
        return downloader.readall()

    def download_file(self, container: str, blob: str, path: str) -> str:
        """Download blob and write directly to file (streaming)."""
        blob_client = self.service_client.get_blob_client(container, blob)
        with open(path, "wb") as f:
            blob_client.download_blob().readinto(f)
        return path
    
    def read_json_from_container(self, container: str, file_path: str):
        """Read a JSON file directly from container into RAM without saving to local."""
        # 1. Lấy blob client từ container và đường dẫn file
        blob_client = self.service_client.get_blob_client(container=container, blob=file_path)
        # 2. Download toàn bộ nội dung file vào RAM
        blob_data = blob_client.download_blob().readall()        
        # 3. Parse JSON từ bytes
        data = json.loads(blob_data.decode('utf-8'))  # phải decode từ bytes -> str trước

        return data

    def list_blobs_by_path(self, container: str, path: str = "") -> List[str]:
        """
        List blobs or pseudo-folders in a given path within a container.
        If path is empty, list top-level blobs/folders.
        """
        container_client = self.service_client.get_container_client(container)
        path = path.strip('/')
        if path:
            path += "/"

        # Dùng delimiter để liệt kê như thư mục
        blob_list = container_client.walk_blobs(name_starts_with=path, delimiter='/')
        return [blob.name for blob in blob_list]

    def delete_blob(self, container_name: str, blob_name: str):
        """
        Delete a blob from a container.
        """
        blob = self.service_client.get_blob_client(container=container_name, blob=blob_name)
        blob.delete_blob()

    def delete_container(self, container_name: str):
        """
        Delete a container and all its blobs.
        """
        container = self.service_client.get_container_client(container_name)
        container.delete_container()

    def upload_folder(
        self,
        container_name: str,
        folder_path: str,
        blob_prefix: str = "",
        overwrite: bool = True
    ) -> None:
        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a directory: {folder_path}")
        container = self.service_client.get_container_client(container_name)
        if not container.exists():
            container.create_container()
        for root, _, files in os.walk(folder_path):
            for fn in files:
                local_file = os.path.join(root, fn)
                rel = os.path.relpath(local_file, folder_path)
                blob_name = f"{blob_prefix}/{rel}".replace("\\","/")
                with open(local_file, "rb") as data:
                    container.get_blob_client(blob_name).upload_blob(data, overwrite=overwrite)

    def copy_blob(
        self,
        container_name: str,
        source_prefix: str,
        dest_prefix: str,
        overwrite: bool = True
    ) -> None:
        container = self.service_client.get_container_client(container_name)
        src = source_prefix.rstrip('/')
        dst = dest_prefix.rstrip('/')
        # Copy folder
        blobs = list(container.list_blobs(name_starts_with=src + '/'))
        if blobs:
            for blob in blobs:
                rel = blob.name[len(src)+1:]
                container.get_blob_client(f"{dst}/{rel}").start_copy_from_url(
                    container.get_blob_client(blob.name).url
                )
        else:
            container.get_blob_client(dst).start_copy_from_url(
                container.get_blob_client(src).url
            )

    def close(self):
        self.service_client.close()


