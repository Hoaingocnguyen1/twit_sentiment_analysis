import os
import sys
import pandas as pd
import logging

# Thêm root_dir vào sys.path để import modules nội bộ
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(root_dir)

from config.dataClient import get_blob_storage


class DataStorage:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.blob = get_blob_storage()
    
    def save_to_blob(self, split_data):
        try:
            container_name="artifact"
            prefix="data/"
            
            train_df = pd.DataFrame({
                "text": split_data['X_train'], 
                "sentiment": split_data['y_train']
            })
            test_df = pd.DataFrame({
                "text": split_data['X_test'], 
                "sentiment": split_data['y_test']
            })

            train_json = train_df.to_json(orient="records").encode('utf-8')
            test_json = test_df.to_json(orient="records").encode('utf-8')

            self.blob.upload_blob(
                container_name, 
                prefix + "train_dataset.json", 
                train_json, 
                overwrite=True
            )
            self.blob.upload_blob(
                container_name, 
                prefix + "test_dataset.json", 
                test_json, 
                overwrite=True
            )

            self.logger.info("Đã upload train và test datasets lên blob storage dưới dạng JSON.")
            return True

        except Exception as e:
            self.logger.error(f"Lỗi trong save_to_blob: {e}")
            raise
    
    def load_from_blob(self, dataset_type="train"):
        try:
            container_name="artifact"
            filename = f"{dataset_type}_dataset.json"
            data = self.blob.read_json_from_container(container_name, "data/" + filename)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                import io
                df = pd.read_json(io.StringIO(data), lines=True)
            
            self.logger.info(f"Đã tải {dataset_type} dataset từ blob storage, shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Lỗi trong load_from_blob: {e}")
            raise