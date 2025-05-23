import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import io
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer  # import tokenizer bạn dùng trong phần main

# Thêm root_dir vào sys.path để import modules nội bộ
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(root_dir)

from config.dataClient import get_db, get_blob_storage
from src.data.utils.label import label_data
from src.models.preprocess import preprocess_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_data_from_db(columns: Optional[List[str]] = None, limit: int = 30
                    ) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[List], Optional[List]]:
    """
    Lấy dữ liệu tweet mới 7 ngày gần đây chưa xử lý, gọi label_data để gán nhãn, chia train/test.
    """
    if columns is None:
        columns = ["tweet_id", "content"]
    try:
        db = get_db()
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')

        rows = db.read(
            table_name='TWEET_STAGING',
            columns=columns,
            conditions={"moved_to_product": False},
            where_clause=f"created_at >= TIMESTAMP '{seven_days_ago}'",
            limit=limit
        )

        logger.info(f"Số dòng đọc được từ DB: {len(rows)}")

        if not rows:
            logger.warning("Không có tweet mới trong vòng 7 ngày.")
            return None, None, None, None

        df = pd.DataFrame(rows, columns=columns)
        texts = df['content'].tolist()

        logger.info("Bắt đầu gán nhãn tweet bằng LLM...")
        labels = label_data(texts)

        # Chia train/test 80%/20%
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

        logger.info(f"Chia train/test thành công: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Lỗi trong get_data_from_db: {e}")
        logger.error(traceback.format_exc())
        return None, None, None, None


def save_train_test_to_blob_in_memory(
    X_train: List[str], X_test: List[str],
    y_train: List, y_test: List,
    container_name: str = "testartifact", prefix: str = "data/"
) -> bool:
    """
    Lưu 2 tập train/test dạng JSON lên blob storage.
    """
    try:
        blob = get_blob_storage()

        train_df = pd.DataFrame({"text": X_train, "sentiment": y_train})
        test_df = pd.DataFrame({"text": X_test, "sentiment": y_test})

        train_json = train_df.to_json(orient="records").encode('utf-8')
        test_json = test_df.to_json(orient="records").encode('utf-8')

        blob.upload_blob(container_name, prefix + "train_dataset.json", train_json, overwrite=True)
        blob.upload_blob(container_name, prefix + "test_dataset.json", test_json, overwrite=True)

        logger.info("Đã upload train và test datasets lên blob storage dưới dạng JSON.")
        return True

    except Exception as e:
        logger.error(f"Lỗi trong save_train_test_to_blob_in_memory: {e}")
        logger.error(traceback.format_exc())
        return False


def load_and_preprocess_from_blob(
    tokenizer, max_length: int = 128, label_map: Optional[dict] = None,
    container_name: str = "testartifact", prefix: str = "data/"
) -> Tuple[Optional[object], Optional[object]]:
    """
    Tải dữ liệu train/test từ blob, chuyển về DataFrame, tiền xử lý thành dataset cho model.
    """
    try:
        blob = get_blob_storage()

        train_json_str = blob.read_json_from_container(container_name, prefix + "train_dataset.json")
        test_json_str = blob.read_json_from_container(container_name, prefix + "test_dataset.json")

        logger.info("Tải dữ liệu JSON từ blob storage thành công.")

        # Kiểm tra dữ liệu trả về có phải list không, nếu không thì đọc JSON từ string
        if isinstance(train_json_str, list):
            train_df = pd.DataFrame(train_json_str)
        else:
            train_df = pd.read_json(io.StringIO(train_json_str), lines=True)

        if isinstance(test_json_str, list):
            test_df = pd.DataFrame(test_json_str)
        else:
            test_df = pd.read_json(io.StringIO(test_json_str), lines=True)

        logger.info(f"Đã parse dataframe - Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        X_train = train_df['text'].tolist()
        y_train = train_df['sentiment'].tolist()
        X_test = test_df['text'].tolist()
        y_test = test_df['sentiment'].tolist()

        logger.info(f"Trích xuất thành công lists - train texts: {len(X_train)}, train labels: {len(y_train)}")
        if X_train:
            logger.info(f"Sample text train: '{X_train[0][:50]}...'")
            logger.info(f"Sample label train: {y_train[0]}")

        # Tiền xử lý dữ liệu
        train_dataset = preprocess_data(tokenizer=tokenizer, texts=X_train, labels=y_train,
                                        max_length=max_length, label_map=label_map)
        test_dataset = preprocess_data(tokenizer=tokenizer, texts=X_test, labels=y_test,
                                       max_length=max_length, label_map=label_map)

        logger.info(f"Tiền xử lý thành công, train dataset size: {len(train_dataset)}, test dataset size: {len(test_dataset)}")
        return train_dataset, test_dataset

    except Exception as e:
        logger.error(f"Lỗi trong load_and_preprocess_from_blob: {e}")
        logger.error(traceback.format_exc())
        return None, None


if __name__ == "__main__":
    try:
        model_name = "answerdotai/ModernBERT-base"
        logger.info(f"Khởi tạo tokenizer từ model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Bắt đầu lấy và gán nhãn dữ liệu từ DB...")
        X_train, X_test, y_train, y_test = get_data_from_db()

        if X_train is None:
            logger.error("Lấy dữ liệu từ DB thất bại, thoát.")
            sys.exit(1)

        logger.info(f"Tổng số mẫu - train: {len(X_train)}, test: {len(X_test)}")

        logger.info("Lưu dữ liệu train/test lên blob storage...")
        if not save_train_test_to_blob_in_memory(X_train, X_test, y_train, y_test):
            logger.error("Lưu dữ liệu lên blob thất bại, thoát.")
            sys.exit(1)

        logger.info("Tải dữ liệu từ blob và tiền xử lý...")
        train_dataset, test_dataset = load_and_preprocess_from_blob(tokenizer)

        if train_dataset is None:
            logger.error("Tải hoặc tiền xử lý dữ liệu thất bại, thoát.")
            sys.exit(1)

        logger.info(f"Tạo dataset thành công: train size = {len(train_dataset)}, test size = {len(test_dataset)}")

#         # Hiển thị 1 sample để kiểm tra
#         sample_idx = 0
#         sample = train_dataset[sample_idx]
#         logger.info(f"Mẫu số {sample_idx} trong train dataset:")
#         logger.info(f"- Input IDs (10 đầu): {sample['input_ids'][:10]}...")
#         logger.info(f"- Attention Mask (10 đầu): {sample['attention_mask'][:10]}...")
#         logger.info(f"- Label: {sample['labels']}")

#         decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
#         logger.info(f"- Giải mã text: {decoded_text[:50]}...")

#         logger.info("Pipeline test hoàn tất thành công!")

    except Exception as e:
        logger.error(f"Lỗi trong quá trình chạy pipeline: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)