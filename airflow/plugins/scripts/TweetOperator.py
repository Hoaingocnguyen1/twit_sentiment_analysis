from datetime import datetime, timedelta
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import logging
import traceback
from scripts.data_extraction import DataExtractor
from scripts.data_labeling import DataLabeler
from scripts.data_splitting import DataSplitter
from scripts.data_storage import DataStorage

class ExtractDataOperator(BaseOperator):
    """
    Operator để lấy dữ liệu từ database
    """
    
    @apply_defaults
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def execute(self, context):
        logger = logging.getLogger(__name__)
        
        try:
            extractor = DataExtractor()
            data_dict = extractor.extract_tweets()
            
            if data_dict is None:
                logger.warning("Không có tweet mới trong vòng 7 ngày.")
                return None
            
            logger.info(f"Extracted {len(data_dict['texts'])} tweets successfully")
            return data_dict
            
        except Exception as e:
            logger.error(f"Lỗi trong ExtractDataOperator: {e}")
            logger.error(traceback.format_exc())
            raise


class LabelTweetOperator(BaseOperator):
    @apply_defaults
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def execute(self, context):
        logger = logging.getLogger(__name__)
        data_dict = context['task_instance'].xcom_pull(task_ids='extract_data_from_db')
        
        if data_dict is None:
            logger.warning("Không có dữ liệu để gán nhãn")
            return None
            
        try:
            labeler = DataLabeler()
            labeled_data = labeler.label_tweets(data_dict)
            
            logger.info(f"Labeled {len(labeled_data['texts'])} tweets successfully")
            return labeled_data
            
        except Exception as e:
            logger.error(f"Lỗi trong LabelTweetOperator: {e}")
            logger.error(traceback.format_exc())
            raise


class SplitTrainTestOperator(BaseOperator):
    """
    Operator để chia dữ liệu thành train/test
    """
    
    @apply_defaults
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def execute(self, context):
        logger = logging.getLogger(__name__)
        
        # Lấy dữ liệu từ task trước
        labeled_data = context['task_instance'].xcom_pull(task_ids='label_tweet_data')
        
        if labeled_data is None:
            logger.warning("Không có dữ liệu được gán nhãn")
            return None
            
        try:
            splitter = DataSplitter()
            split_data = splitter.split_data(labeled_data)
            
            logger.info(f"Split data successfully: train={len(split_data['X_train'])}, test={len(split_data['X_test'])}")
            return split_data
            
        except Exception as e:
            logger.error(f"Lỗi trong SplitTrainTestOperator: {e}")
            logger.error(traceback.format_exc())
            raise


class SaveDataToBlobOperator(BaseOperator):
    """
    Operator để lưu dữ liệu lên blob storage
    """
    
    @apply_defaults
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def execute(self, context):
        logger = logging.getLogger(__name__)
        
        # Lấy dữ liệu từ task trước
        split_data = context['task_instance'].xcom_pull(task_ids='split_train_test')
        
        if split_data is None:
            logger.warning("Không có dữ liệu để lưu")
            return False
            
        try:
            storage = DataStorage()
            success = storage.save_to_blob(split_data)
            
            if success:
                logger.info("Đã upload train và test datasets lên blob storage thành công.")
            
            return success
            
        except Exception as e:
            logger.error(f"Lỗi trong SaveDataToBlobOperator: {e}")
            logger.error(traceback.format_exc())
            raise