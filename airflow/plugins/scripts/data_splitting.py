from sklearn.model_selection import train_test_split
import logging


class DataSplitter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def split_data(self, labeled_data):
        try:
            texts = labeled_data['texts']
            labels = labeled_data['labels']
            
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, 
                test_size=0.2, 
                random_state=42,
                stratify=labels)
            self.logger.info(f"Chia train/test thành công: train={len(X_train)}, test={len(X_test)}")
            self._log_split_distribution(y_train, y_test)
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi trong split_data: {e}")
            raise
    
    def _log_split_distribution(self, y_train_list, y_test_list):
        from collections import Counter
        
        train_dist = Counter(y_train_list)
        test_dist = Counter(y_test_list)
        
        self.logger.info(f"Train distribution: {dict(train_dist)}")
        self.logger.info(f"Test distribution: {dict(test_dist)}")
    
    def validate_split(self, split_data):
        try:
            train_size = len(split_data['X_train'])
            test_size = len(split_data['X_test'])
            if train_size == 0 or test_size == 0:
                self.logger.error("Một trong hai tập train/test bị rỗng")
                return False
            if (len(split_data['X_train']) != len(split_data['y_train']) or
                len(split_data['X_test']) != len(split_data['y_test'])):
                self.logger.error("Số lượng texts và labels không khớp")
                return False
            
            self.logger.info("Validation passed cho split data")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi trong validate_split: {e}")
            return False