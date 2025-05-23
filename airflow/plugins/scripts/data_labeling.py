import os
import sys
import logging

# Thêm root_dir vào sys.path để import modules nội bộ
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(root_dir)

from src.data.utils.label import label_data


class DataLabeler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def label_tweets(self, data_dict):
        try:
            texts = data_dict['texts']
            tweet_ids = data_dict['tweet_ids']
            
            self.logger.info("Bắt đầu gán nhãn tweet bằng LLM...")
            labels = label_data(texts)
            
            self.logger.info(f"Đã gán nhãn thành công cho {len(labels)} tweets")
            
            return {
                'texts': texts,
                'labels': labels,
                'tweet_ids': tweet_ids
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi trong label_tweets: {e}")
            raise
    
    def validate_labels(self, labels):
        """
        Kiểm tra tính hợp lệ của các nhãn
        
        Args:
            labels (list): Danh sách các nhãn
            
        Returns:
            bool: True nếu tất cả nhãn hợp lệ
        """
        valid_labels = ['positive', 'negative', 'neutral']
        
        for label in labels:
            if label not in valid_labels:
                self.logger.warning(f"Nhãn không hợp lệ: {label}")
                return False
        
        return True
    
    def get_label_distribution(self, labels):
        """
        Thống kê phân bố các nhãn
        
        Args:
            labels (list): Danh sách các nhãn
            
        Returns:
            dict: Dictionary chứa thống kê phân bố
        """
        from collections import Counter
        
        distribution = Counter(labels)
        total = len(labels)
        
        result = {}
        for label, count in distribution.items():
            result[label] = {
                'count': count,
                'percentage': round((count / total) * 100, 2)
            }
        
        self.logger.info(f"Label distribution: {result}")
        return result