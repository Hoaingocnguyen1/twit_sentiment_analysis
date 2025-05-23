import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import logging

# Thêm root_dir vào sys.path để import modules nội bộ
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(root_dir)

from config.dataClient import get_db


class DataExtractor:
    """
    Class để xử lý việc trích xuất dữ liệu từ database
    """
    
    def __init__(self, limit):
        self.logger = logging.getLogger(__name__)
        self.db = get_db()
        self.limit = limit
    
    def extract_tweets(self, limit):
        try:
            columns = ["tweet_id", "content"]
            seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')

            rows = self.db.read(
                table_name='TWEET_STAGING',
                columns=columns,
                conditions={"moved_to_product": False},
                where_clause=f"created_at >= TIMESTAMP '{seven_days_ago}'",
                limit=self.limit
            )

            self.logger.info(f"Số dòng đọc được từ DB: {len(rows)}")

            if not rows:
                self.logger.warning("Không có tweet mới trong vòng 7 ngày.")
                return None

            df = pd.DataFrame(rows, columns=columns)
            texts = df['content'].tolist()
            
            return {
                'texts': texts,
                'tweet_ids': df['tweet_id'].tolist()
            }

        except Exception as e:
            self.logger.error(f"Lỗi trong extract_tweets: {e}")
            raise
    
    def get_tweet_count(self, days=7):
        try:
            days_ago = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
            
            count = self.db.count(
                table_name='TWEET_STAGING',
                conditions={"moved_to_product": False},
                where_clause=f"created_at >= TIMESTAMP '{days_ago}'"
            )
            return count
            
        except Exception as e:
            self.logger.error(f"Lỗi trong get_tweet_count: {e}")
            raise