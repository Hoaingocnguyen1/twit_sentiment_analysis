import logging
from datetime import datetime as dt, timedelta
import asyncio
import numpy as np
import sys

sys.path.append("/opt/airflow")
from src.data import TwikitClient, TweetCache
from src.data import crawl_tweet

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def crawl_task(base_dir, hashtags_dict, output_file, **context):
    try:
        if not base_dir:
            raise ValueError("Base directory is not provided.")
        if not hashtags_dict:
            raise ValueError("Hashtags dictionary is not provided.")
        if not output_file:
            raise ValueError("Output file name is not provided.")

        client = TwikitClient()
        tweet_cache = TweetCache()

        # Crawl dữ liệu
        tweets_by_topic = asyncio.run(crawl_tweet(client, hashtags_dict))

        now = dt.now()
        current_hour = now.strftime("%Y%m%d_%H")

        seen_texts = set()
        unique_tweet_list = []

        for topic, tag_tweets in tweets_by_topic.items():
            logging.info(
                f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Crawled {len(tag_tweets)} tweets for topic: {topic}"
            )

            for tweet in tag_tweets:
                if tweet.text.strip() not in seen_texts:
                    seen_texts.add(tweet.text.strip())
                    unique_tweet_list.append(
                        {
                            "text": tweet.text.strip(),
                            "date_created": tweet.created_at_datetime.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "date_scrape": now.strftime("%Y-%m-%d %H:%M:%S"),
                            "topic": topic,
                        }
                    )

        tweet_cache.add(unique_tweet_list)
        raw_file = tweet_cache.save_to_file(base_dir, output_file)
        logging.info(
            f"Data saved for hour {output_file} with {len(unique_tweet_list)} unique tweets"
        )
        return raw_file

    except Exception as e:
        logging.error(f"Error during tweet crawling: {e}")
        raise
