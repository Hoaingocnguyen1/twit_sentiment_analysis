import logging
import asyncio
import numpy as np
from .schemas import TwitterQuery

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def crawl_tweet(client, hashtags_dict: dict):
    """
    Crawl tweets for multiple topics and their associated hashtags.

    :param client: TwikitClient instance.
    :param hashtags_dict: Dictionary where keys are topics and values are lists of hashtags.
    :return: Dictionary with topics as keys and crawled tweets as values.
    """
    await client.login(mode='cookies')

    tweets_by_topic = dict()

    for topic, hashtags in hashtags_dict.items():
        logging.info(f"Starting to crawl tweets for topic: {topic}")
        topic_tweets = []

        # Shuffle hashtags to randomize the order
        tags = np.random.permutation(hashtags)

        for tag in tags:
            tq = TwitterQuery(
                query=tag,
                mode='Latest', 
            )
            try:
                tag_tweets = await client.get_tweets(tq)
                topic_tweets.extend([tweet for tweet in tag_tweets])

                await asyncio.sleep(15)  # async sleep để tránh lỗi
            except Exception as e:
                logging.error(f"Error crawling {tag} for topic {topic}: {e}")
                continue

        # Store tweets for the current topic
        tweets_by_topic[topic] = topic_tweets
        logging.info(f"Finished crawling {len(topic_tweets)} tweets for topic: {topic}")

    return tweets_by_topic
