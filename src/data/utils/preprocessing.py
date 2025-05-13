from typing import List
import re
from config.constants import TWITTER_HASHTAGS
import string
from typing import List, Optional

def clean_data_v1(raw: List[str] = None):
    if not raw:
        return 
    #text = [t.text for t in raw]

    # Remove all emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U00002600-\U000026FF"  # Misc symbols
        "\U000025A0-\U000025FF"  # Geometric Shapes
        "]+",
        flags=re.UNICODE
    )

    cleaned_tweets = [re.sub(emoji_pattern, '', t) for t in raw]

    # Remove the hashtags that's in crawling list
    for tag in TWITTER_HASHTAGS:
        cleaned_tweets = [t.replace(tag, '') for t in cleaned_tweets]

    # Remove newlines
    cleaned_tweets = [t.replace('\n', ' ').replace('\r', ' ') for t in cleaned_tweets]

    # # Remove all non-English tweets, using fasttext
    # cleaned_tweets = [t[0] for t in detect_language(cleaned_tweets, keep_non_english=False)]

    # Remove links
    cleaned_tweets = [re.sub(r'https://t.co/\w+', '', t) for t in cleaned_tweets]

    # Remove trailing whitespaces
    cleaned_tweets = [t.strip() for t in cleaned_tweets]

    # Remove empty
    cleaned_tweets = [t for t in cleaned_tweets if t != '']

    return cleaned_tweets

def clean_data_v2(raw: Optional[List[str]] = None, hashtags: Optional[List[str]] = None) -> List[str]:
    if not raw:
        return []
    
    if hashtags is None:
        hashtags = TWITTER_HASHTAGS

    # Regex patterns
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "\U000025A0-\U000025FF"
        "]+",
        flags=re.UNICODE
    )
    url_pattern = re.compile(r'https?://\S+')

    # Punctuation to remove
    punctuation_table = str.maketrans('', '', string.punctuation)

    cleaned = []
    for text in raw:
        if not text:
            continue

        # Lowercase
        text = text.lower()

        # Remove emojis
        text = emoji_pattern.sub('', text)

        # Remove hashtags
        for tag in hashtags:
            text = text.replace(tag.lower(), '')  # lowercase để chắc chắn match

        # Remove links
        text = url_pattern.sub('', text)

        # Remove newlines, carriage returns
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text)

        # Remove punctuation
        text = text.translate(punctuation_table)

        # Remove leading/trailing spaces
        text = text.strip()

        if text:
            cleaned.append(text)

    return cleaned

    
