import logging
from typing import List, Tuple, Union
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def preprocess_data(
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    labels: List[Union[int, str, float]],
    max_length: int = 128,
    label_map: dict = None
) -> SentimentDataset:
    """
    Preprocess the data.
    
    - Filters out NaN and invalid labels
    - Maps raw labels to consecutive integer class indices
    - Supports both numeric and string labels
    
    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use
        texts (List[str]): List of texts
        labels (List[Union[int, str, float]]): List of raw labels (can be int, float, or str)
        max_length (int): Maximum sequence length
        label_map (dict, optional): Predefined mapping from raw labels to indices.
            If None, will create one automatically from the data.
    
    Returns:
        SentimentDataset: Dataset with mapped labels from 0..num_classes-1
    """
    # Step 1: Filter out NaN or empty labels
    filtered_texts = []
    filtered_raw_labels = []
    
    for t, l in zip(texts, labels):
        # Skip NaN values
        if isinstance(l, float) and (torch.isnan(torch.tensor(l)) or l != l):
            logger.warning(f"Skipping sample with NaN label")
            continue
            
        # Skip empty strings
        if isinstance(l, str) and not l.strip():
            logger.warning(f"Skipping sample with empty label")
            continue
            
        filtered_texts.append(t)
        filtered_raw_labels.append(l)
    
    # Step 2: Create or use mapping from raw labels to class indices
    if label_map is None:
        # Create a new label mapping
        unique_labels = sorted(set(filtered_raw_labels))
        label_map = {raw: idx for idx, raw in enumerate(unique_labels)}
        logger.info(f"Created label mapping: {label_map}")
    
    # Step 3: Map the labels
    try:
        mapped_labels = [label_map[l] for l in filtered_raw_labels]
    except KeyError as e:
        # If we encounter a label not in the mapping
        unknown_label = str(e).strip("'")
        logger.error(f"Found label '{unknown_label}' not in label mapping: {label_map}")
        # Add the new label or use a default value
        if isinstance(unknown_label, str) and unknown_label.upper() in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
            # Common case: standard sentiment labels
            sentiment_map = {"POSITIVE": 2, "NEGATIVE": 0, "NEUTRAL": 1}
            suggested_mapping = {k.upper(): v for k, v in label_map.items()}
            for k, v in sentiment_map.items():
                if k not in suggested_mapping:
                    suggested_mapping[k] = v
            logger.info(f"Suggested complete mapping: {suggested_mapping}")
        raise

    # Log stats
    logger.info(
        f"Label stats - unique raw labels: {len(set(filtered_raw_labels))}, "
        f"mapped classes: {len(set(mapped_labels))}, "
        f"label distribution: {dict(list(zip(filtered_raw_labels, mapped_labels))[:5])}"
    )
    
    return SentimentDataset(filtered_texts, mapped_labels, tokenizer, max_length)


# Predefined common sentiment mappings
COMMON_SENTIMENT_MAPPINGS = {
    # String labels to index
    "text": {
        "NEGATIVE": 0,
        "NEUTRAL": 1,
        "POSITIVE": 2
    }
    # # For datasets that use -1, 0, 1
    # "numeric_3class": {
    #     -1: 0,  # -1 → 0 (negative)
    #     0: 1,   # 0 → 1 (neutral)
    #     1: 2    # 1 → 2 (positive)
    # },
    # # For datasets that use 1, 2, 3
    # "numeric_123": {
    #     1: 0,  # 1 → 0 (negative)
    #     2: 1,  # 2 → 1 (neutral)
    #     3: 2   # 3 → 2 (positive)
    # }
}