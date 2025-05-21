import logging
from typing import List, Tuple, Union, Optional
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
import torch
from collections import Counter # Import Counter for better label distribution logging

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

# Configure logging for this module.
# Note: logging.basicConfig() should typically be called once at the application's entry point.
# If this module is part of a larger application, you might want to remove or comment out
# logging.basicConfig(level=logging.DEBUG) here and configure it in your main script.
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG) # Consider moving this to your main application entry point

class SentimentDataset(Dataset):
    """
    A custom PyTorch Dataset for sentiment analysis.
    Encodes text using a Hugging Face tokenizer and prepares it for model input.
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer: PreTrainedTokenizer, max_length: int = 128):
        """
        Initializes the SentimentDataset.

        Args:
            texts (List[str]): List of text strings.
            labels (List[int]): List of integer labels corresponding to the texts.
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer to encode texts.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single sample (text and label) by index and processes it.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids', 'attention_mask' (both as flattened PyTorch tensors),
                  and 'labels' (as a PyTorch long tensor).
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Encode the text using the tokenizer
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,    # Add special tokens like [CLS], [SEP]
            max_length=self.max_length, # Pad/truncate to max_length
            padding='max_length',       # Pad to max_length
            truncation=True,            # Truncate if longer than max_length
            return_tensors='pt'         # Return PyTorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),          # Flatten to 1D tensor
            'attention_mask': encoding['attention_mask'].flatten(), # Flatten to 1D tensor
            'labels': torch.tensor(label, dtype=torch.long)        # Convert label to long tensor
        }


def preprocess_data(
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    labels: List[Union[int, str, float]],
    max_length: int = 128,
    label_map: Optional[dict] = None # Optional predefined mapping, defaults to None
) -> SentimentDataset:
    """
    Preprocesses the raw text and label data into a SentimentDataset.
    
    This function performs the following steps:
    1. Filters out samples with NaN or empty string labels.
    2. Determines the label mapping:
       - If `label_map` is provided, it uses that mapping.
       - If `label_map` is None and all labels are standard sentiment strings (e.g., "POSITIVE"),
         it automatically uses `COMMON_SENTIMENT_MAPPINGS['text']`.
       - Otherwise, it creates a new mapping from unique labels found in the data.
    3. Maps the raw labels to consecutive integer class indices (0 to num_classes-1).
    4. Creates and returns a SentimentDataset.
    
    Args:
        tokenizer (PreTrainedTokenizer): The Hugging Face tokenizer to use for text encoding.
        texts (List[str]): A list of raw text strings.
        labels (List[Union[int, str, float]]): A list of raw labels corresponding to the texts.
                                                Can contain integers, strings, or floats (e.g., NaN).
        max_length (int): The maximum sequence length for tokenization. Defaults to 128.
        label_map (dict, optional): An optional predefined mapping from raw labels to integer indices.
                                    If provided, this mapping will be used. If None, the function
                                    will attempt to auto-generate a mapping.
    
    Returns:
        SentimentDataset: A PyTorch Dataset containing the preprocessed texts and mapped integer labels.
    
    Raises:
        KeyError: If a label is found in `filtered_raw_labels` that is not present in the
                  `final_label_map` (either provided or auto-generated).
    """
    logger.info("Starting data preprocessing...")

    # Step 1: Filter out NaN or empty labels
    filtered_texts = []
    filtered_raw_labels = []
    
    for t, l in zip(texts, labels):
        # Check for float NaN (handles both numpy.nan and standard float('nan'))
        if isinstance(l, float) and (torch.isnan(torch.tensor(l)) or (l != l)):
            logger.warning(f"Skipping sample with NaN label: text='{t}'")
            continue
            
        # Skip empty strings as labels
        if isinstance(l, str) and not l.strip():
            logger.warning(f"Skipping sample with empty label: text='{t}'")
            continue
            
        filtered_texts.append(t)
        filtered_raw_labels.append(l)
    
    logger.info(f"Filtered {len(texts) - len(filtered_texts)} samples due to invalid labels.")

    # Step 2: Determine the final label mapping
    final_label_map = label_map 
    
    if final_label_map is None:
        # Attempt to auto-detect if labels match common sentiment strings
        if all(isinstance(lbl, str) and lbl.upper() in COMMON_SENTIMENT_MAPPINGS['text'] for lbl in filtered_raw_labels):
            final_label_map = {k.upper(): v for k, v in COMMON_SENTIMENT_MAPPINGS['text'].items()}
            logger.info(f"Automatically used COMMON_SENTIMENT_MAPPINGS for labels: {final_label_map}")
        else:
            # Otherwise, create a new label mapping from unique labels in the data
            # Sort unique labels by their string representation for consistent mapping order
            unique_labels = sorted(list(set(filtered_raw_labels)), key=str)
            final_label_map = {raw: idx for idx, raw in enumerate(unique_labels)}
            logger.info(f"Created new label mapping from unique data labels: {final_label_map}")
    else:
        logger.info(f"Using provided label mapping: {final_label_map}")

    # Step 3: Map the labels
    mapped_labels = []
    for l in filtered_raw_labels:
        try:
            mapped_labels.append(final_label_map[l])
        except KeyError:
            # If a label is not in the final_label_map, it's an unmappable label.
            # This indicates an issue with the provided label_map or unexpected data.
            logger.error(f"Found label '{l}' not in the final label mapping: {final_label_map}. Cannot map this sample.")
            raise # Re-raise the KeyError to signal an unmappable label

    # Log stats
    label_counts = Counter(mapped_labels)
    logger.info(
        f"Label stats - Total filtered samples: {len(filtered_texts)}, "
        f"Unique raw labels: {len(set(filtered_raw_labels))}, "
        f"Mapped classes: {len(set(mapped_labels))}, "
        f"Label distribution (mapped indices): {label_counts}"
    )
    
    logger.info("Data preprocessing completed.")
    return SentimentDataset(filtered_texts, mapped_labels, tokenizer, max_length)