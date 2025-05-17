# # sentiment_prediction.py
# import os
# import sys
# from pathlib import Path
# import pandas as pd
# import logging
# import torch
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
# )

# logger = logging.getLogger(__name__)

# def predict_tweet_sentiment(
#     model_name='modernbert', 
#     project_root_path=None, 
#     staging_table="TWEET_STAGING", 
#     product_table="TWEET_PRODUCT",
#     batch_size=32
# ):
#     """
#     Function to load the sentiment model, retrieve unpublished tweets from
#     staging table, predict sentiment, and insert results into product table.
    
#     Args:
#         model_name (str): Name of the model to load
#         project_root_path (str): Path to project root, defaults to Path.cwd().parent
#         staging_table (str): Name of the staging table
#         product_table (str): Name of the product table
#         batch_size (int): Batch size for predictions
        
#     Returns:
#         int: Number of records processed
#     """
#     # Setup project path
#     if project_root_path:
#         project_root = Path(project_root_path)
#     else:
#         project_root = Path.cwd().parent  # if cwd is project/notebooks
        
#     sys.path.append(str(project_root))
#     BASE_DIR = project_root
    
#     logger.info(f"Project root: {project_root}")
#     logger.info(f"BASE_DIR: {BASE_DIR}")
    
#     # Import dependent modules after path setup
#     from src.models.transformer_manager import ModelManager
#     from config.dataClient import get_db
    
#     # Model configuration
#     model_path = f"{BASE_DIR}/mlflow/artifact/models/{model_name}"
#     model_dir = Path(model_path).resolve()
    
#     try:
#         logger.info(f"Loading model from: {model_dir}")
        
#         # Load model and tokenizer
#         device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#         logger.info(f"Using device: {device}")
        
#         model_obj = AutoModelForSequenceClassification.from_pretrained(
#             model_dir,
#             local_files_only=True,
#             device_map=device
#         )
        
#         tokenizer_obj = AutoTokenizer.from_pretrained(
#             model_dir,
#             local_files_only=True,
#             device_map=device
#         )
        
#         # Initialize model manager
#         manager = ModelManager(
#             model_name=model_name,
#             model_path=model_dir,
#             finetune=False,
#             model_obj=model_obj,
#             tokenizer_obj=tokenizer_obj
#         )
        
#         # Get database connection
#         db = get_db()
        
#         # Step 1: Get tweets that haven't been processed
#         rows = db.read(
#             table_name=staging_table,
#             columns=["tweet_id", "topic", "content", "created_at"],
#             conditions={"moved_to_product": False}
#         )
        
#         if not rows:
#             logger.info("No new records to process.")
#             return 0
        
#         logger.info(f"Processing {len(rows)} tweets")
        
#         # Step 2: Extract contents and predict sentiments in batches
#         all_predictions = []
#         for i in range(0, len(rows), batch_size):
#             batch = rows[i:i+batch_size]
#             contents = [row[2] for row in batch]
            
#             # Run prediction on batch
#             batch_predictions = manager.predict_batch(contents)
#             all_predictions.extend(batch_predictions)
            
#             logger.info(f"Processed batch {i//batch_size + 1}/{(len(rows)-1)//batch_size + 1}")
        
#         # Step 3: Prepare data for insertion
#         data_to_insert = [
#             {
#                 "staging_tweet_id": row[0],
#                 "topic": row[1],
#                 "predicted_sentiment": pred,
#                 "created_at": row[3]
#             }
#             for row, pred in zip(rows, all_predictions)
#         ]
        
#         # Step 4: Insert into product table
#         column_mapping = {
#             "staging_tweet_id": "staging_tweet_id",
#             "topic": "topic",
#             "predicted_sentiment": "predicted_sentiment",
#             "created_at": "created_at"
#         }
        
#         logger.info("Inserting prediction results into product table")
#         db.batch_insert(product_table, data_to_insert, column_mapping)
        
#         # Step 5: Update staging records to mark as processed
#         tweet_ids = [row[0] for row in rows]
#         for tweet_id in tweet_ids:
#             db.update(
#                 table_name=staging_table,
#                 data={"moved_to_product": True},
#                 conditions={"tweet_id": tweet_id}
#             )
        
#         logger.info(f"Successfully processed and inserted {len(rows)} records")
#         return len(rows)
        
#     except Exception as e:
#         logger.error(f"Error in sentiment prediction: {str(e)}")
#         raise

# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
#     predict_tweet_sentiment()