import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import mlflow.transformers
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    default_data_collator,
    get_scheduler,
    PreTrainedModel,
    PreTrainedTokenizer
)
from torch.optim import AdamW
from torch.nn.functional import softmax
import mlflow
import json

from .eval import compute_metrics, plot_confusion_matrix_image
from src.database.baseDatabase import BlobStorage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelManager:
    def __init__(
        self,
        model_path: str,
        finetune: bool = False,
        output_dir: str = "artifact/models",
        model_name: Optional[str] = None,
        seed: int = 42,
        model_obj: Optional[PreTrainedModel] = None,
        tokenizer_obj: Optional[PreTrainedTokenizer] = None
    ):
        """
        Initialize model manager.
        Args:
            model_path (str): HuggingFace name or local path
            finetune (bool): if True, initialize with `num_labels=3`
            output_dir (str): local directory to save artifacts
            model_name (str): alias for experiment/blob
            seed (int): random seed for reproducibility
        """
        self.model_path = model_path
        self.finetune = finetune
        self.output_dir = output_dir
        self.model_name = model_name or os.path.basename(model_path.strip("/"))

        # Reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

         # If preloaded, use them; otherwise load from pretrained
        if tokenizer_obj is not None and model_obj is not None:
            self.tokenizer = tokenizer_obj
            self.model = model_obj
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Metrics/report functions
        self.metrics_fn = compute_metrics
        self.report_fn = plot_confusion_matrix_image

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-05,
        scheduler_type: str = "linear"
    ) -> Dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None, "Model not loaded"
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=default_data_collator
        )
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)

                outputs = self.model(**inputs)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

            if eval_dataset:
                eval_metrics = self.evaluate(eval_dataset)
                logger.info({k: v for k, v in eval_metrics.items() if k != 'confusion_matrix_fig'})
                confusion_matrix_fig = eval_metrics.get('confusion_matrix_fig')

        return {
            "final_train_loss": avg_loss,
            "eval_metrics": {k: v for k, v in eval_metrics.items() if k != 'confusion_matrix_fig'} if eval_metrics else None,
            "confusion_matrix": confusion_matrix_fig
        }

    def evaluate(self, dataset: Dataset,batch_size) -> Dict[str, Any]:
        assert self.model is not None, "Model not loaded"
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=default_data_collator)
        self.model.eval()

        all_preds, all_labels = [], []
        total_loss = 0.0
        loss_fn = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)

                outputs = self.model(**inputs)
                loss = loss_fn(outputs.logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(loader)
        metrics = self.metrics_fn(all_labels, all_preds)
        fig = self.report_fn(all_labels, all_preds)

        return {**metrics, "loss": avg_loss, "confusion_matrix_fig": fig}

    def predict(self, text: str) -> Dict[str, float]:
        assert self.model is not None, "Model not loaded"
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = softmax(outputs.logits[0], dim=0).cpu().numpy()

        config = AutoConfig.from_pretrained(self.model.config.name_or_path)
        ranking = np.argsort(probs)[::-1]
        return {config.id2label[i]: float(round(probs[i], 4)) for i in ranking}

    def save_model(
        self,
        blob_storage: Optional[BlobStorage] = None,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        cm_fig: Optional[Any] = None
    ) -> None:
        """
        Save model, tokenizer, and additional artifacts (metrics, confusion matrix) to local output_dir,
        then upload artifacts to Blob Storage under versioned folder and update 'latest' pointer.
        """
        assert self.model is not None and self.tokenizer is not None, "Model not initialized"
        try:
            # Local save of model and tokenizer (includes config.json)
            os.makedirs(self.output_dir, exist_ok=True)
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"Model and tokenizer saved locally at {self.output_dir}")

            # Save additional artifacts
            if metrics:
                metrics_path = os.path.join(self.output_dir, "metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Metrics saved at {metrics_path}")

            if cm_fig:
                fig_path = os.path.join(self.output_dir, "confusion_matrix.png")
                cm_fig.savefig(fig_path)
                logger.info(f"Confusion matrix figure saved at {fig_path}")

            # Upload to blob storage if provided
            if blob_storage:
                # Create appropriate paths for blob storage
                tag = version or "latest"
                
                # Get the relative path from the artifact directory
                if os.path.normpath(self.output_dir).startswith("artifact/"):
                    # If output_dir is already under "artifact/" directory
                    relative_path = os.path.relpath(self.output_dir, "artifact")
                else:
                    # Otherwise, just use the model name
                    relative_path = os.path.join("models", self.model_name, tag)
                
                # Ensure the blob path is properly formatted
                blob_prefix = relative_path.replace("\\", "/")
                
                # Upload directory to blob storage
                blob_storage.upload_folder(
                    container_name="artifact",
                    folder_path=self.output_dir,
                    blob_prefix=blob_prefix,
                    overwrite=True  # Set to True to ensure we can update files
                )
                logger.info(f"Artifacts uploaded to Blob under {blob_prefix}")

                # If a specific version is provided, also update the 'latest' pointer
                if version:
                    source_prefix = blob_prefix
                    # Calculate the destination path for the 'latest' version
                    # Remove the tag from the path and replace with 'latest'
                    if tag in source_prefix:
                        dest_prefix = source_prefix.replace(tag, "latest")
                    else:
                        # Fallback if tag is not in the path
                        dest_prefix = os.path.join("models", self.model_name, "latest").replace("\\", "/")
                    
                    # Ensure both paths end with "/"
                    if not source_prefix.endswith("/"):
                        source_prefix += "/"
                    if not dest_prefix.endswith("/"):
                        dest_prefix += "/"
                    
                    # Copy the current version to become the 'latest'
                    blob_storage.copy_blob(
                        container_name="artifact",
                        source_prefix=source_prefix,
                        dest_prefix=dest_prefix,
                        overwrite=True
                    )
                    logger.info(f"Blob 'latest' updated to version: {tag}")

        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            raise
    
    # Add a convenience method to make calling save_model easier
    def save(self, **kwargs):
        """Convenience method that calls save_model with the same parameters"""
        return self.save_model(**kwargs)
    
    def predict_batch(self, texts: List[str]) -> List[int]:
        """
        Nhận vào list các chuỗi `texts`, trả về list các index label có xác suất cao nhất.
        Ví dụ với 2 câu: ["...", "..."] sẽ return [2, 1]
        """
        # Tokenize + chạy model giống hệt predict_batch nhưng không cần build dict
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # logits shape: (batch_size, num_labels)
            probs = softmax(outputs.logits, dim=-1).cpu().numpy()

        # np.argmax lấy index của giá trị lớn nhất trên mỗi hàng (mỗi mẫu)
        top_indices = np.argmax(probs, axis=1)
        return top_indices.tolist()
