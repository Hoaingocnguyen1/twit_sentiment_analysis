import os
import logging
from typing import Dict, Any, Optional, List
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

from .eval import compute_metrics, plot_confusion_matrix_image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


##chỈ CÓ TRAIN PREDICT INPUT LIST TEXT LABEL 0 1 2 -> OUTPUT TƯƠNG TỰ
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
        scheduler_type: str = "linear", 
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
                eval_metrics = self.evaluate(eval_dataset, batch_size)
                logger.info({k: v for k, v in eval_metrics.items() if k != 'confusion_matrix_fig'})
                confusion_matrix_fig = eval_metrics.get('confusion_matrix_fig')

        return {
            "final_train_loss": avg_loss,
            "eval_metrics": {k: v for k, v in eval_metrics.items() if k != 'confusion_matrix_fig'} if eval_metrics else None,
            "confusion_matrix": confusion_matrix_fig
        }

    def evaluate(self, dataset: Dataset, batch_size) -> Dict[str, Any]:
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
    
    def predict_batch(self, texts: List[str]) -> List[int]:
        """
        Nhận vào list các chuỗi `texts`, trả về list các index label có xác suất cao nhất.
        Ví dụ với 2 câu: ["...", "..."] sẽ return [2, 1]
        """
        assert self.model is not None, "Model not loaded"
        self.model.eval()
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
