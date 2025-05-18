from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import io
from typing import Dict, List, Any, Tuple

def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    """
    Tính toán các metrics cho đánh giá mô hình.
    
    Args:
        labels: List các nhãn thực tế
        preds: List các nhãn dự đoán
        
    Returns:
        Dict[str, float]: Dictionary chứa các metrics
    """
    # Xác định các labels tồn tại trong dataset
    unique_labels = sorted(list(set(labels)))
    
    # Tính toán accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Tính toán precision, recall, f1 với average=weighted để xử lý mất cân bằng nhãn
    precision = precision_score(labels, preds, average='weighted', zero_division=0, labels=unique_labels)
    recall = recall_score(labels, preds, average='weighted', zero_division=0, labels=unique_labels)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0, labels=unique_labels)
    
    # Tính toán precision, recall, f1 cho từng class (nếu có đủ các class)
    metrics_by_class = {}
    if len(unique_labels) > 1:
        for avg in ['macro', 'weighted']:
            metrics_by_class[f'precision_{avg}'] = precision_score(
                labels, preds, average=avg, zero_division=0, labels=unique_labels
            )
            metrics_by_class[f'recall_{avg}'] = recall_score(
                labels, preds, average=avg, zero_division=0, labels=unique_labels
            )
            metrics_by_class[f'f1_{avg}'] = f1_score(
                labels, preds, average=avg, zero_division=0, labels=unique_labels
            )
    
    # Kết hợp các metrics lại
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        **metrics_by_class
    }
    
    return metrics

def plot_confusion_matrix_image(labels: List[int], preds: List[int]) -> Any:
    """
    Vẽ confusion matrix và trả về đối tượng figure.
    
    Args:
        labels: List các nhãn thực tế
        preds: List các nhãn dự đoán
        
    Returns:
        matplotlib.figure.Figure: Đối tượng figure của confusion matrix
    """
    # Tính toán confusion matrix
    unique_labels = sorted(list(set(labels + preds)))
    cm = confusion_matrix(labels, preds, labels=unique_labels)
    
    # Tạo figure mới
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Vẽ heatmap
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    
    # Thêm labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Thêm giá trị vào từng ô
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), 
                    ha="center", va="center", 
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    # Thêm ticks
    num_classes = len(unique_labels)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels([str(i) for i in unique_labels])
    ax.set_yticklabels([str(i) for i in unique_labels])
    
    # Chỉnh sửa layout để hiển thị đầy đủ
    plt.tight_layout()
    
    return fig