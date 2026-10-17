"""Reusable evaluation harness: one place that computes accuracy, F1, and a
confusion matrix, so every model trained from Week 6 onward (optical-only,
SAR-only, fused) gets compared the exact same way.

Metric formulas and what they mean are explained in
Learning_Notes/Project_Explained.docx section 7.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate_predictions(y_true, y_pred, labels: list[str]) -> dict:
    """Accuracy, macro F1 (each class weighted equally, fair when classes
    are imbalanced), and weighted F1 (each class weighted by how common it
    is), plus the raw confusion matrix.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
