"""Evaluation helpers: predictions, metrics, confusion matrix."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402


@torch.no_grad()
def predict(model, loader, device, criterion=None):
    model.eval()
    y_true, y_pred, total_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * x.size(0)
            n += x.size(0)
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
    loss = total_loss / n if n else float("nan")
    return np.concatenate(y_true), np.concatenate(y_pred), loss


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def report(y_true, y_pred, classes) -> str:
    return classification_report(y_true, y_pred, labels=range(len(classes)),
                                 target_names=classes, digits=3, zero_division=0)


def plot_confusion_matrix(y_true, y_pred, classes, path: str, title: str = "") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize="true")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title or "Normalized confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
