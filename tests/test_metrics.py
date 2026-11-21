"""Evaluation metrics."""

import numpy as np

from src.evaluation.metrics import compute_metrics, plot_confusion_matrix, report


def test_metrics_perfect_and_macro_f1():
    y = np.array([0, 0, 1, 2])
    assert compute_metrics(y, y)["accuracy"] == 1.0
    m = compute_metrics(y, np.array([0, 0, 0, 2]))       # class 1 always missed
    assert m["accuracy"] == 0.75 and m["macro_f1"] < m["weighted_f1"]


def test_report_and_confusion_matrix(tmp_path):
    y, p, classes = np.array([0, 1, 1]), np.array([0, 1, 0]), ["a", "b"]
    assert "a" in report(y, p, classes)
    out = tmp_path / "cm.png"
    plot_confusion_matrix(y, p, classes, str(out))
    assert out.exists() and out.stat().st_size > 0
