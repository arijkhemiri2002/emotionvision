"""Data preparation: FER+ majority vote and class weights."""

import numpy as np
import pytest
import torch

from src.data.dataset import build_transforms, class_weights
from src.data.prepare import FERPLUS_COLS, majority_label

KEEP = [FERPLUS_COLS.index(c) for c in ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]]


def votes(**kw) -> np.ndarray:
    v = np.zeros(len(FERPLUS_COLS))
    for k, n in kw.items():
        v[FERPLUS_COLS.index(k)] = n
    return v


def test_clear_majority_is_kept():
    assert majority_label(votes(happiness=8, neutral=2), KEEP) == FERPLUS_COLS.index("happiness")


def test_single_votes_are_noise():
    # 6 happiness + 4 single votes -> after removing singles, happiness has 100%
    v = votes(happiness=6, neutral=1, sadness=1, anger=1, fear=1)
    assert majority_label(v, KEEP) == FERPLUS_COLS.index("happiness")


def test_no_clear_majority_is_dropped():
    assert majority_label(votes(happiness=5, surprise=5), KEEP) is None


@pytest.mark.parametrize("cls", ["contempt", "unknown", "NF"])
def test_excluded_classes_are_dropped(cls):
    assert majority_label(votes(**{cls: 9, "neutral": 1}), KEEP) is None


def test_class_weights_favor_rare_classes():
    targets = [0] * 900 + [1] * 100
    w_inv = class_weights(targets, 2, "inv")
    w_sqrt = class_weights(targets, 2, "sqrt_inv")
    assert w_inv[1] > w_inv[0] and w_sqrt[1] > w_sqrt[0]
    assert (w_sqrt[1] / w_sqrt[0]) < (w_inv[1] / w_inv[0])   # sqrt softens the ratio
    assert class_weights(targets, 2, "none") is None


def test_transforms_output_3_channels():
    from PIL import Image

    img = Image.fromarray((np.random.rand(48, 48) * 255).astype(np.uint8), mode="L")
    for train in (True, False):
        x = build_transforms(112, train)(img)
        assert isinstance(x, torch.Tensor) and x.shape == (3, 112, 112)
