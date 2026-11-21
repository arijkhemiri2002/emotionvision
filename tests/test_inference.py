"""Inference engine helpers (preprocessing, face box)."""

import numpy as np

from api.inference import EmotionEngine


def test_square_box_is_clipped_to_image():
    x, y, w, h = EmotionEngine._square_box(x=5, y=5, bw=50, bh=80, w=100, h=100)
    assert x >= 0 and y >= 0
    assert x + w <= 100 and y + h <= 100
    assert w > 0 and h > 0


def test_preprocess_matches_model_input(model_files):
    eng = EmotionEngine(model_files["onnx"], model_files["config"], model_files["detector"])
    face = (np.random.default_rng(0).random((200, 150, 3)) * 255).astype(np.uint8)
    x, display = eng.preprocess(face)
    assert x.shape == (1, 3, 32, 32) and x.dtype == np.float32
    assert display.shape == (32, 32, 3)
    # grayscale replicated on 3 channels (before per-channel normalization)
    unnorm = x[0] * eng.std + eng.mean
    assert np.allclose(unnorm[0], unnorm[1], atol=1e-5) and np.allclose(unnorm[1], unnorm[2], atol=1e-5)
