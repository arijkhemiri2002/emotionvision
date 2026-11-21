"""Class Activation Maps from the ONNX model outputs (no gradients needed).

For a network ending with Global Average Pooling + Linear, Grad-CAM reduces to CAM:
    Grad-CAM weights  a_k^c = mean(dy_c / dA_k) = W[c, k] / (H*W)
so  Grad-CAM_c = ReLU(sum_k W[c,k] * A_k) / (H*W)  ==  CAM_c up to a constant.
The exported ONNX graph therefore returns the CAMs of all classes directly.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)
    rng = cam.max() - cam.min()
    return (cam - cam.min()) / rng if rng > 1e-8 else np.zeros_like(cam)


def cam_for_class(cams: np.ndarray, class_idx: int, size: tuple[int, int],
                  mode: str = "contrastive") -> np.ndarray:
    """cams: (n_classes, h, w) from the ONNX output -> normalized (H, W) heatmap in [0, 1].

    mode="raw"         : CAM of the class logit y_c (classic CAM / Grad-CAM).
    mode="contrastive" : CAM of (y_c - mean_k y_k). Softmax ignores anything shared by all
                         logits, so regions that raise every class equally (e.g. border /
                         padding activations of the 4x4 map) cancel out and only the
                         class-discriminative regions remain.
    """
    cams = cams.astype(np.float32)
    cam = cams[class_idx] - cams.mean(axis=0) if mode == "contrastive" else cams[class_idx]
    cam = normalize_cam(cam)
    return cv2.resize(cam, (size[1], size[0]), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """image_rgb: (H, W, 3) uint8, heatmap: (H, W) in [0, 1] -> blended uint8 RGB image."""
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, colored, alpha, 0)
