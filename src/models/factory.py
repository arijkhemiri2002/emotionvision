"""Model factory: baseline CNN (from scratch) + pretrained backbones (timm).

Every model ends with Global Average Pooling + one Linear layer, so Grad-CAM can
later be computed from the ONNX model without gradients (CAM = W_c . feature maps).
"""

from __future__ import annotations

import timm
import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Small VGG-style CNN trained from scratch (reference point for transfer learning)."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.4):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(self.dropout(x))


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "baseline_cnn":
        return BaselineCNN(num_classes)
    if name in {"efficientnet_b0", "resnet50", "convnext_tiny"}:
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=0.3)
    raise ValueError(f"Unknown model '{name}'")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classification head (warm-up phase)."""
    head = model.get_classifier() if hasattr(model, "get_classifier") else model.classifier
    head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable
