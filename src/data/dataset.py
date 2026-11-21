"""Datasets, transforms and class weights for FER+ PNG folders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(input_size: int, train: bool) -> transforms.Compose:
    """Grayscale 48x48 -> 3-channel `input_size` tensor (needed for ImageNet backbones)."""
    common_end = [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((input_size, input_size)),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            *common_end,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((input_size, input_size)),
        *common_end,
    ])


def build_loaders(processed_dir: str, classes: list[str], input_size: int,
                  batch_size: int, num_workers: int = 2):
    root = Path(processed_dir)
    loaders, dsets = {}, {}
    for split in ("train", "val", "test"):
        ds = datasets.ImageFolder(root / split, transform=build_transforms(input_size, split == "train"))
        # ImageFolder sorts classes alphabetically: force the order from params.yaml
        assert sorted(classes) == ds.classes, f"Class mismatch in {split}: {ds.classes}"
        remap = {ds.class_to_idx[c]: i for i, c in enumerate(classes)}
        ds.samples = [(p, remap[t]) for p, t in ds.samples]
        ds.targets = [t for _, t in ds.samples]
        ds.classes, ds.class_to_idx = list(classes), {c: i for i, c in enumerate(classes)}
        dsets[split] = ds
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
    return loaders, dsets


def class_weights(targets: list[int], n_classes: int, mode: str = "sqrt_inv") -> torch.Tensor | None:
    """Loss weights against imbalance. `sqrt_inv` softens the inverse-frequency weights
    (plain `inv` gives disgust a weight of ~30, which makes training unstable)."""
    if mode == "none":
        return None
    counts = np.bincount(targets, minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * counts)
    if mode == "sqrt_inv":
        w = np.sqrt(w)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)
