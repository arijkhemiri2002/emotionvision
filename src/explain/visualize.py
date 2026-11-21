"""DVC stage `explain`: Grad-CAM examples on the test set with the ONNX model.

For each class, takes the first correctly classified test images and shows:
face | raw CAM (logit y_c) | contrastive CAM (y_c - mean_k y_k).

Output: reports/figures/gradcam_examples.png

Usage:
    python -m src.explain.visualize
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.dataset import build_transforms  # noqa: E402
from src.explain.cam import cam_for_class, overlay  # noqa: E402

PER_CLASS = 2
COLS = 3  # face, raw, contrastive
DISPLAY = 144


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    cfg = json.loads(Path("models/model_config.json").read_text())
    classes = cfg["classes"]
    tf = build_transforms(cfg["input_size"], train=False)
    sess = ort.InferenceSession("models/emotionvision.onnx", providers=["CPUExecutionProvider"])
    test_dir = Path(params["data"]["processed_dir"]) / "test"

    fig, axes = plt.subplots(len(classes), PER_CLASS * COLS, figsize=(PER_CLASS * COLS * 1.6, len(classes) * 1.75))
    for row, cls in enumerate(classes):
        shown = 0
        for img_path in sorted((test_dir / cls).glob("*.png")):
            img = Image.open(img_path).convert("RGB")
            logits, cams = sess.run(None, {"input": tf(img).unsqueeze(0).numpy()})
            pred = int(logits[0].argmax())
            if pred != row:
                continue
            face = np.array(img.resize((DISPLAY, DISPLAY)))
            raw = cam_for_class(cams[0], pred, (DISPLAY, DISPLAY), mode="raw")
            con = cam_for_class(cams[0], pred, (DISPLAY, DISPLAY), mode="contrastive")
            p = np.exp(logits[0] - logits[0].max())
            conf = p[pred] / p.sum()
            c0 = shown * COLS
            axes[row, c0].imshow(face)
            axes[row, c0].set_title(f"{cls} {conf:.0%}", fontsize=8)
            axes[row, c0 + 1].imshow(overlay(face, raw))
            axes[row, c0 + 2].imshow(overlay(face, con))
            if row == 0:
                axes[row, c0 + 1].set_title("raw CAM", fontsize=8)
                axes[row, c0 + 2].set_title("contrastive CAM", fontsize=8)
            shown += 1
            if shown == PER_CLASS:
                break
        axes[row, 0].set_ylabel(cls, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle("Grad-CAM from ONNX outputs: raw vs contrastive - correctly classified test faces", fontsize=10)
    plt.tight_layout()
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("reports/figures/gradcam_examples.png", dpi=150)
    print("[ok] reports/figures/gradcam_examples.png")


if __name__ == "__main__":
    main()
