"""DVC stage `export`: champion model (MLflow registry) -> models/emotionvision.onnx

The ONNX graph has 2 outputs:
    logits : (B, 7)
    cams   : (B, 7, h, w)   class activation maps for every class (see src/explain/cam.py)

Usage:
    python -m src.models.export_onnx                    # champion from MLflow (DagsHub)
    python -m src.models.export_onnx --source local     # models/<model>_best.pt + models/model_config.json

Also writes models/champion.pt (checkpoint used by the benchmark).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from torch import nn

from src.models.factory import build_model

MODELS_DIR = Path("models")
ONNX_PATH = MODELS_DIR / "emotionvision.onnx"
CONFIG_PATH = MODELS_DIR / "model_config.json"
CKPT_PATH = MODELS_DIR / "champion.pt"


class LogitsAndCAM(nn.Module):
    """Wraps a GAP+Linear classifier so it returns (logits, CAMs of all classes)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        if hasattr(model, "forward_features"):          # timm models
            self.head = model.get_classifier()
        else:                                           # BaselineCNN
            self.head = model.classifier

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(x)
        return self.model.features(x)

    def forward(self, x: torch.Tensor):
        feats = self.features(x)                                   # (B, C, h, w)
        logits = self.head(feats.mean(dim=(2, 3)))                 # GAP + Linear
        cams = torch.einsum("kc,bchw->bkhw", self.head.weight, feats)
        return logits, cams


def fetch_champion() -> None:
    """Download checkpoint + config of the registered `champion` model from MLflow."""
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias("emotionvision", "champion")
    except Exception:  # registry without aliases -> tag fallback
        versions = client.search_model_versions("name='emotionvision'")
        mv = next(v for v in versions if v.tags.get("role") == "champion")
    print(f"[mlflow] champion = emotionvision v{mv.version} (run {mv.run_id})")

    tmp = Path(mlflow.artifacts.download_artifacts(run_id=mv.run_id, dst_path="models/_tmp"))
    shutil.copy(tmp / "model_config.json", CONFIG_PATH)
    cfg = json.loads(CONFIG_PATH.read_text())
    shutil.copy(tmp / f"{cfg['model']}_best.pt", CKPT_PATH)
    # Keep the champion's training figures for the report / Streamlit dashboard
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("confusion_matrix_*.png"):
        shutil.copy(f, Path("reports/figures") / f.name)
    for f in tmp.glob("classification_report_*.txt"):
        shutil.copy(f, Path("reports") / f.name)
    shutil.rmtree("models/_tmp", ignore_errors=True)


def export(opset: int) -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    model = build_model(cfg["model"], len(cfg["classes"]), pretrained=False)
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    wrapper = LogitsAndCAM(model).eval()

    s = cfg["input_size"]
    dummy = torch.randn(1, 3, s, s)
    kwargs = dict(
        input_names=["input"], output_names=["logits", "cams"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "cams": {0: "batch"}},
        opset_version=opset,
    )
    try:
        torch.onnx.export(wrapper, dummy, str(ONNX_PATH), dynamo=False, **kwargs)
    except TypeError:  # older torch without the `dynamo` argument
        torch.onnx.export(wrapper, dummy, str(ONNX_PATH), **kwargs)
    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))

    # Numerical check PyTorch vs ONNX Runtime on REAL validation faces
    # (random noise inputs are out of distribution and exaggerate float32 rounding).
    x = real_batch(cfg, n=64)
    with torch.no_grad():
        t_logits, _ = wrapper(torch.from_numpy(x))
    t_logits = t_logits.numpy()

    def run(level) -> tuple[np.ndarray, np.ndarray]:
        so = ort.SessionOptions()
        so.graph_optimization_level = level
        sess = ort.InferenceSession(str(ONNX_PATH), so, providers=["CPUExecutionProvider"])
        # small chunks: at 224 px the un-optimized graph needs large buffers (low-RAM PCs)
        outs = [sess.run(None, {"input": x[i:i + 8]}) for i in range(0, len(x), 8)]
        return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])

    o_logits, o_cams = run(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    n_logits, _ = run(ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

    def diag(o: np.ndarray, label: str) -> tuple[float, float]:
        prob_diff = float(np.abs(softmax(o) - softmax(t_logits)).max())
        agree = float((o.argmax(1) == t_logits.argmax(1)).mean())
        print(f"  [{label:13s}] max|logit diff|={np.abs(o - t_logits).max():.4f} "
              f"(max|logit|={np.abs(t_logits).max():.1f})  max|prob diff|={prob_diff:.5f}  "
              f"argmax agreement={agree:.1%}")
        return prob_diff, agree

    print("[check] PyTorch vs ONNX Runtime on 64 validation faces")
    prob_diff, agree = diag(o_logits, "ORT optimized")
    diag(n_logits, "ORT no-optim")
    assert agree >= 0.98 and prob_diff < 0.05, "ONNX predictions differ from PyTorch (see diagnostics above)"

    size_mb = ONNX_PATH.stat().st_size / 1e6
    print(f"[ok] {ONNX_PATH} ({size_mb:.1f} MB) | cams shape {o_cams.shape[1:]}")


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def real_batch(cfg: dict, n: int) -> np.ndarray:
    """n preprocessed faces from data/processed/val (same transform as evaluation)."""
    from PIL import Image

    from src.data.dataset import build_transforms

    params = yaml.safe_load(open("params.yaml"))
    tf = build_transforms(cfg["input_size"], train=False)
    paths = sorted((Path(params["data"]["processed_dir"]) / "val").rglob("*.png"))
    step = max(len(paths) // n, 1)
    imgs = [tf(Image.open(p).convert("RGB")).numpy() for p in paths[::step][:n]]
    return np.stack(imgs).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["mlflow", "local"], default="mlflow")
    args = p.parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    if args.source == "mlflow":
        fetch_champion()
    else:  # local checkpoint produced by src.training.train
        cfg = json.loads(CONFIG_PATH.read_text())
        shutil.copy(MODELS_DIR / f"{cfg['model']}_best.pt", CKPT_PATH)
    opset = yaml.safe_load(open("params.yaml"))["export"]["onnx_opset"]
    export(opset)


if __name__ == "__main__":
    main()
