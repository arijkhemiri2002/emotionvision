"""DVC stage `benchmark`: PyTorch FP32 vs ONNX Runtime FP32 vs ONNX Runtime INT8 (CPU).

Measures on the test set: accuracy, macro-F1, model size, latency (batch 1) and
throughput (batch 32). INT8 = static quantization (QDQ, per-channel) calibrated on
validation images.

Output:
    models/emotionvision_int8.onnx
    reports/benchmark.json   (DVC metrics)
    reports/benchmark.md     (table for README / report)

Usage:
    python -m src.models.benchmark
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch.utils.data import DataLoader, Subset

from src.data.dataset import build_loaders
from src.evaluation.metrics import compute_metrics
from src.models.factory import build_model

MODELS_DIR = Path("models")
FP32_PATH = MODELS_DIR / "emotionvision.onnx"
INT8_PATH = MODELS_DIR / "emotionvision_int8.onnx"


class ValCalibrationReader(CalibrationDataReader):
    def __init__(self, batches: list[np.ndarray]):
        self.batches, self.i = batches, 0

    def get_next(self):
        if self.i >= len(self.batches):
            return None
        self.i += 1
        return {"input": self.batches[self.i - 1]}


# Quantization configs compared on the VALIDATION set; the best one is kept.
#   ops=None -> quantize every supported op (incl. SiLU/Sigmoid/Mul of the SE blocks)
#   ops=[...] -> only these op types (activations/SE gates stay in float)
QUANT_CONFIGS = {
    "minmax_all":        dict(method=CalibrationMethod.MinMax, ops=None, exclude_ends=False),
    "minmax_conv":       dict(method=CalibrationMethod.MinMax, ops=["Conv", "Gemm", "MatMul"], exclude_ends=False),
    "percentile_conv":   dict(method=CalibrationMethod.Percentile, ops=["Conv", "Gemm", "MatMul"], exclude_ends=False),
    "percentile_conv_x": dict(method=CalibrationMethod.Percentile, ops=["Conv", "Gemm", "MatMul"], exclude_ends=True),
}


def calibration_batches(val_ds, n_calib: int, seed: int, bs: int = 32) -> list[np.ndarray]:
    """Random (class-mixed) subset of the validation set.
    NB: ImageFolder is sorted by class, so taking the first N images would calibrate on one class only."""
    idx = np.random.default_rng(seed).permutation(len(val_ds))[:n_calib].tolist()
    loader = DataLoader(Subset(val_ds, idx), batch_size=bs)
    return [x.numpy() for x, _ in loader]


def end_nodes(model_path: Path) -> list[str]:
    """First Conv (sees raw pixels) + classifier Gemm/MatMul: most sensitive to INT8."""
    import onnx

    nodes = onnx.load(str(model_path)).graph.node
    first_conv = next(n.name for n in nodes if n.op_type == "Conv")
    head = [n.name for n in nodes if n.op_type in ("Gemm", "MatMul")]
    return [first_conv, *head]


def quantize_int8(val_ds, val_loader, seed: int, n_calib: int = 512) -> tuple[str, dict]:
    pre = MODELS_DIR / "_pre.onnx"
    quant_pre_process(str(FP32_PATH), str(pre))
    batches = calibration_batches(val_ds, n_calib, seed)
    # Histogram methods keep every activation in RAM: use a smaller calibration set
    small = calibration_batches(val_ds, 128, seed, bs=16)
    ablation, best_name, best_f1 = {}, None, -1.0
    for name, c in QUANT_CONFIGS.items():
        out = MODELS_DIR / f"_q_{name}.onnx"
        quantize_static(
            str(pre), str(out),
            ValCalibrationReader(batches if c["method"] == CalibrationMethod.MinMax else small),
            quant_format=QuantFormat.QDQ, per_channel=True,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            calibrate_method=c["method"], op_types_to_quantize=c["ops"],
            nodes_to_exclude=end_nodes(pre) if c["exclude_ends"] else None,
            extra_options={"CalibMovingAverage": True},
        )
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        m = evaluate(lambda x: sess.run(["logits"], {"input": x})[0], val_loader)
        ablation[name] = {"val_accuracy": round(m["accuracy"], 4), "val_macro_f1": round(m["macro_f1"], 4),
                          "size_mb": round(out.stat().st_size / 1e6, 2)}
        print(f"  [int8 {name:18s}] val_acc={m['accuracy']:.4f} val_macro_f1={m['macro_f1']:.4f}")
        if m["macro_f1"] > best_f1:
            best_name, best_f1 = name, m["macro_f1"]
    shutil.copy(MODELS_DIR / f"_q_{best_name}.onnx", INT8_PATH)
    for f in MODELS_DIR.glob("_q_*.onnx"):
        f.unlink()
    pre.unlink(missing_ok=True)
    print(f"  -> kept '{best_name}' as {INT8_PATH}")
    return best_name, ablation


def evaluate(predict_fn, loader) -> dict[str, float]:
    y_true, y_pred = [], []
    for x, y in loader:
        y_pred.append(predict_fn(x.numpy()).argmax(1))
        y_true.append(y.numpy())
    return compute_metrics(np.concatenate(y_true), np.concatenate(y_pred))


def timing(predict_fn, input_size: int, batch: int, runs: int, warmup: int = 10) -> np.ndarray:
    x = np.random.rand(batch, 3, input_size, input_size).astype(np.float32)
    for _ in range(warmup):
        predict_fn(x)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        predict_fn(x)
        t.append((time.perf_counter() - t0) * 1000)
    return np.array(t)


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    cfg = json.loads((MODELS_DIR / "model_config.json").read_text())
    s, classes = cfg["input_size"], cfg["classes"]

    loaders, dsets = build_loaders(params["data"]["processed_dir"], classes, s, batch_size=64, num_workers=0)
    print("[quantize] static INT8: comparing calibration configs on the validation set...")
    best_cfg, ablation = quantize_int8(dsets["val"], loaders["val"], params["seed"])

    # --- predictors -------------------------------------------------------
    torch.set_num_threads(os.cpu_count() or 4)
    model = build_model(cfg["model"], len(classes), pretrained=False)
    model.load_state_dict(torch.load(MODELS_DIR / "champion.pt", map_location="cpu"))
    model.eval()

    def torch_predict(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(x)).numpy()

    def onnx_predictor(path: Path):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return lambda x: sess.run(["logits"], {"input": x})[0]

    variants = {
        "pytorch_fp32": (torch_predict, MODELS_DIR / "champion.pt"),
        "onnx_fp32": (onnx_predictor(FP32_PATH), FP32_PATH),
        "onnx_int8": (onnx_predictor(INT8_PATH), INT8_PATH),
    }

    results = {}
    for name, (fn, path) in variants.items():
        print(f"[bench] {name}")
        m = evaluate(fn, loaders["test"])
        lat = timing(fn, s, batch=1, runs=200)
        thr = timing(fn, s, batch=32, runs=20)
        results[name] = {
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "test_accuracy": round(m["accuracy"], 4),
            "test_macro_f1": round(m["macro_f1"], 4),
            "latency_ms_mean": round(float(lat.mean()), 2),
            "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 2),
            "throughput_img_s": round(32 / (thr.mean() / 1000), 1),
        }

    base = results["pytorch_fp32"]["latency_ms_mean"]
    for r in results.values():
        r["speedup_vs_pytorch"] = round(base / r["latency_ms_mean"], 2)

    Path("reports").mkdir(exist_ok=True)
    results["onnx_int8"]["quant_config"] = best_cfg
    Path("reports/benchmark.json").write_text(json.dumps(results, indent=2))
    Path("reports/quantization_ablation.json").write_text(json.dumps(ablation, indent=2))

    cols = ["size_mb", "test_accuracy", "test_macro_f1", "latency_ms_p50",
            "latency_ms_p95", "throughput_img_s", "speedup_vs_pytorch"]
    lines = ["| variant | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    lines += [f"| {n} | " + " | ".join(str(r[c]) for c in cols) + " |" for n, r in results.items()]
    table = "\n".join(lines)
    abl = ["| INT8 config | val_accuracy | val_macro_f1 |", "|---|---|---|"]
    abl += [f"| {n} | {a['val_accuracy']} | {a['val_macro_f1']} |" for n, a in ablation.items()]
    Path("reports/benchmark.md").write_text(
        f"CPU benchmark ({cfg['model']}, input {s}x{s})\n\n{table}\n\n"
        f"INT8 quantization ablation (validation set, kept: {best_cfg})\n\n" + "\n".join(abl) + "\n")
    print("\n" + table)
    print(f"\nINT8 ablation (val, kept: {best_cfg})\n" + "\n".join(abl))

    if os.getenv("MLFLOW_TRACKING_URI"):
        import mlflow
        mlflow.set_experiment("emotionvision")
        with mlflow.start_run(run_name=f"benchmark_{cfg['model']}"):
            mlflow.log_param("model", cfg["model"])
            for n, r in results.items():
                mlflow.log_metrics({f"{n}_{k}": v for k, v in r.items() if isinstance(v, (int, float))})
            mlflow.log_param("int8_quant_config", best_cfg)
            mlflow.log_artifact("reports/benchmark.md")


if __name__ == "__main__":
    main()
